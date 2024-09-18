import importlib
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
import sys
import wandb

import analysis.plots as pl
from analysis.analysis_utils import read_time_series_from_file


# STABLE BASELINES
def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    Based on: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html

    :param float initial_value: Initial learning rate.
    :return: Schedule that computes
      current learning rate depending on remaining progress
      progress_remaining = 1.0 - (num_timesteps / total_timesteps)
    """
    def func(progress_remaining):
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


# WANDB
def initialize_wandb(config, ext="", run_id=None):
    """
    Connect to wandb and initialize.
    """

    wandb.login()
    run = wandb.init(
        project=config.project,
        entity=config.entity,
        config=config.__dict__,
        group=config.run_name,
        job_type=config.mode,
        name=f"{config.run_name}_{config.mode}{ext}",
        notes=config.notes,
        dir=config.output_path,
        resume="allow",
        # From https://docs.wandb.ai/ref/python/init
        # If id is set with init(id="UNIQUE_ID") or WANDB_RUN_ID="UNIQUE_ID" and it is identical to a previous run,
        # wandb will automatically resume the run with that id. Otherwise, wandb will start a new run.
        id=run_id)

    return run


# CALLBACKS
class CustLogCallback(BaseCallback):
    """
    Custom callback for tracking info in a dataframe
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.cust_num_ts = 0  # own counter to be able to also use the same function during evaluation
        self.data = []

    def log(self, locals):
        # Separate function because evaluate_policy takes a callback FUNCTION and not the callback object itself (1.8.0)
        self.cust_num_ts += 1

        if isinstance(locals["infos"], list):
            locals["infos"] = locals["infos"][0]
        if isinstance(locals["actions"], list):
            locals["actions"] = locals["actions"][0]

        self.data.append({
            "step": self.cust_num_ts,
            "time": locals["infos"]["time"],
            "reward": float(locals["rewards"]),
            "action": float(locals["actions"]),
            "price_n": locals["infos"]["price_n"],
            "p_tot": locals["infos"]["p_tot"],
            "p_infl": locals["infos"]["p_infl"],
            "p_flx": locals["infos"]["p_flx"],
            "p_wh": locals["infos"]["p_wh"],
            "p_hp": locals["infos"]["p_hp"],
            "p_ev": locals["infos"]["p_ev"],
            **locals["infos"]["cost_components"]
        })

    def get_data_as_df(self):
        return pd.DataFrame.from_records(self.data, index="step")

    def _on_step(self) -> bool:
        # This method will be called by the model after each call to `env.step()`
        self.log(self.locals)

        return True


class CustLogProgressCallback(BaseCallback):
    """
    Custom callback for tracking the learning progress on wandb
    """

    def __init__(self, output_path, verbose=0):
        super().__init__(verbose)
        self._reset_eps_dict()  # data for the current episode
        self.dict_all_eps = {  # data with mean values in each episode
            "reward": [],
            "cost_1": [],
            "cost_2": [],
            "cost_3": [],
            "cost_4": [],
            "loss": []
        }
        self.output_path = output_path

    def _reset_eps_dict(self):
        self.dict_eps = {
            "reward": [],
            "cost_1": [],
            "cost_2": [],
            "cost_3": [],
            "cost_4": [],
            "loss": []
        }

    def _on_step(self) -> bool:
        self.dict_eps["reward"].extend(self.locals["rewards"])
        self.dict_eps["cost_1"].append(self.locals["infos"][0]["cost_components"]["cost_1"])
        self.dict_eps["cost_2"].append(self.locals["infos"][0]["cost_components"]["cost_2"])
        self.dict_eps["cost_3"].append(self.locals["infos"][0]["cost_components"]["cost_3"])
        self.dict_eps["cost_4"].append(self.locals["infos"][0]["cost_components"]["cost_4"])

        # Log the loss
        if "train/loss" in self.logger.name_to_value.keys():  # Only available once training has started;
            # it is the mean of the performed gradient steps; this does not depend on the number of environments.
            self.dict_eps["loss"].append(self.logger.name_to_value["train/loss"])
        else:
            self.dict_eps["loss"].append(np.nan)

        if self.locals["dones"][0]:  # End of the episode
            # Append the mean values to the dict_all_eps
            for k in self.dict_eps.keys():
                mean_val = np.mean(self.dict_eps[k])
                self.dict_all_eps[k].append(mean_val)
                wandb.log({k: mean_val}, step=self.num_timesteps)
            # Reset the dict_eps
            self._reset_eps_dict()

        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        Save the results in dict_all_eps to a csv file.
        """

        df = pd.DataFrame(self.dict_all_eps)
        df.to_csv(self.output_path / f"df_progress.csv", sep=";", header=True)
        logging.info("Training results saved")

        return True


class CustCheckpointCallback(BaseCallback):
    """
    230421 KK: based on https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py

    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    By default, it only saves model checkpoints,
    you need to pass ``save_replay_buffer=True``,
    and ``save_vecnormalize=True`` to also save replay buffer checkpoints
    and normalization statistics checkpoints.
    .. warning::
      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``
    :param save_freq: Save checkpoints every ``save_freq`` call of the callback.
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param save_replay_buffer: Save the model replay buffer
    :param save_vecnormalize: Save the ``VecNormalize`` statistics
    :param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.
        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """

        # Return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}")
        # 230421 KK: adjusted
        return os.path.join(self.save_path, f"{self.name_prefix}{checkpoint_type}.{extension}")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(f"model_{int(self.n_calls / self.save_freq)}", extension="zip")  # KK: "model_{int(self.n_calls / self.save_freq)}" added
            self.model.save(model_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                # If model has a replay buffer, save it too
                replay_buffer_path = self._checkpoint_path("replay_buffer", extension="pkl")  # KK: underscore removed
                self.model.save_replay_buffer(replay_buffer_path)
                if self.verbose > 1:
                    print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                # Save the VecNormalize statistics
                vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
                self.model.get_vec_normalize_env().save(vec_normalize_path)
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

        return True


# OTHER
def get_eval_metrics(df):
    """
    Compute the evaluation metrics for the results from the evaluation run.

    :param df: df with the results from the evaluation run
    """
    # Metrics other than cost components
    metrics = {
        "sum_p_tot_sq_eval": f"{round(sum(df['p_tot'] ** 2), 2):e}",
        "max_p_tot_eval": round(max(abs(df["p_tot"])), 2),
    }

    # Cost components
    for col_name in df.columns:
        if "cost" in col_name:
            metrics[f"{col_name}_eval"] = round(sum(df[col_name]), 2)
    logging.info("Computed the evaluation metrics: %s", metrics)

    return metrics


def plot_eval_results(config):
    """
    Plot the results from the evaluation run to get some first insights.

    :param config: configuration object
    :return:
    """
    # Settings
    path_res = config.output_path.parents[1]  # overall results directory
    file_res = "df_evaluation.csv"  # file with the evaluation results
    path_res_rl = config.output_path  # results directory of the evaluation run
    dict_res = {}
    colors_single_days = ["red", "blue"]  # which colors to use for: inflexible load, (no control case), RL agent
    if config.dir_prop is not None:
        dict_res["prop"] = {"dir": config.dir_prop, "label": "Proportional price", "color": "bronze"}
    if config.dir_no_control is not None:
        dict_res["none"] = {"dir": config.dir_no_control, "label": "No control", "color": "green80"}
        colors_single_days.insert(1, "green80")
    dict_res["rl"] = {"dir": f"{path_res_rl.parent.name}/{path_res_rl.name}", "label": "RL agent", "color": "blue"}
    colors_single_days = [pl.utils.dict_colors[c] for c in colors_single_days]

    # Reformat to dictionary with dataframes, needed for some plot functions
    dict_data = {}
    for k, v in dict_res.items():
        df = read_time_series_from_file(path_res / v["dir"] / file_res, tz=config.tz_local)
        dict_data[v["label"]] = (df, pl.dict_colors[v["color"]])
    # Add inflexible load at the beginning of the dictionary
    df_infl = dict_data["RL agent"][0].copy().loc[:, ["p_infl"]].rename(columns={"p_infl": "p_tot"})
    dict_data = {"$P^{\mathrm{infl}}$": (df_infl, pl.dict_colors["red"]), **dict_data}

    # Plot extreme days; each plot also includes the previous and the following day
    with plt.rc_context(pl.params.single_days):
        _ = pl.plot_single_days(path_res=path_res,
                                dict_res=dict_res,
                                path_output=path_res_rl,
                                tz_plots=config.tz_local,
                                flag_cust_res=True,
                                list_colors=colors_single_days,
                                lw_legend_frame=0.2)

    # Plot duration curves
    with plt.rc_context(pl.params.duration):
        _ = pl.plot_multiple_duration_curves(dict_data=dict_data,
                                             col="p_tot",
                                             path_output=path_res_rl / f"{path_res_rl.parent.name}_duration.png",
                                             **config.args_duration_curves)

    # Plot mean daily profiles
    list_cols = [["p_tot", "$P^{\mathrm{tot}}$"],
                 ["p_wh", "$P^{\mathrm{EWH}}$"],
                 ["p_hp", "$P^{\mathrm{HP}}$"],
                 ["p_ev", "$P^{\mathrm{EV}}$"]]
    with plt.rc_context(pl.params.mean_daily):
        _ = pl.plot_mean_daily(dict_data=dict_data,
                               list_cols=list_cols,
                               fontsize_ylabel=7,
                               alpha=0.9,
                               args_legend={"loc": "center",
                                            "bbox_to_anchor": (0.45, 0.98),
                                            "ncol": 4,
                                            "columnspacing": 0.64,
                                            "borderpad": 0.3
                                            },
                               figsize=(2.15, 1.1),
                               path_output=path_res_rl / f"{path_res_rl.parent.name}_mean_daily.png"
                               )

    # Heatmap plots
    list_cols = ["p_tot", "price_n", "p_infl", "p_wh", "p_hp", "p_ev"]
    with plt.rc_context(pl.params.heatmap):
        for col in list_cols:
            fig = pl.HeatmapFigure(series=dict_data["RL agent"][0].loc[:, col],
                                   figsize=(3, 2),
                                   flag_histx=False,
                                   flag_histy=False,
                                   cbar_label="",
                                   annotate_suntimes=False,
                                   )
            # Adjustments
            fig.ax_heatmap.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            fig.ax_heatmap.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            fig.ax_heatmap.xaxis.set_minor_locator(mdates.MonthLocator())
            fig.savefig(path_res_rl / f"{path_res_rl.parent.name}_heatmap_{col}.png")

    return


def import_class_from_script(script, class_name):
    # Load the script as a module
    spec = importlib.util.spec_from_file_location(script, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the class from the module
    c = getattr(module, class_name)

    return c


def import_class_from_package(script, path_init, class_name):
    spec = importlib.util.spec_from_file_location(script, path_init)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # Get the class from the module
    c = getattr(module, class_name)

    return c
