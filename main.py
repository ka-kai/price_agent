import argparse
from copy import deepcopy
import datetime
import dotenv
import json
import logging
import os
from pathlib import Path
import pytz
import stable_baselines3 as sb3
import sys

# Environment variables
dotenv.load_dotenv(Path("config/local.env"))

# Relative imports
from model.feature_extractor import CustomCombinedExtractor
from model.double_dqn import DoubleDQN
from documentation import setup_and_documentation
from sim.policies import ProportionalPriceInflPerDay
from utils_price_agent import *


def train(config):
    # Initialize wandb
    initialize_wandb(config=config)

    # Training environment
    env = SimEnv(config=config, env_type="training")

    # Callbacks
    cb_list = []
    # Callback for progress on training environment
    cb_list.append(CustLogProgressCallback(output_path=config.output_path))
    # Save the model after every 50 episode
    cb_list.append(CustCheckpointCallback(save_freq=config.days_eps * config.K * 50,
                                          save_path=config.output_path,
                                          name_prefix="",
                                          save_replay_buffer=config.flag_rb))

    # Model parameters
    # Network architecture, see https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
    config.a["model_args"]["policy_kwargs"] = dict(net_arch=[config.a["hidden_size_linear"]] * config.a["num_layers_linear"],)
    config.a["model_args"]["policy_kwargs"]["features_extractor_class"] = CustomCombinedExtractor
    config.a["model_args"]["policy_kwargs"]["features_extractor_kwargs"] = config.a["feature_extractor_args"]
    # Learning rate schedule; overwrite constant with function
    if config.a["lr_schedule"] == "linear":
        config.a["model_args"]["learning_rate"] = linear_schedule(config.a["model_args"]["learning_rate"])

    # Model
    if config.a["algo"] == "DQN":
        model = sb3.DQN
    elif config.a["algo"] == "DoubleDQN":
        model = DoubleDQN
    if config.mode == "train":
        model = model(**config.a["model_args"], env=env)
    else:  # Resume training
        # Load the model; non-specified values remain the same as in saved model
        args_resume = {"path": config.path_pretrained_model / "model",
                       "env": env,
                       "learning_starts": 0,
                       "exploration_fraction": 0}
        model = model.load(**args_resume)
        model.load_replay_buffer(config.path_pretrained_model / "replay_buffer")
        logging.info(f"Loaded the model and replay buffer from {config.path_pretrained_model}. Replay buffer position: {model.replay_buffer.pos}")
        # Note that pos is the current position; it can be < replay buffer size even though the replay buffer is full
    logging.info(" Model architecture ".center(80, "*"))
    logging.info(model.policy)

    # Train
    logging.info(" Start training ".center(80, "*"))
    model.learn(total_timesteps=config.K * config.days_eps * config.n_eps, callback=cb_list)
    logging.info(" Training completed ".center(80, "*"))

    # Save
    logging.info("Saving model")
    model.save(config.output_path / "model")
    if config.flag_rb:
        model.save_replay_buffer(config.output_path / "replay_buffer")

    # Run the evaluation script
    c = deepcopy(config)
    c.path_pretrained_model = config.output_path
    c.flag_rb = False
    # Results directory for final evaluation
    c.output_path = config.output_path / "final_eval"
    os.makedirs(c.output_path, mode=0o700)
    # Run evaluation
    df, metrics = evaluate(c)
    # Log the evaluation metrics to wandb
    for key, value in metrics.items():
        wandb.run.summary[key + "_final"] = value
    logging.info("Logged the final evaluation metrics to wandb")

    # End wandb run
    wandb.finish()

    return


def evaluate(config):
    # Evaluation environment
    env = SimEnv(config=config, env_type="evaluation")
    env = sb3.common.vec_env.DummyVecEnv([lambda: env])

    # Determine the policy
    if config.policy == "rl":
        path = config.path_pretrained_model / f"model_{config.model}" if config.model is not None else config.path_pretrained_model / "model"
        if config.a["algo"] == "DQN":
            model = sb3.DQN
        elif config.a["algo"] == "DoubleDQN":
            model = DoubleDQN
        policy = model.load(path)
    elif config.policy == "prop_infl_per_day":
        policy = ProportionalPriceInflPerDay(p_infl=env.envs[0].p_infl_net.copy(), config=config)
    elif config.policy == "none":
        pass

    # Callbacks
    cb_log = CustLogCallback()

    # Reset the environment
    observations = env.reset()
    actions = [0]  # not None because we use float(locals["actions"]) in the log callback;
                   # list such that same callback function can be used for both rl and non-rl policies
    dones = False

    # Run the simulation
    logging.info(" Start evaluation ".center(80, "*"))
    while not dones:
        if config.policy == "rl":
            actions, _ = policy.predict(observations, deterministic=True)
        elif config.policy == "prop_infl_per_day":
            actions = [policy.predict(ts=env.envs[0].ts)]
        elif config.policy == "none":
            pass  # actions remains unchanged
        actions = np.array(actions)
        new_observations, rewards, dones, infos = env.step(actions)
        cb_log.log(locals())
        observations = new_observations
        if env.envs[0].ts.hour == 0 and env.envs[0].ts.minute == 0:
            logging.info(f"Current timestamp: {env.envs[0].ts}")
    logging.info(" Evaluation completed ".center(80, "*"))

    # Get the results
    df = pd.DataFrame(cb_log.get_data_as_df())

    # Save the results
    if config.start_date_eval != config.start_date_eval_cons:
        # First save the complete df
        df.to_csv(config.output_path / f"df_evaluation_full.csv", sep=";", index=True, header=True)
        # Only consider time after start_date_eval_cons
        start_date_eval_cons = pytz.timezone(config.tz).localize(
            datetime.datetime.strptime(config.start_date_eval_cons, "%Y-%m-%d")).replace(hour=0, minute=0)
        df = df[df["time"] >= start_date_eval_cons].reset_index(drop=True)
    df.to_csv(config.output_path / f"df_evaluation.csv", sep=";", index=True, header=True)

    # Compute and save the evaluation metrics
    metrics = get_eval_metrics(df=df)
    with open(config.output_path / "metrics.txt", "w") as file:
        file.write(json.dumps(metrics))

    # Plot the results
    if config.policy == "rl":
        plot_eval_results(config=config)

    return df, metrics


if __name__ == "__main__":
    if sys.gettrace() is None:  # not in debugging mode
        # Parse console inputs
        parser = argparse.ArgumentParser(description="Settings.")
        # See assert statements for the argument options
        parser.add_argument("--dev_ctrl", dest="dev_ctrl", action="store", type=str, required=True)  # external device controller
        parser.add_argument("--policy", dest="policy", action="store", type=str, required=True)  # pricing policy
        parser.add_argument("--mode", dest="mode", action="store", type=str, required=False)  # mode ("train", "resume", or "eval"; only applies to reinforcement learning)
        parser.add_argument("--path", dest="path_pretrained_model", action="store", type=str)  # path to the model (only applies to reinforcement learning)
        parser.add_argument("--model", dest="model", action="store", type=int, default=None)  # name of the model file: "model_<provided int>" (only applies to reinforcement learning)
        parser.add_argument("--scripts_prev", dest="scripts_prev", action="store", type=bool, default=False)  # if True, use Config and SimEnv from the training run
        parser.add_argument("--n_eps", dest="n_eps", action="store", type=int)  # number of episodes for training (only applies to reinforcement learning)
        parser.add_argument("--notes", dest="notes", action="store", type=str)  # notes that are saved with the wandb run
        parser.add_argument("--config", dest="config", action="store", type=str)  # settings different from config file; expected input: string with k1:v1,k2:v2,....
        args = parser.parse_args()
    else:
        args = None

    # Import SimEnv and Config; use current or training run files depending on argument scripts_prev
    path_config_and_env_gym_scripts = Path.cwd() if args is None or not args.scripts_prev else Path(args.path_pretrained_model) / "scripts"
    print(f"Importing Config and SimEnv from: {path_config_and_env_gym_scripts}\n")
    Config = import_class_from_script(script=str(path_config_and_env_gym_scripts / "config.py"), class_name="Config")
    SimEnv = import_class_from_package(script="env_gym", path_init=str(path_config_and_env_gym_scripts / "sim" / "__init__.py"), class_name="SimEnv")

    # Setup
    config = Config()
    config = setup_and_documentation(args=args, config=config)
    logging.info(f"Config incl. adjustments from --config (if applicable):\n{config.__dict__}\n")  # do not move this line before setup_and_documentation()

    # Training or evaluation
    if config.mode in ["train", "resume"]:
        train(config=config)
    elif config.mode == "eval":
        evaluate(config=config)

    logging.info(" Script completed ".center(80, "*"))
