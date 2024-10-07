# Dynamic Grid Tariffs for Power Peak Reduction Using Reinforcement Learning

**Author:** Katharina Kaiser, Power Systems Laboratory, ETH Zurich: <kkaiser@ethz.ch>

This repository contains the Python code for the [paper](https://ieeexplore.ieee.org/document/10694665):

> *K. Kaiser and G. Hug, "Dynamic Grid Tariffs for Power Peak Reduction Using Reinforcement Learning," in 2024 International Conference on
Smart Energy Systems and Technologies (SEST), 2024.*

If you make use of code in this repository, please reference the following citation:
```bibtex
@INPROCEEDINGS{Kaiser10694665,
  title={Dynamic Grid Tariffs for Power Peak Reduction Using Reinforcement Learning}, 
  booktitle={2024 International Conference on Smart Energy Systems and Technologies (SEST)},
  author={Kaiser, Katharina and Hug, Gabriela},
  year={2024},
  doi={10.1109/SEST61601.2024.10694665}
}
```

---

## Installation

The code has been tested with Python 3.10.11. Create a virtual environment, activate it, and install the required packages using:

```sh
pip install -r requirements.txt
```

It is recommended to train on a GPU. Check the PyTorch website for installation details.
If needed, you can also find the full conda environment file `environment.yml`; However, it may include packages that are not required.

---
## Usage

### Preliminaries
- Create a separate directory for the results and specify the path as `PATH_SIM_RESULTS` in `config/local.env`.
- In case you want to track the results on <https://wandb.ai/>, activate the correct virtual environment, and login using `wandb login`.
  Add the project name and your username to `config/local.env` and set `WANDB_MODE` to `online`.
  In case you do not want to track the results, set `WANDB_MODE` in `config/local.env` to `disabled`.

### Training
1) Ensure that the configurations in `config/config.yaml` and `config/args_doubledqn.yaml` are set correctly
2) Start the terminal with the correct environment and navigate to the `main.py` directory
3) Run `python main.py --mode=train --dev_ctrl=dynamic --policy=rl --n_eps=<number of episodes> --notes="<notes for easier reference>"`; see `main.py` for further arguments

The agent will then be trained for the specified number of episodes, and the final agent will be evaluated.
The results are saved in a subdirectory of `PATH_SIM_RESULTS`.
The following files are saved:
- a copy of the current version of the config files and scripts used for training
- the final evaluation results; `df_evaluation.csv` stores the overall results, while `df_whhps.csv` and `df_evs.csv` are the results of the individual flexible devices; if `start_date_eval_cons` is different from `start_date_eval` (see `config.yaml`), i.e., only part of the simulation period is considered for the evaluation, we additionally save `df_evaluation_full.csv` with the results for the entire simulation period.
- `log_file.log` contains the training logs
- `metrics.txt` contains some key metrics
- `df_progress.csv` contains the mean reward, loss, and cost components for each episode
- `model.zip` is the final model, while `model_1.zip`, etc. are intermediate models saved every 50 episodes (see `CustCheckpointCallback` in `main.py`)
- `replay_buffer.pkl` is the replay buffer corresponding to the latest model; it is only saved if `flag_rb` is set to True in `config.yaml`
- several plots for the final evaluation results

To use different settings than in `config.yaml`, pass `--config=k1:v1,k2:v2,....` where k denotes the parameter name and v the value.
This is implemented this way such that multiple runs with different settings can be started in parallel without changing the config file.

To test different settings, you can also run a wandb sweep, see the instructions in `config/config_sweep.yaml`.
The resulting plots can be summarized using `analysis/plots_to_ppt.py`.

### Evaluation
1) Ensure that the configurations in `config/config.yaml` and `config/args_doubledqn.yaml` are set correctly
2) Start the terminal with the correct environment and navigate to the `main.py` directory
3) Depending on what you want to evaluate, run one of the following commands: 
   - No control: `python main.py --mode=eval --dev_ctrl=none --policy=none`
   - Proportional price: `python main.py --mode=eval --dev_ctrl=dynamic --policy=prop_infl_per_day`
   - RL agent: `python main.py --mode=eval --dev_ctrl=dynamic --policy=rl --path=<path to the results directory, which contains the desired model.zip>`

For the no control and the proportional price case, a new directory will be created in `PATH_SIM_RESULTS` and the results will be saved there.
For the RL agent, the results will be saved in the same directory as the model. There is the option to import `Config` and `SimEnv` from the scripts folder of the training run, see argument `scripts_prev` in `main.py`.

### Time convention
Timestamps correspond to the start of a time interval, i.e., timestamp 00:00:00 corresponds to 00:00:00 - 00:15:00 (and not 23:45:00 - 00:00:00).

### Plots
The plots that were included in the SEST 2024 publication were created using `analysis/analyze_cases.ipynb`.

### Branches
- `individual`: The devices are assigned to customers, and each customer is simulated separately.
  In the current implementation, it is assumed that all customers have all 3 devices (electric water heater, heat pump, electric vehicle).
  Add `if` statements in `sim/env_gym` > `_sim_ts` if this does not apply.
- `vectorized`: All devices of a given type are simulated at once. Further improvements are required to reach intended speedup. 

---

## Input Files
All input data is stored in the `sim_data` directory:

**General**
- `msd_<start date>_<end date (included)>_<number of previous weeks considered>_<time window considered in a loop>` contains the start of the 5 most similar 24 h periods for each time step, which are used to determine the customers' price forecasts; these files are created in a preprocessing step to reduce computation time during training
- `P_infl_load.csv` contains data on inflexible consumption
- `P_pv_1kW.csv` contains the PV generation data
- `weather_data.csv` contains the weather data (not normalized)
- `weather_data_norm.csv` contains the normalized weather data

**Branch `individual`**
- `cust_config.xlsx` contains the customer configurations; see sheet "Description" for further information
- `customer_data/<customer_id>.csv` contains the time series data for one customer

**Branch `vectorized`**
- `ev_config.xlsx` contains the electric vehicle (EV) configurations; see sheet "Description" for further information
- `ev_data.csv` contains the EV time series data
- `wh_hp_config.xlsx` contains the electric water heater and heat pump configurations; see sheet "Description" for further information 
- `wh_hp_data.csv` contains the electric water heater and heat pump time series data
---
