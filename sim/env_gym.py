import copy
from dateutil.relativedelta import relativedelta as rd
import datetime
import gymnasium as gym
import holidays
import logging
import numpy as np
import pandas as pd
import pytz

from .ev_sim import EVs
from .wh_hp_sim import WHHPs
from .policies import ProportionalPriceInflPerDay


def read_time_series_from_file(path, tz):
    """
    :param pathlib.Path path: file to be read
    :param str tz: desired timezone for the output (string or pytz object)
    :return: dataframe with datetime index and the specified timezone
    :rtype: pd.DataFrame
    """
    df = pd.read_csv(path, sep=";", index_col="time")
    df.index = pd.DatetimeIndex(df.index).tz_convert(tz)
    dt = df.index[1] - df.index[0]
    try:
        df.index.freq = dt
    except:
        logging.warning("The sample frequency of the data is not consistent throughout the whole dataframe.")
    return df


class SimEnv(gym.Env):
    def __init__(self, config, env_type=None):
        """
        :param config: configuration object
        :param str env_type: type of the environment; "training" or "evaluation"
        """
        assert env_type in ["training", "evaluation"],\
            "Please specify the type of the environment ('training' or 'evaluation')"
        logging.info(f"Initializing environment of type {env_type}.")
        self.config = copy.deepcopy(config)  # don't change the original config object
        self.config.env_type = env_type

        if self.config.env_type == "evaluation":
            self.config.start_date = self.config.start_date_eval
            self.config.path_msd = self.config.path_msd_eval

        # Initializations related to time
        # Holidays
        self.ch_holidays = holidays.CH(subdiv="ZH")
        # Time initializations
        self.tz = pytz.timezone(self.config.tz)
        self.tz_local = pytz.timezone(self.config.tz_local)
        self.ts_init = self.tz.localize(datetime.datetime.strptime(self.config.start_date, "%Y-%m-%d")).replace(hour=0, minute=0)
        self.ts = self.ts_init
        self._get_time()
        self._get_time_hist()

        # Define the date range which will be simulated
        end_date = self.ts_init + rd(days=self.config.days_eps) \
            if self.config.env_type == "training" \
            else self.ts_init + rd(days=self.config.days_eval)  # evaluation
        self.daterange = pd.date_range(start=self.ts_init, end=end_date, freq=f"{self.config.DT}H", tz=self.tz, inclusive="left")  # forward-looking; right end not included

        # Read weather data
        self.weather_data = read_time_series_from_file(self.config.path_weather, self.config.tz)
        self.weather_data_n = read_time_series_from_file(self.config.path_weather_n, self.config.tz)
        assert self.daterange[0] - rd(hours=self.config.a["h"] * self.config.DT) in self.weather_data.index,\
            "The weather data does not start h time steps before the simulation period;" \
            "this is required because the historic weather data is part of the observation"
        assert self.daterange[-1] + rd(hours=24 - self.config.DT) in self.weather_data.index,\
            "The weather data does not end one day after the simulation period;" \
            "this is required because the future weather data is part of the observation"

        # Initialize the customer devices
        self.whhps = WHHPs(config=self.config, arr_T_amb=self.weather_data["t_2m_C"].round(4), daterange=self.daterange)
        self.evs = EVs(config=self.config, daterange=self.daterange)

        # Read the inflexible net load (inflexible consumption - generation)
        p_infl_load = read_time_series_from_file(self.config.path_infl_load, self.config.tz).squeeze()  # df only has one column
        p_pv = read_time_series_from_file(self.config.path_pv, self.config.tz).squeeze()  # df only has one column
        self.p_infl_net = p_infl_load * self.config.fac_infl_load - p_pv * self.config.fac_pv

        # Initialize the load history
        self.p_tot_hist_n = [0.0] * self.config.a["h"]  # total load in the last h time steps
        p_infl_n = self.p_infl_net.loc[self.daterange].copy() / self.config.fac_norm_p
        assert max(abs(p_infl_n)) <= 1,  "The inflexible load exceeds fac_norm_p; please choose a higher value."
        self.p_tot_full_hist_n = p_infl_n.copy()  # full history of the total load
        self.p_tot = 0
        self.p_infl_full_hist_n = p_infl_n.copy()  # full history of the inflexible load
        self.p_infl = 0
        self.th_hist_n = [self.config.th / self.config.fac_norm_p] * self.config.a["h"]  # threshold for the total load

        # Initialize the prices
        if self.config.dev_ctrl == "dynamic":  # dataframe with the most similar day for each time step and forecast group
            self.df_msd = read_time_series_from_file(self.config.path_msd, self.config.tz)
            for col in self.df_msd.columns:
                self.df_msd[col] = pd.to_datetime(self.df_msd[col])  # convert to datetime
        self._initialize_prices()

        # Define observation space; all values are normalized between -1 and 1 --> "_n"
        # Using sin/cos encoding to capture cyclic pattern of time indicators; e.g., midnight (00:00) corresponds to x = 0, y = 1 for the quarter-hour of the day; it then moves clock-wise
        self.observation_space = gym.spaces.Dict(
            {
                "q_x_hist":     gym.spaces.Box(low=-1, high=1, shape=(self.config.a["h"],), dtype=np.float32),  # quarter-hour of the day, x-value
                "q_y_hist":     gym.spaces.Box(low=-1, high=1, shape=(self.config.a["h"],), dtype=np.float32),  # quarter-hour of the day, y-value
                "dw_x_hist":    gym.spaces.Box(low=-1, high=1, shape=(self.config.a["h"],), dtype=np.float32),  # day of the week, x-value
                "dw_y_hist":    gym.spaces.Box(low=-1, high=1, shape=(self.config.a["h"],), dtype=np.float32),  # day of the week, y-value
                "holiday_hist": gym.spaces.Box(low=-1, high=1, shape=(self.config.a["h"],), dtype=np.float32),  # holiday (1) or not (-1)
                "price_hist_n": gym.spaces.Box(low=-1, high=1, shape=(self.config.a["h"],), dtype=np.float32),  # price history
                "p_tot_hist_n": gym.spaces.Box(low=-1, high=1, shape=(self.config.a["h"],), dtype=np.float32),  # load history (total)
                "I_sol_hist_n": gym.spaces.Box(low=-1, high=1, shape=(self.config.a["h"],), dtype=np.float32),  # solar irradiation history
                "T_amb_hist_n": gym.spaces.Box(low=-1, high=1, shape=(self.config.a["h"],), dtype=np.float32),  # ambient temperature history
                "th_hist_n":    gym.spaces.Box(low=0, high=1, shape=(self.config.a["h"],), dtype=np.float32),  # threshold for total load
                "I_sol_fc_n":   gym.spaces.Box(low=-1, high=1, shape=(self.config.a["f"],), dtype=np.float32),  # solar irradiation forecast
                "T_amb_fc_n":   gym.spaces.Box(low=-1, high=1, shape=(self.config.a["f"],), dtype=np.float32),  # temperature forecast
            }
        )

        # Define action space
        if self.config.a["action_type"] == "abs":
            self.action_space = gym.spaces.Discrete(self.config.a["n_prices"])
        elif self.config.a["action_type"] == "rel":
            self.action_space = gym.spaces.Discrete(len(self.config.a["rel_actions"]))
        self.price_vals_n = np.linspace(-1, 1, self.config.a["n_prices"])  # possible price values in the normalized space
        self.price_n_rel = -1  # normalized price value that results from relative action before clipping

        # Initialize episode counter, 1 episode corresponds to "days_eps" days, defined in the config
        self.cnt_eps = -1
        self.truncated = False  # end of episode reached

        # Other initializations
        self.dict_cost_components = {}

        return

    def _initialize_prices(self):
        """
        Initialize the price history and the price for the first time step;
        this function is called in __init__
        """
        self.price_hist_n = [-1] * self.config.a["h"]  # last h values
        self.price_n = -1

        # Full history for the considered time window; used to determine price forecasts
        # Use the inflexible load for initialization
        p = ProportionalPriceInflPerDay(p_infl=self.p_infl_full_hist_n.copy(), config=self.config)
        # The returned actions are according to action_type "abs" with values {0, 1, ... , self.config.a["n_prices"] - 1}
        # --> convert to [-1, 1]
        self.price_full_hist_n = p.actions.apply(lambda x: self._discrete_action_to_n(x))

        return

    def _get_obs(self):
        dict_return = {
            "q_x_hist":         np.array(self.q_x_hist),
            "q_y_hist":         np.array(self.q_y_hist),
            "dw_x_hist":        np.array(self.dw_x_hist),
            "dw_y_hist":        np.array(self.dw_y_hist),
            "holiday_hist":     np.array(self.holiday_hist),
            "price_hist_n":     np.array(self.price_hist_n),
            "p_tot_hist_n":     np.array(self.p_tot_hist_n),
            "I_sol_hist_n":     self.I_sol_hist_n,
            "T_amb_hist_n":     self.T_amb_hist_n,
            "th_hist_n":        np.array(self.th_hist_n),
            "I_sol_fc_n":       self.I_sol_fc_n,
            "T_amb_fc_n":       self.T_amb_fc_n
            }
        return dict_return

    def _get_weather(self):
        # Get NORMALIZED weather data
        idx = self.weather_data_n.index.get_loc(self.ts)
        # Forecast
        df_weather_data_n = self.weather_data_n.iloc[idx:idx + self.config.a["f"]]
        self.I_sol_fc_n = df_weather_data_n["global_rad_W_per_m2_n"].values
        self.T_amb_fc_n = df_weather_data_n["t_2m_C_n"].values
        # Past values
        df_weather_data_n = self.weather_data_n.iloc[idx - self.config.a["h"]:idx]  # idx / self.ts not included
        self.I_sol_hist_n = df_weather_data_n["global_rad_W_per_m2_n"].values
        self.T_amb_hist_n = df_weather_data_n["t_2m_C_n"].values

        return

    def _get_time(self):
        ts_local = self.ts.astimezone(self.tz_local)
        self.is_holiday = (1 if self.ch_holidays.get(ts_local) is not None else -1)
        # Quarter-hour of the day
        q = (ts_local.time().hour * (1 / self.config.DT) + ts_local.time().minute / (self.config.DT * 60))  # starts counting at 0
        self.q_x = np.sin(np.radians((360 / self.config.K) * q))
        self.q_y = np.cos(np.radians((360 / self.config.K) * q))
        # Day of the week
        dw = (ts_local.weekday())
        self.dw_x = np.sin(np.radians((360 / 7) * dw))
        self.dw_y = np.cos(np.radians((360 / 7) * dw))

        return

    def _get_time_hist(self):
        local_timesteps = pd.date_range(start=(self.ts - rd(hours=self.config.a["h"] * self.config.DT)).astimezone(self.tz_local),
                                        end=self.ts.astimezone(self.tz_local),
                                        freq=f"{self.config.DT}H",
                                        tz=self.tz_local,
                                        inclusive="left")
        qs = [(ts.time().hour * (1 / self.config.DT) + ts.time().minute / (self.config.DT * 60)) for ts in local_timesteps]
        dws = [ts.weekday() for ts in local_timesteps]
        self.q_x_hist = [np.sin(np.radians((360 / self.config.K) * q)) for q in qs]
        self.q_y_hist = [np.cos(np.radians((360 / self.config.K) * q)) for q in qs]
        self.dw_x_hist = [np.sin(np.radians((360 / 7) * dw)) for dw in dws]
        self.dw_y_hist = [np.cos(np.radians((360 / 7) * dw)) for dw in dws]
        self.holiday_hist = [1 if self.ch_holidays.get(ts) is not None else -1 for ts in local_timesteps]

        return

    def _sim_ts(self):
        dict_price_24h_fc = {}
        if self.config.dev_ctrl == "dynamic":
            # Determine the price forecast (does not contain the current price, but only the remaining 23.75 hours)
            msd = list(self.df_msd.loc[self.ts])
            for i in range(self.config.n_price_fc_group):
                ts_msd = msd[i]
                dict_price_24h_fc[i] = list(self.price_full_hist_n[ts_msd + rd(hours=self.config.DT):ts_msd + rd(hours=24 - self.config.DT)].round(4))
                # Note: instead of rounding the prices above and when passing price_next below, we could also round
                # self.price_n at the beginning of the step function; however, this was not implemented like this in the SEST 2024 paper

        # Determine the load components
        self.p_infl = float(self.p_infl_net[self.ts])  # total inflexible load; no individual inflexible load anymore
        self.p_wh, self.p_hp = self.whhps.sim(ts=self.ts, price_next=round(self.price_n, 4), dict_price_24h_fc=dict_price_24h_fc.copy())  # copy dict to avoid changing the original
        self.p_ev = self.evs.sim(ts=self.ts, price_next=round(self.price_n, 4), dict_price_24h_fc=dict_price_24h_fc.copy())
        self.p_flx = self.p_wh + self.p_hp + self.p_ev  # total flexible load

        # Determine the total load and normalize it
        self.p_tot = self.p_infl + self.p_flx
        self.p_tot_n = self.p_tot / self.config.fac_norm_p
        if abs(self.p_tot_n) > 1.1:
            logging.warning("p_tot > 1.1 * fac_norm_p. Consider to increase fac_norm_p in config.yaml")
        self.p_tot_n = np.clip(a=self.p_tot_n, a_min=-1, a_max=1)  # between -1 and 1

        return

    def _get_reward(self):
        # Penalty for exceeding a fixed threshold
        cost_th = max(0, (abs(self.p_tot) - self.config.th) / self.config.th) ** 2

        # Penalty for the total power
        cost_p_tot = self.p_tot_n ** 2

        # Penalty for changing the price value
        cost_smoothing = abs(self.price_n - self.price_hist_n[-1])

        # Penalty for leaving the price range
        cost_price_range = 1 if self.config.a["action_type"] == "rel" and abs(self.price_n_rel) > 1 else 0

        # Weighted cost components
        self.dict_cost_components = {
            "cost_1": self.config.w["w1"] * cost_th,
            "cost_2": self.config.w["w2"] * cost_p_tot,
            "cost_3": self.config.w["w3"] * cost_smoothing,
            "cost_4": self.config.w["w4"] * cost_price_range
        }

        return - sum(self.dict_cost_components.values())

    def _discrete_action_to_n(self, action):
        # Mapping of discrete action ({0, 1, ... , self.config.a["n_prices"] - 1}) to values between -1 and 1
        return float(action) / (self.config.a["n_prices"] - 1) * 2 - 1

    def reset(self, seed=None, options=None):
        self.truncated = False
        logging.info(f"Resetting the {self.config.env_type} environment")

        # episode counter; initialized with -1 --> first episode is eps 0
        self.cnt_eps += 1
        logging.info(f"Number of {self.config.env_type} episodes collected: {self.cnt_eps}")

        # TRAINING environment
        # Reset at the very beginning, and whenever the end of the episode is reached
        if self.config.env_type == "training":
            # Time step
            self.ts = self.ts_init  # go back to the initial time step
            self._get_time()
            self._get_time_hist()

            # Keep other values for customer devices, prices, and load history as is --> different starting point when training over same data multiple times;
            # At the beginning, the values are initialized in __init__

        # EVALUATION environment
        # Reset only at the very beginning
        elif self.config.env_type == "evaluation":
            # No need to update anything here except for time step;
            # Customer devices, prices, and load history are initialized in __init__
            self.ts = self.ts_init

        # Initial observation
        logging.info(f"Timestamp at the start of the next episode: {self.ts}")
        self._get_weather()  # not performed as part of __init__; included here as we need it anyway whenever the time step is changed as part of the reset
        observation = self._get_obs()

        return observation, None

    def step(self, action=None):
        # Normalized price
        if self.config.policy == "rl":
            if self.config.a["action_type"] == "abs":
                self.price_n = self._discrete_action_to_n(action)
            elif self.config.a["action_type"] == "rel":
                self.price_n_rel = self.price_n + self.config.a["rel_actions"][int(action)] * 2 / (self.config.a["n_prices"] - 1)
                # Always adding to previous value might lead to rounding issues over time; additionally, the value should stay within [-1, 1] --> map to the closest value
                idx = (np.abs(self.price_vals_n - self.price_n_rel)).argmin()
                self.price_n = self.price_vals_n[idx]
        else:
            if isinstance(action, list):
                action = action[0]  # when testing other approaches, a list is passed as action such that the callbacks work the same way
            if self.config.policy == "prop_infl_per_day":  # formatted same way as action_type "abs"
                self.price_n = self._discrete_action_to_n(action)

        # Simulate the behavior of the environment
        self._sim_ts()

        # Compute the reward
        reward = self._get_reward()

        # Save information in a dictionary; this information can also be used in the callbacks
        dict_info = {
            "time": self.ts,  # saved in forward-looking notation (ts has not been updated at this point)
            "price_n": self.price_n,
            "p_tot": self.p_tot,
            "p_infl": self.p_infl,
            "p_flx": self.p_flx,
            "p_wh": self.p_wh,
            "p_hp": self.p_hp,
            "p_ev": self.p_ev,
            "cost_components": self.dict_cost_components
        }

        # Update the histories
        # Time
        self.q_x_hist = self.q_x_hist[1:] + [self.q_x]
        self.q_y_hist = self.q_y_hist[1:] + [self.q_y]
        self.dw_x_hist = self.dw_x_hist[1:] + [self.dw_x]
        self.dw_y_hist = self.dw_y_hist[1:] + [self.dw_y]
        self.holiday_hist = self.holiday_hist[1:] + [self.is_holiday]
        # Prices
        self.price_hist_n = self.price_hist_n[1:] + [self.price_n]
        self.price_full_hist_n[self.ts] = self.price_n
        # Load
        self.p_tot_hist_n = self.p_tot_hist_n[1:] + [self.p_tot_n]
        self.p_tot_full_hist_n[self.ts] = self.p_tot_n
        # self.p_infl_full_hist_n remains unchanged, only the total power changes --> no update

        # Update the time information
        self.ts += rd(hours=self.config.DT)
        self._get_time()

        # Update the weather
        self._get_weather()

        # End of episode reached?
        # Truncated
        self.truncated = True if self.ts == self.daterange[-1] + rd(hours=self.config.DT) else False  # end of episode / evaluation is reached
        # Terminated; continuous task --> no terminal state
        terminated = False

        # Observation
        observation = self._get_obs()

        # Save the results of the individual customers; gets overwritten after each episode --> only for sporadic checks
        if (self.config.env_type == "evaluation" or self.config.flag_cust_res) and self.truncated:
            self.whhps.save_results(self.config.output_path / f"df_whhps.csv")
            self.evs.save_results(self.config.output_path / f"df_evs.csv")
        return observation, reward, terminated, self.truncated, dict_info
