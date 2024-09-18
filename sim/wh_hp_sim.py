import ast
import datetime
import numpy as np
import pandas as pd
import pytz


c_w = 4.184 * 1e3 / 3.6e+6  # [kWh/(kg*K)], specific heat capacity of water
dens_w = 1  # [kg/l], density of water


class WHHPs:
    """
    Electric water heater (WH) and heat pump (HP) simulation class.

    The following references are used:
    [1] D. Fischer, T. Wolf, J. Scherer, and B. Wille-Haussmann, "A stochastic bottom-up model for space heating and domestic hot water
        load profiles for German households," Energy and Buildings, vol. 124, 2016, https://doi.org/10.1016/j.enbuild.2016.04.069.
    [2] D. Fischer, T. Wolf, J. Wapler, R. Hollinger, H. Madani, "Model-based flexibility assessment of a residential heat pump pool,"
        Energy, vol. 118, 2017, https://doi.org/10.1016/j.energy.2016.10.111.
    [3] B. Hilpisch, "Synthetic Load Modelling for Load Flow Simulations," Master Thesis, ETH Zurich, 2021.
    """

    def __init__(self, config, arr_T_amb, daterange, T_env=15, T_sh_hys=5):
        # From the general config
        self.DT = config.DT
        self.K = config.K
        self.tz = pytz.timezone(config.tz)
        self.tz_local = pytz.timezone(config.tz_local)
        self.flag_save = config.env_type == "evaluation" or config.flag_cust_res
        if self.flag_save:
            self.results = []

        # WH and HP config
        self.df_config = pd.read_excel(config.path_wh_hp_config, index_col=0)

        # Number and masks for the different configurations
        self.n_customers = len(self.df_config)
        self.has_wh = self.df_config["flag_WH"].values
        self.whs = self.df_config.index[self.has_wh]
        self.n_whs = sum(self.has_wh)
        self.has_hp = self.df_config["flag_HP"].values
        self.hps = self.df_config.index[self.has_hp]
        self.n_hps = sum(self.has_hp)
        self.has_both = self.has_wh & self.has_hp
        self.only_wh = self.has_wh & ~self.has_hp
        self.only_hp = ~self.has_wh & self.has_hp

        # Append values that are needed in every time step to save computation time
        self.price_fc_group = self.df_config["price_fc_group"].values.astype(int)
        # WH
        self.P_rated_WH = self.df_config["P_rated_WH"][self.has_wh].values
        self.T_LO_WH = self.df_config["T_LO_WH"][self.has_wh].values
        self.T_UP_WH = self.df_config["T_UP_WH"][self.has_wh].values
        self.Lambda_tank_WH = self.df_config["Lambda_tank_WH"][self.has_wh].values
        self.V_tank_WH = self.df_config["V_tank_WH"][self.has_wh].values
        self.K_block_day_WH = self.df_config["K_block_day_WH"][self.has_wh].values.astype(int)
        # HP
        self.P_rated_HP = self.df_config["P_rated_HP"][self.has_hp].values
        self.Lambda_tank_HP = self.df_config["Lambda_tank_HP"][self.has_hp].values
        self.V_tank_HP = self.df_config["V_tank_HP"][self.has_hp].values
        self.K_block_day_HP = self.df_config["K_block_day_HP"][self.has_hp].values.astype(int)
        self.K_block_instance_HP = self.df_config["K_block_instance_HP"][self.has_hp].values.astype(int)
        self.K_min_block_HP = self.df_config["K_min_block_HP"][self.has_hp].values.astype(int)
        self.K_min_unblock_HP = self.df_config["K_min_unblock_HP"][self.has_hp].values.astype(int)

        # Assign the function for determining the blocking signals based on the chosen setting
        if config.dev_ctrl == "none":
            self.get_control = self._ctrl_no_blocking
        elif config.dev_ctrl == "ripple":
            self.get_control = self._ctrl_ripple
            self.df_config["cmds_nov-mar_WH"] = self.df_config["cmds_nov-mar_WH"].apply(lambda x: ast.literal_eval(x) if not pd.isna(x) else np.nan)
            self.df_config["cmds_apr-oct_WH"] = self.df_config["cmds_apr-oct_WH"].apply(lambda x: ast.literal_eval(x) if not pd.isna(x) else np.nan)
        elif config.dev_ctrl == "dynamic":
            self.get_control = self._ctrl_dynamic

        # Time series data
        self.df_ts = pd.read_csv(config.path_wh_hp_data, sep=";", index_col=0, header=[0, 1])
        self.df_ts.index = pd.DatetimeIndex(self.df_ts.index)
        self.df_ts.columns = self.df_ts.columns.set_levels(self.df_ts.columns.levels[0].astype(int), level=0)  # customer id to int
        # Filter for devices that are part of the config file and sort the columns
        self.df_ts = self.df_ts.loc[:, (self.df_config.index, slice(None))]
        # Round all values to 4 decimal places
        self.df_ts = self.df_ts.round(4)
        # Only keep simulation daterange
        self.df_ts = self.df_ts.loc[daterange]
        # Change dtype
        self.df_ts = self.df_ts.astype(np.float32)

        # Add columns for the simulation results
        if self.flag_save:
            cols_wh = ["WH_u", "WH_rho", "WH_T", "WH_p"]
            cols_hp = ["HP_u", "HP_rho", "HP_T", "HP_p"]
            df_wh = pd.DataFrame(index=self.df_ts.index, columns=pd.MultiIndex.from_product([self.whs, cols_wh]), dtype=np.float32)
            df_hp = pd.DataFrame(index=self.df_ts.index, columns=pd.MultiIndex.from_product([self.hps, cols_hp]), dtype=np.float32)
            self.df_ts = pd.concat([self.df_ts, df_wh, df_hp], axis=1)
            self.df_ts = self.df_ts.loc[:, (self.df_config.index, slice(None))]  # sort the columns once again

        # Initialize the device history with the previous values
        # Logic for initializing the action history:
        # 1)  The last value should be 0, such that all devices are initially off --> avoids major peak in the first time step;
        #     Like this, the hysteresis controller will wait until the temperature drops below T_LO, which varies between customers
        # 2)  Both 0 and 1 should be feasible in the next time step
        #     --> EWH:    all 1 except for last action; the last action is set to 0
        #         HP:     all 1 except for last K_min_block_HP actions; these are set to 0
        self.T_prev_WH = self.df_config.loc[self.has_wh, "T_prev_WH"].values
        self.T_prev_HP = self.df_config.loc[self.has_hp, "T_prev_HP"].values
        self.rho_WH = np.repeat(0, self.n_whs)
        self.rho_HP = np.repeat(0, self.n_hps)
        self.prev_applied_WH = np.tile([1] * (self.K - 2) + [0], (self.n_whs, 1)).T
        min_K_min_block = min(self.K_min_block_HP)
        self.prev_applied_HP = np.tile([1] * (self.K - 1 - min_K_min_block) + [0] * min_K_min_block, (self.n_hps, 1)).T

        # Assumed temperature of EWH/HP environment
        self.T_env = T_env

        # Compute the temperature of the inflowing cold water; used for the WH simulation
        self.arr_T_w_c = self._get_cold_water_T(arr_T_amb)  # [°C] series with time index; do not consider year (only one profile for all years)

        # HP settings
        # Coefficients for the space heating temperature set-point, the COP and the capacity, based on Table 1 in [2];
        # Note: so far, we have only included an "ASHP" (Air Source Heat Pump) with "floor_W31" (A-10/W31)
        self.coeff_T_sh_set = [27.98, -0.33, -0.0039]
        self.coeff_cop = [5.06, -0.05, 0.00006]
        self.coeff_capacity = [5.8, 0.21]

        # Hysteresis band, based on section 2.1.6 in [2]
        self.T_sh_hys = T_sh_hys  # [°C]
        arr_T_sh_set = self._compute_T_set(arr_T_amb)
        self.arr_T_LO = arr_T_sh_set  # [°C] series with time index
        self.arr_T_UP = arr_T_sh_set + self.T_sh_hys  # [°C] series with time index
        # Coefficient of performance
        self.arr_COP = self._compute_cop(arr_T_amb, self.arr_T_UP)  # series with time index

        # Smart meter measurements show that the power in the operanden table refers to the electrical power;
        # Further, the electrical power when the heat pump is on does not vary a lot for different temperatures
        # --> use fixed electrical power and compute thermal power by multiplying with the COP; for earlier version, see simulation classes > customer.py

        # Heating system activated or not; per customer --> add to df_ts
        for c_id in self.df_config.index[self.has_hp]:
            arr_heat_sys_act = self._get_heat_sys_act(arr_T_amb, T_build_heat_lim=self.df_config.loc[c_id, "T_heating_lim"], K=self.K)
            self.df_ts.loc[:, (c_id, "heat_sys_act")] = arr_heat_sys_act

        # Which HPs are considered to be modulating
        self.modulating = (self.P_rated_HP >= config.modulating_HP_th)
        self.modulating_min_factor = config.modulating_HP_min_factor

        # Sort the columns once again
        self.df_ts = self.df_ts.loc[:, (self.df_config.index, slice(None))]

        return

    def _get_cold_water_T(self, T_amb):
        # Temperature of incoming cold water (based on [1, 3]), used as a lower limit for the storage temperature;
        # We use the average temperature over all years as the underlying temperature profile instead of computing the profile for each year
        # to avoid a sudden shift from one year to the other;
        # further, the data may only cover part of the final year, which would lead to a misleading n_days_offset
        T_amb = T_amb.copy()  # copy in order to not change the original series
        T_amb.index = T_amb.index.map(lambda t: t.replace(year=2000))
        avg_T_amb = T_amb.groupby(T_amb.index).mean()
        # Use rolling mean (with left-sided window) to account for temperature change delay due to laying depth of water pipes
        n_days_offset = avg_T_amb.rolling(datetime.timedelta(weeks=2)).mean().idxmin().dayofyear
        # Eq. (2) in [1]
        T_w_c = pd.Series(avg_T_amb.mean() - 3 * np.cos((2 * np.pi) / 365 * (avg_T_amb.index.dayofyear - n_days_offset)), index=avg_T_amb.index)  # [°C]

        return T_w_c

    def _compute_T_set(self, T_amb):
        # Eq. (3) in [2], temperature set-point, serves as lower bound for the hysteresis band as written in section 2.1.6, important: only valid for ambient temperatures below 15°C
        return self.coeff_T_sh_set[0] + self.coeff_T_sh_set[1] * np.minimum(T_amb, 15) + self.coeff_T_sh_set[2] * np.minimum(T_amb, 15) ** 2

    def _compute_cop(self, T_amb, T_UP):
        # Eq. (5) in [2], temperature difference T_sink (approximated by upper bound of hysteresis band) - T_source
        dT = T_UP - T_amb
        # Eq. (4) in [2], coefficient of performance (COP)
        return self.coeff_cop[0] + self.coeff_cop[1] * dT + self.coeff_cop[2] * dT ** 2

    def _get_heat_sys_act(self, T_amb, T_build_heat_lim=15, n_days_heat_lim=3, n_days_min=3, K=96):
        # Determine whether the heating system is activated or not
        # Approach: If the average of the ambient temperature in the next n_days_heat_lim days is above T_build_heat_lim, the heating system is assumed to be deactivated;
        # The state (heating system activated or not) has to be the same for at least n_days_min days, otherwise the state is overwritten with the previous one
        avg_T_amb = T_amb.rolling(f"{n_days_heat_lim}d").mean().shift(-int(n_days_heat_lim * K)).ffill()  # values for last n_days_heat_lim are filled with previous value
        heat_sys_act = (avg_T_amb < T_build_heat_lim).astype(np.float32)
        n_consec = heat_sys_act.groupby((heat_sys_act != heat_sys_act.shift()).cumsum()).transform("size")  # counts how often the value is repeated since the last change
        heat_sys_act[n_consec < n_days_min * K] = np.nan
        heat_sys_act.ffill(inplace=True)
        heat_sys_act.bfill(inplace=True)  # backward fill in case the period at the beginning is too short
        return heat_sys_act

    def _count_last_change(self, l):
        """
        :param list l: list to be analyzed
        :return: repetitions of the last value l[-1] at the end of the list; e.g., returns 3 for l = [0, 0, 1, 1, 1]
        :rtype: int
        """
        count = 0
        for i in reversed(l):
            if i == l[-1]:
                count += 1
            else:
                break
        return count

    def _verify_wh(self, prev_applied, action):
        """
        Vectorized version of the function verify_action for the WH.

        :param np.array prev_applied: array containing previous 95 actions (1 - unblock, 0 - block) for all WHs
        :param np.array action: proposed action for the next time step (1 - unblock, 0 - block) for all WHs
        :return: corrected action (1 - unblock, 0 - block) for all WHs
        :rtype: np.array
        """
        sum_prev_applied = np.sum(prev_applied, axis=0)
        mask_invalid = sum_prev_applied + action < self.K - self.K_block_day_WH
        action[mask_invalid] = 1

        return action

    def _verify_hp(self, prev_applied, action):
        """
        Vectorized version of the function verify_action for the HP.

        :param np.array prev_applied: array containing previous 95 actions (1 - unblock, 0 - block) for all HPs
        :param np.array action: proposed action for the next time step (1 - unblock, 0 - block) for all HPs
        :return: corrected action (1 - unblock, 0 - block) for all HPs
        :rtype: np.array
        """
        # Masks for invalid actions under different conditions
        n_last_change = np.array([self._count_last_change(prev_applied[:, i]) for i in range(self.n_hps)])
        mask_K_block_day = np.sum(prev_applied, axis=0) + action < self.K - self.K_block_day_HP
        mask_K_block_instance = (prev_applied[-1, :] == 0) & (action == 0) & (n_last_change >= self.K_block_instance_HP)
        mask_K_min_block = (prev_applied[-1, :] == 0) & (action == 1) & (n_last_change < self.K_min_block_HP)
        mask_K_min_block_block_day = (prev_applied[-1, :] == 1) & (action == 0) \
                                     & [sum(prev_applied[-(self.K - self.K_min_block_HP[i]):, i]) < self.K - self.K_block_day_HP[i] for i in range(self.n_hps)]
        mask_K_min_unblock = (prev_applied[-1, :] == 1) & (action == 0) & (n_last_change < self.df_config["K_min_unblock_HP"][self.has_hp].values)

        # Combine the masks
        mask_1 = mask_K_block_day | mask_K_block_instance | mask_K_min_block_block_day | mask_K_min_unblock  # set to 1
        mask_0 = mask_K_min_block  # set to 0
        assert np.max(np.sum(np.array([mask_1, mask_0]).astype(int), axis=0)) <= 1, "Contradictory masks"

        # Correct actions based on the masks
        action[mask_1] = 1
        action[mask_0] = 0

        return action

    def _ctrl_no_blocking(self, ts, **kwargs):
        """
        Benchmark scenario: None of the devices is blocked
        """

        self.WH_u = np.full(self.n_whs, 1)
        self.HP_u = np.full(self.n_hps, 1)

        return

    def _ctrl_ripple(self, ts, **kwargs):
        """
        Rule-based controller: implements the current ripple control signals which only depend on the current time.
        """
        # Ripple control signal depends on the local time
        ts_local = ts.astimezone(tz=self.tz_local)

        # WH
        WH_cmds = self.df_config["cmds_nov-mar_WH"].values if ((ts_local.month <= 3) or (ts_local.month >= 11))\
            else self.df_config["cmds_apr-oct_WH"].values
        u = np.full(self.n_whs, 0)  # blocked is the default
        for i, cmd in enumerate(WH_cmds[self.has_wh]):  # loop through customers with WH
            for times in cmd:
                start, end = [datetime.datetime.strptime(time, "%H:%M").time() for time in times]
                if (ts_local.time() >= start) & (ts_local.time() < end):  # forward-looking time convention (timestamp 00:00 --> time interval 00:00 - 00:15)
                    u[i] = 1
        self.WH_u = u

        # HP
        u = np.full(self.n_hps, 1)  # unblocked is the default
        # If flag_HP_blocked is True, the HP is blocked on weekdays between 11 am and noon
        dev_blocked = ((self.df_config["flag_HP_blocked"].values)
                       & (ts_local.time() >= datetime.time(11, 0))
                       & (ts_local.time() < datetime.time(12, 0))
                       & (ts_local.weekday() < 5))[self.has_hp]
        u[dev_blocked] = 0
        self.HP_u = u

        return

    def _ctrl_dynamic(self, price_next, dict_price_24h_fc, **kwargs):
        """
        Rule-based controller: The price for the next time step is compared to the price forecast for the remaining 23.75 hours.
        The WH is unblocked if the price is among the (K - K_block_day) lowest prices.
        The HP is unblocked if the price is among the (K - K_block_day) lowest prices.

        :param datetime.datetime ts: timestamp
        :param float price_next: price for the next time step (between -1 and 1)
        :param dict dict_price_24h_fc: dictionary with price forecasts for the remaining 23.75 h (between -1 and 1)

        Note: do not sort price forecast as part of env_gym; for the EVs, the input should not be sorted
        """
        # sort the price forecast for the remaining 23.75 h
        for k, v in dict_price_24h_fc.items():
            dict_price_24h_fc[k] = sorted(v)

        # WH
        fc_group = self.price_fc_group[self.has_wh]
        limit = np.round([dict_price_24h_fc[fc_group[i]][self.K - self.K_block_day_WH[i] - 1] for i in range(self.n_whs)], 4)  # -1 bc index starts at 0
        WH_u = (price_next <= limit)  # unblock as early as possible if values are the same

        # HP
        fc_group = self.price_fc_group[self.has_hp]
        limit = np.round([dict_price_24h_fc[fc_group[i]][self.K - self.K_block_day_HP[i] - 1] for i in range(self.n_hps)], 4)  # -1 bc index starts at 0
        HP_u = (price_next <= limit)  # unblock as early as possible if values are the same

        # Verify the actions
        self.WH_u = self._verify_wh(prev_applied=self.prev_applied_WH, action=WH_u)
        self.HP_u = self._verify_hp(prev_applied=self.prev_applied_HP, action=HP_u)

        # Update the action history
        self.prev_applied_WH = np.vstack((self.prev_applied_WH[1:, :], self.WH_u))
        self.prev_applied_HP = np.vstack((self.prev_applied_HP[1:, :], self.HP_u))

        return

    def _sim_WHs(self, ts):
        """
        Simulate device behavior for one time step; vectorized version
        Rho represents whether the device is running or not.
        When the device is operating and is blocked before reaching the upper temperature bound,
        it is assumed that it will stay off until the storage reaches the lower temperature bound
        (in contrast to continuing operating once it is unblocked again).
        """

        # Hysteresis control
        self.rho_WH[self.T_prev_WH < self.T_LO_WH] = 1  # turn on if below lower bound
        self.rho_WH[self.T_prev_WH > self.T_UP_WH] = 0  # turn off if above upper bound

        # Blocking signal; the device only operates if the internal state is ON, and it is not blocked
        dev_blocked = self.WH_u == 0
        self.rho_WH[dev_blocked] = 0

        # Supplied thermal energy
        power = self.P_rated_WH * self.rho_WH  # [kW]
        Q_in = power * self.DT  # [kWh], we assume a COP of 1

        # Extracted heat through water tapping
        Q_out = self.df_ts.loc[ts, (self.df_config.index[self.has_wh], "Q_DHW_[kWh]")].values  # [kWh]

        # Storage heat losses
        Q_loss = self.Lambda_tank_WH * (self.T_prev_WH - self.T_env) * self.DT  # [kWh] Eq. (3.19) in [3]

        # Overall change in contained energy and temperature
        d_Q = Q_in - Q_out - Q_loss  # [kWh] Eq. (3.18) in [3]
        d_T = d_Q / (self.V_tank_WH * dens_w * c_w)  # [°C] Eq. (3.17) in [3]

        # Compute water temperature in the tank
        T = self.T_prev_WH + d_T  # [°C] Eq. (3.16) in [3]
        # The water temperature cannot drop below the temperature of the incoming cold water T_w_c
        T_w_c = self.arr_T_w_c[ts.replace(year=2000)]  # T_w_c same for all years, with year 2000
        T[T < T_w_c] = T_w_c

        # Update
        self.T_prev_WH = T

        return power

    def _sim_HPs(self, ts):
        """
        Simulate device behavior for one time step; vectorized version
        Rho represents whether the device is running or not.
        When the device is operating and is blocked before reaching the upper temperature bound,
        it is assumed that it will stay off until the storage reaches the lower temperature bound
        (in contrast to continuing operating once it is unblocked again).
        """
        # Heating system active
        heat_sys_act = self.df_ts.loc[ts, (self.df_config.index[self.has_hp], "heat_sys_act")].values == 1

        # Extracted heat for space heating (SH)
        Q_out = self.df_ts.loc[ts, (self.df_config.index[self.has_hp], "Q_SH_[kWh]")].values  # [kWh]
        Q_out[~heat_sys_act] = 0  # no heat extraction if the heating system is not activated

        # Storage heat losses
        Q_loss = self.Lambda_tank_HP * (self.T_prev_HP - self.T_env) * self.DT  # [kWh] Eq. (3.19) in [3]

        # Hysteresis control
        self.rho_HP[self.T_prev_HP < self.arr_T_LO[ts]] = 1  # turn on if below lower bound
        self.rho_HP[self.T_prev_HP > self.arr_T_UP[ts]] = 0  # turn off if above upper bound

        # Blocking signal and heating system active
        # The device only operates if the internal state is ON, it is not blocked, and the heating system is activated
        dev_blocked = self.HP_u == 0
        self.rho_HP[dev_blocked | ~heat_sys_act] = 0

        # Supplied thermal energy
        # On/off control (values for modulating HPs are overwritten below)
        power = self.P_rated_HP * self.rho_HP  # [kW]
        COP = self.arr_COP[ts]
        Q_in = power * COP * self.DT  # [kWh]
        # Modulating HPs
        # The thermal energy is equal to Q_out + Q_loss, unless the value is smaller than Q_in_min or larger than Q_in_max;
        # In these cases, the thermal energy is set to the minimum or maximum value, respectively
        Q_in_mod_max = self.P_rated_HP[self.modulating] * COP * self.DT  # [kWh]
        Q_in_mod_min = self.modulating_min_factor * Q_in_mod_max  # [kWh]
        Q_in_mod = np.maximum(Q_out[self.modulating] + Q_loss[self.modulating], Q_in_mod_min)  # [kWh]; bound to minimum power
        Q_in_mod = np.minimum(Q_in_mod, Q_in_mod_max)  # [kWh]; bound to maximum power
        mask = self.T_prev_HP[self.modulating] < self.arr_T_LO[ts]
        Q_in_mod[mask] = Q_in_mod_max[mask]  # [kWh]; run at maximum power if temperature below lower bound
        Q_in_mod = Q_in_mod * self.rho_HP[self.modulating]  # [kWh]; only if Rho is 1
        power[self.modulating] = Q_in_mod / (self.arr_COP[ts] * self.DT)  # [kW]
        Q_in[self.modulating] = Q_in_mod  # [kWh]

        # Overall change in contained energy and temperature
        d_Q = Q_in - Q_out - Q_loss  # [kWh] Eq. (3.18) in [3]
        d_T = d_Q / (self.V_tank_HP * dens_w * c_w)  # [°C] Eq. (3.17) in [3]

        # Compute water temperature in the tank
        T = self.T_prev_HP + d_T  # [°C] Eq. (3.16) in [3]
        # Note: we do not model the thermal inertia of the building --> the temperature in the tank may drop to low values,
        # while the temperature in the building is expected to be sufficiently high

        # Update
        self.T_prev_HP = T

        return power

    def sim(self, ts, price_next=None, dict_price_24h_fc=None, price_hist=None):
        # Determine the blocking signals
        self.get_control(ts=ts, price_next=price_next, dict_price_24h_fc=dict_price_24h_fc, price_hist=price_hist)

        # Determine power
        p_WH = self._sim_WHs(ts)
        p_HP = self._sim_HPs(ts)

        if self.flag_save:
            self.results.append((self.WH_u, self.rho_WH, self.T_prev_WH, p_WH,
                                 self.HP_u, self.rho_HP, self.T_prev_HP, p_HP))

        return sum(p_WH), sum(p_HP)

    def save_results(self, path):
        # Process the results
        WH_u, WH_rho, WH_T, WH_p, HP_u, HP_rho, HP_T, HP_p = zip(*self.results)
        n_ts = len(WH_u)
        df = self.df_ts.copy()
        idx = df.index[n_ts - 1]
        df.loc[:idx, (slice(None), "WH_u")] = WH_u
        df.loc[:idx, (slice(None), "WH_rho")] = WH_rho
        df.loc[:idx, (slice(None), "WH_T")] = WH_T
        df.loc[:idx, (slice(None), "WH_p")] = WH_p
        df.loc[:idx, (slice(None), "HP_u")] = HP_u
        df.loc[:idx, (slice(None), "HP_rho")] = HP_rho
        df.loc[:idx, (slice(None), "HP_T")] = HP_T
        df.loc[:idx, (slice(None), "HP_p")] = HP_p

        # Save
        df.to_csv(path, sep=";", index=True, header=True, index_label="time")

        # Reset the results
        self.results = []

        return
