import datetime
import numpy as np
import pandas as pd

c_w = 4.184 * 1e3 / 3.6e+6  # [kWh/(kg*K)], specific heat capacity of water
dens_w = 1  # [kg/l], density of water


class Customer:
    """
    Customer simulation class.

    The following references are used:
    [1] D. Fischer, T. Wolf, J. Scherer, and B. Wille-Haussmann, "A stochastic bottom-up model for space heating and domestic hot water
        load profiles for German households," Energy and Buildings, vol. 124, 2016, https://doi.org/10.1016/j.enbuild.2016.04.069.
    [2] D. Fischer, T. Wolf, J. Wapler, R. Hollinger, H. Madani, "Model-based flexibility assessment of a residential heat pump pool,"
        Energy, vol. 118, 2017, https://doi.org/10.1016/j.energy.2016.10.111.
    [3] B. Hilpisch, "Synthetic Load Modelling for Load Flow Simulations," Master Thesis, ETH Zurich, 2021.
    """

    def __init__(self, c_id, dict_cust_config, arr_T_amb, path_data):
        self.c_id = c_id

        # Load the customer configuration
        self.price_fc_group = dict_cust_config["price_fc_group"]
        self.has_wh = dict_cust_config["flag_WH"]
        self.has_hp = dict_cust_config["flag_HP"]
        self.has_ev = dict_cust_config["flag_EV"]
        self.has_pv = dict_cust_config["flag_PV"]

        # Read file with data; only used for initialization, should not be attached to customer object
        df_data = pd.read_csv(path_data / f"{self.c_id}.csv", sep=";", index_col="time")
        df_data.index = pd.DatetimeIndex(df_data.index).tz_convert("UTC")

        # Add the inflexible power profile
        self.arr_p_infl = (df_data["P_infl_load_[kW]"] - df_data["P_PV_[kW]"]).round(4) \
            if self.has_pv else df_data["P_infl_load_[kW]"].round(4)  # series with time index

        # Add the device information
        if self.has_wh:
            self.wh = Waterheater(c_id=c_id, dict_cust_config=dict_cust_config, df_data=df_data, arr_T_amb=arr_T_amb)
        if self.has_hp:
            self.hp = Heatpump(c_id=c_id, dict_cust_config=dict_cust_config, df_data=df_data, arr_T_amb=arr_T_amb)
        if self.has_ev:
           self.ev = Ev(c_id=c_id, dict_cust_config=dict_cust_config, df_data=df_data)

    def __repr__(self):
        return f"{self.id}: {[dev for dev in ['wh', 'hp', 'ev'] if dev in self.__dict__]}"


class Waterheater:
    def __init__(self, c_id, dict_cust_config, df_data, arr_T_amb, T_env=15):
        self.c_id = c_id
        self.P_rated = dict_cust_config["P_rated_WH"]  # [kW]
        self.T_LO = dict_cust_config["T_LO_WH"]  # [°C]
        self.T_UP = dict_cust_config["T_UP_WH"]  # [°C]
        self.Lambda_tank = dict_cust_config["Lambda_tank_WH"]  # [kW/K]
        self.V_tank = dict_cust_config["V_tank_WH"]  # [l]
        self.K_block_day = int(dict_cust_config["K_block_day_WH"])

        # Initial temperature and internal state
        self.T_prev = dict_cust_config["T_prev_WH"]  # [°C]
        self.rho = 0  # Internal device state
        # 0 such that all devices are initially off --> avoids major peak in the first time step;
        # Like this, the hysteresis controller will wait until the temperature drops below T_LO, which varies between customers

        # Domestic hot water (DHW) demand
        self.arr_q_dhw = df_data["Q_DHW_[kWh]"].round(4)  # [kWh]; series with time index

        # Assumed temperature of EWH environment
        self.T_env = T_env

        # Compute the temperature of the inflowing cold water
        self.arr_T_w_c = self._get_cold_water_T(arr_T_amb)  # [°C] series with time index; do not consider year (only one profile for all years)

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

    def sim_ts(self, ts, u, DT=0.25):
        """
        Simulate device behavior for one time step

        Rho represents whether the device is running or not.
        When the device is operating and is blocked before reaching the upper temperature bound,
        it is assumed that it will stay off until the storage reaches the lower temperature bound
        (in contrast to continuing operating once it is unblocked again).

        :param ts: timestamp
        :param u: blocking signal (0 - blocked, 1 - unblocked)
        :param DT: duration of one time step in h
        :return: power
        :rtype: float
        """
        # Hysteresis control
        if self.T_prev < self.T_LO:
            self.rho = 1  # turn on if below lower bound
        elif self.T_prev > self.T_UP:
            self.rho = 0  # turn off if above upper bound

        # Blocking signal; the device only operates if the internal state is ON, and it is not blocked
        if u == 0:
            self.rho = 0

        # Supplied thermal energy
        power = self.P_rated * self.rho  # [kW]
        Q_in = power * DT  # [kWh], we assume a COP of 1

        # Extracted heat through water tapping
        Q_out = self.arr_q_dhw[ts]  # [kWh]

        # Storage heat losses
        Q_loss = self.Lambda_tank * (self.T_prev - self.T_env) * DT  # [kWh]

        # Overall change in contained energy and temperature
        d_Q = Q_in - Q_out - Q_loss  # [kWh] Eq. (3.18) in [3]
        d_T = d_Q / (self.V_tank * dens_w * c_w)  # [°C] Eq. (3.17) in [3]

        # Compute water temperature in the tank
        T = self.T_prev + d_T  # [°C] Eq. (3.16) in [3]
        # The water temperature cannot drop below the temperature of the incoming cold water T_w_c
        T = max(T, self.arr_T_w_c[ts.replace(year=2000)])  # T_w_c same for all years, with year 2000

        # Update
        self.T_prev = T

        return power


class Heatpump:
    def __init__(self, c_id, dict_cust_config, df_data, arr_T_amb, T_env=15, T_sh_hys=5):
        self.c_id = c_id
        # self.Q_rated = dict_cust_config["Q_rated_HP"]  # [kW]
        self.P_rated = dict_cust_config["P_rated_HP"]  # [kW]
        self.Lambda_tank = dict_cust_config["Lambda_tank_HP"]  # [kW/K]
        self.V_tank = dict_cust_config["V_tank_HP"]  # [l]
        self.K_block_day = int(dict_cust_config["K_block_day_HP"])
        self.K_block_instance = int(dict_cust_config["K_block_instance_HP"])
        self.K_min_block = int(dict_cust_config["K_min_block_HP"])
        self.K_min_unblock = int(dict_cust_config["K_min_unblock_HP"])

        # Initial temperature and internal state
        self.T_prev = dict_cust_config["T_prev_HP"]  # [°C]
        self.rho = 0  # Internal device state
        # 0 such that all devices are initially off --> avoids major peak in the first time step;
        # Like this, the hysteresis controller will wait until the temperature drops below T_LO, which varies between customers

        # Space heating (SH) demand
        self.arr_q_shd = df_data["Q_SH_[kWh]"].round(4)  # [kWh]; series with time index

        # Assumed temperature of HP environment
        self.T_env = T_env

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
        # --> use fixed electrical power and compute thermal power by multiplying with the COP
        # Maximal thermal capacity in kW
        # self.arr_Q = self.compute_capacity(arr_T_amb, dict_cust_config["Q_rated_HP"])  # [kW] series with time index
        # HP power demand if ON
        # self.arr_P = np.divide(self.arr_Q, self.arr_COP)  # [kW] series with time index

        # Heating system active
        self.arr_heat_sys_act = self._get_heat_sys_act(arr_T_amb, T_build_heat_lim=dict_cust_config["T_heating_lim"])

    def _compute_T_set(self, T_amb):
        # Eq. (3) in [2], temperature set-point, serves as lower bound for the hysteresis band as written in section 2.1.6, important: only valid for ambient temperatures below 15°C
        return self.coeff_T_sh_set[0] + self.coeff_T_sh_set[1] * np.minimum(T_amb, 15) + self.coeff_T_sh_set[2] * np.minimum(T_amb, 15) ** 2

    # def compute_capacity(self, T_amb, Q_rated): see comments above
    #    # Scaling of heat pump capacity based on Q_nom at A-7/W35
    #    f_scal = Q_rated / (self.coeff_capacity[0] + self.coeff_capacity[1] * (-7))
    #    self.coeff_capacity = [f_scal * c for c in self.coeff_capacity]
    #    # Eq. (6), maximal thermal capacity in kW based on source temperature
    #    return self.coeff_capacity[0] + self.coeff_capacity[1] * T_amb

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

    def sim_ts(self, ts, u, DT=0.25):
        """
        Simulate device behavior for one time step

        Rho represents whether the device is running or not.
        When the device is operating and is blocked before reaching the upper temperature bound,
        it is assumed that it will stay off until the storage reaches the lower temperature bound
        (in contrast to continuing operating once it is unblocked again).

        :param ts: timestamp
        :param u: blocking signal (0 - blocked, 1 - unblocked)
        :param DT: duration of one time step in h
        :return: power
        :rtype: float
        """
        # Hysteresis control
        if self.T_prev < self.arr_T_LO[ts]:
            self.rho = 1  # turn on if below lower bound
        elif self.T_prev > self.arr_T_UP[ts]:
            self.rho = 0  # turn off if above upper bound

        # Blocking signal and heating system active;
        # the device only operates if the internal state is ON, it is not blocked, and the heating system is activated
        if u == 0 or self.arr_heat_sys_act[ts] == 0:
            self.rho = 0

        # Supplied thermal energy
        power = self.P_rated * self.rho  # [kW]
        Q_in = power * self.arr_COP[ts] * DT  # [kWh]
        # TODO modulating heat pumps if needed

        # Extracted heat for space heating (SH)
        Q_out = self.arr_q_shd[ts] if self.arr_heat_sys_act[ts] == 1 else 0  # [kWh]

        # Storage heat losses
        Q_loss = self.Lambda_tank * (self.T_prev - self.T_env) * DT  # [kWh]

        # Overall change in contained energy and temperature
        d_Q = Q_in - Q_out - Q_loss  # [kWh] Eq. (3.18) in [3]
        d_T = d_Q / (self.V_tank * dens_w * c_w)  # [°C] Eq. (3.17) in [3]

        # Compute water temperature in the tank
        T = self.T_prev + d_T  # [°C] Eq. (3.16) in [3]
        # Note: we do not model the thermal inertia of the building --> the temperature in the tank may drop to low values,
        # while the temperature in the building is expected to be sufficiently high

        # Update
        self.T_prev = T

        return power


class Ev:
    def __init__(self, c_id, dict_cust_config, df_data):
        self.c_id = c_id
        self.P_rated = dict_cust_config["charging_rate_EV"]  # [kW]
        self.efficiency = dict_cust_config["charging_efficiency_EV"]  # charging efficiency
        self.capacity = dict_cust_config["battery_capacity_EV"]  # [kWh]
        self.SOC_prev = 1  # state-of-charge at the end of the previous time step

        # Time series data
        self.is_home = df_data["EV_home"]  # series with time index
        self.SOC_arrival_home = df_data["SOC_arrival_home"]  # series with time index
        self.time_steps_until_departure = df_data["time_steps_until_departure"]  # series with time index
        self.desired_SOC = df_data["desired_SOC"]  # series with time index

        return

    def sim_ts(self, ts, u, DT=0.25):
        """
        Simulate device behavior for one time step

        :param ts: timestamp
        :param u: charging signal (0 - do not charge, 1 - charge)
        :param DT: duration of one time step in h
        :return: power
        :rtype: float
        """
        # We do not model losses --> only do computations when EV charges
        if u == 0 or not self.is_home[ts]:
            return 0
        else:
            # Change in state-of-charge when charging at rated power
            delta_SOC_max = (self.P_rated * self.efficiency * DT) / self.capacity

            # Limit the SOC to 1
            delta_SOC = min(delta_SOC_max, 1 - self.SOC_prev)

            # Power
            power = (delta_SOC * self.capacity) / (self.efficiency * DT)
            # power below P_rated if charging at P_rated would lead to SoC > 1

            # Update previous SOC value for the next time step
            self.SOC_prev = self.SOC_prev + delta_SOC

            return power
