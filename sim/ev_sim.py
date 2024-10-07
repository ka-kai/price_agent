import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class EVs:
    def __init__(self, config, daterange, flag_plot=False):
        # From the general config
        self.DT = config.DT
        self.K = config.K
        self.minimum_soc_factor = config.minimum_soc_factor
        self.flag_save = config.env_type == "evaluation" or config.flag_cust_res
        if self.flag_save:
            self.results = []

        # EV config
        self.df_config = pd.read_excel(config.path_ev_config, index_col="id")
        self.df_config = self.df_config[["battery_capacity", "charging_rate", "charging_efficiency", "price_fc_group"]]
        # Add the maximum SOC change per time step, and the factor to compute the average power in a time step
        self.df_config["delta_SOC_max"] = self.df_config["charging_rate"] * self.df_config["charging_efficiency"] * self.DT / self.df_config["battery_capacity"]
        self.df_config["factor_power"] = self.df_config["battery_capacity"] / (self.df_config["charging_efficiency"] * self.DT)
        # Number of EVs
        self.n_evs = len(self.df_config)
        self.evs = self.df_config.index.astype(str)
        # Append values that are needed in every time step to save computation time
        self.delta_SOC_max = self.df_config["delta_SOC_max"].values
        self.factor_comp_power = self.df_config["factor_power"].values
        self.fc_group = self.df_config["price_fc_group"].values.astype(int)

        # Time series data
        df_ts = pd.read_csv(config.path_ev_data, sep=";", index_col=0, header=[0, 1])
        df_ts.index = pd.DatetimeIndex(df_ts.index)
        # Filter for EVs that are part of the config file and sort the columns
        df_ts = df_ts.loc[:, (self.evs, slice(None))]
        # Match the data to the daterange
        start = daterange[0]
        end = daterange[-1]
        if start in df_ts.index and end in df_ts.index:
            # Only keep simulation daterange
            self.df_ts = df_ts.loc[start:end]
            # We do not have to worry about partial sessions at the beginning and end of the daterange;
            # In the simulation, the SOC is set to 100 % for daterange[0]
        else:
            # Extend the EV data to the years of the simulation; logic:
            # We insert 52 weeks of data starting from the year before the simulation daterange;
            # the start date for inserting the data is the timestamp in df_ts_ext that has the same weekday and is closest to the first timestamp in the EV data (df_ts);
            # since we consider 52 full weeks (i.e., 364 days instead of 1 year), we can continue inserting the data every 52 weeks, while the weekday/weekend pattern remains the same
            # we repeat the process until we reach the year after the simulation daterange
            # finally, we only keep the simulation daterange

            # Initialize
            if df_ts.index[-1] + datetime.timedelta(hours=self.DT) - df_ts.index[0] < datetime.timedelta(days=365):
                raise ValueError("The EV data should cover at least one year.")
            df_ts_ext = pd.DataFrame(index=pd.DatetimeIndex(pd.date_range(start=f"{start.year - 1}-01-01", end=f"{end.year + 2}-01-01",
                                                                          freq=f"{self.DT}H", tz=df_ts.index.tz, inclusive="left")),
                                     columns=df_ts.columns)  # extend to one year before and after the simulation daterange
            # Get the day in df_ts_ext that has the same day of the week as self.df_ts.index[0] and is closest to the date of self.df_ts.index[0]
            timestamps = df_ts_ext.index[(df_ts_ext.index.weekday == df_ts.index[0].weekday()) & (df_ts_ext.index.time == df_ts.index[0].time())]
            closest_timestamp = timestamps[np.argmin(np.abs(timestamps[timestamps.year == start.year - 1].day_of_year - df_ts.index[0].day_of_year))]
            # Get the data for the next 52 weeks, i.e., 364 days
            data = df_ts.loc[:df_ts.index[0] + datetime.timedelta(weeks=52, hours=-self.DT)].copy()
            # Set SOC_arrival_home to 1 for EVs that are at home at the beginning of data,
            # but SOC_arrival_home is NaN (e.g., because the csv file has been truncated)
            mask = (data.loc[data.index[0], (slice(None), "EV_home")] == 1).values & \
                   (data.loc[data.index[0], (slice(None), "SOC_arrival_home")].isna()).values
            data.loc[data.index[0], (self.evs[mask], "SOC_arrival_home")] = 1
            # Neglect the last charging session for EVs that are at home at the end of data
            for ev in self.evs:
                if data.loc[data.index[-1], (ev, "EV_home")] == 1:
                    idx_arrival = data.index[~data.loc[:, (ev, "SOC_arrival_home")].isna()]
                    data.loc[idx_arrival[-1]:, (ev, slice(None))] = [0, np.nan, np.nan, np.nan]
            # Insert the data
            s_insert = closest_timestamp
            for i in range(df_ts_ext.index[-1].year - df_ts_ext.index[0].year):
                e_insert = s_insert + datetime.timedelta(weeks=52, hours=-self.DT)
                print(f"Inserting EV data from {s_insert} to {e_insert}...")
                df_ts_ext.loc[s_insert:e_insert] = data.values
                s_insert += datetime.timedelta(weeks=52)
            # Only keep simulation daterange
            self.df_ts = df_ts_ext.loc[start:end]
        # Change dtype
        self.df_ts = self.df_ts.astype(np.float32)

        # Add columns for the simulation results
        if self.flag_save:
            cols = ["EV_p", "EV_SOC"]
            df_ = pd.DataFrame(index=self.df_ts.index, columns=pd.MultiIndex.from_product([self.evs, cols]), dtype=np.float32)
            self.df_ts = pd.concat([self.df_ts, df_], axis=1)
            self.df_ts = self.df_ts.loc[:, (self.evs, slice(None))]  # sort the columns once again

        # Initialize the SOC
        self.SOC_prev = np.full(self.n_evs, 1.0)  # previous SOC

        if flag_plot:
            self._plot()

        return

    def _plot(self):
        fig, axs = plt.subplots(figsize=(15, 6), nrows=self.n_evs, sharex=True, sharey=True)
        for i, ev in enumerate(self.evs):
            # Plot
            ts = self.df_ts.loc[:, (ev, "EV_home")].copy()
            ts[ts == 0] = np.nan
            ts.plot(ax=axs[i], linewidth=5, label=ev)

            axs[i].set_ylim(0.9, 1.1)
            axs[i].set_yticks([])  # remove y-ticks
            axs[i].set_xticks([], minor=True)  # remove minor x-ticks

            axs[i].legend(loc="center left", bbox_to_anchor=(1, 0.5), handlelength=0.2)

        fig.savefig("evs.png", dpi=300)

    def set_full_soc(self):
        self.SOC_prev = np.full(self.n_evs, 1.0)
        return

    def sim(self, ts, price_next=None, dict_price_24h_fc=None):
        """
        Simulate the EVs charging behavior;
        If price_24h_fc and price_next are not provided, the EVs are charged without any control, i.e., they charge immediately;
        If they are provided, the EVs are charged according to the dynamic price threshold control, i.e.,
        the price for the next time step is compared to the price forecast for the remaining 23.75 hours;
        the EV charges if the price is among the (number of time steps required to reach desired SOC) lowest prices
        within the (number of time steps until departure) next time steps.

        :param datetime.datetime ts: timestamp
        :param float price_next: price for the next time step (between -1 and 1)
        :param dict dict_price_24h_fc: dictionary with price forecasts for the remaining 23.75 h (between -1 and 1)
        :return:
        """
        # Get the previous SOC
        if ts == self.df_ts.index[0]:
            # All EVs are assumed to have a SOC of 100% SOC at the beginning
            self.set_full_soc()

        # For EVs that are arriving at home in this time step, the SOC is set to the arrival SOC
        evs_arrival = ~self.df_ts.xs("SOC_arrival_home", level=1, axis=1).loc[ts].isna()
        self.SOC_prev[evs_arrival] = self.df_ts.loc[ts, (self.evs[evs_arrival], "SOC_arrival_home")]

        # Compute the action
        evs_charging = self.df_ts.xs("EV_home", level=1, axis=1).loc[ts] == 1  # only consider EVs that are at home
        self.SOC_prev[~evs_charging] = np.nan  # no SOC for EVs that are not at home
        desired_SOC = self.df_ts.loc[ts, (self.evs[evs_charging], "desired_SOC")].values
        n_ts_charge = np.ceil((desired_SOC - self.SOC_prev[evs_charging]) / self.delta_SOC_max[evs_charging]).astype(int)  # number of time steps the EV needs to charge to reach the desired SOC
        evs_charging[evs_charging] = n_ts_charge > 0  # only consider EVs that have to charge

        if sum(evs_charging) > 0:  # if 0, no EV is charging
            mask = n_ts_charge > 0
            n_ts_charge = n_ts_charge[mask]
            desired_SOC = desired_SOC[mask]
            if len(dict_price_24h_fc) > 0:  # dynamic price threshold; if dict_price_24h_fc is empty, the EVs are charged without any control
                n_ts_departure = np.minimum(self.df_ts.loc[ts, (self.evs[evs_charging], "time_steps_until_departure")].values, self.K).astype(int)  # number of time steps until departure
                fc_group = self.fc_group[evs_charging]
                limit = np.round([sorted(dict_price_24h_fc[fc_group[i]][:n_ts_departure[i]])[n_ts_charge[i] - 1] for i in range(sum(evs_charging))], 4)  # -1 bc index starts at 0
                evs_charging[evs_charging] = ((price_next <= limit) | (n_ts_charge == n_ts_departure) | (self.SOC_prev[evs_charging] < self.minimum_soc_factor * desired_SOC))

        # Update the SOC and compute power
        power = np.zeros(self.n_evs)
        if sum(evs_charging) > 0:  # if 0, no EV is charging
            delta_SOC = np.minimum(self.delta_SOC_max[evs_charging], 1 - self.SOC_prev[evs_charging])
            power[evs_charging] = delta_SOC * self.factor_comp_power[evs_charging]
            self.SOC_prev[evs_charging] = self.SOC_prev[evs_charging] + delta_SOC
            # Note: we assume that the EV is charged for the whole time step, i.e., the SOC can be > desired_SOC; however, we limit the SOC to 1

        if self.flag_save:
            self.results.append((power, self.SOC_prev.tolist()))

        return sum(power)

    def save_results(self, path):
        # Process the results
        df = self.df_ts.copy()
        EV_p, EV_SOC = zip(*self.results)
        n_ts = len(EV_p)
        idx = df.index[n_ts - 1]
        df.loc[:idx, (slice(None), "EV_p")] = EV_p
        df.loc[:idx, (slice(None), "EV_SOC")] = EV_SOC

        # Save
        df.to_csv(path, sep=";", index=True, header=True, index_label="time")

        # Reset the results
        self.results = []

        return
