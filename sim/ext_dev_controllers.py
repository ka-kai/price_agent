import ast
import datetime
import math


class NoBlocking:
    """
    Benchmark scenario: None of the devices is blocked
    """
    def __init__(self):
        pass

    def get_control(self, c, **kwargs):
        """
        :param c: customer object
        :return:
        """
        WH_u = 1 if c.has_wh else None
        HP_u = 1 if c.has_hp else None
        EV_u = 1 if c.has_ev else None
        return WH_u, HP_u, EV_u


class Ripple:
    """
    Rule-based controller: implements the current ripple control signals which only depend on the current time.
    """
    def __init__(self, df_cust_config, tz_local):
        """
        :param df_cust_config: dataframe with customer configuration incl. the ripple control signals
        :param tz_local: local time zone
        """
        self.df_cust_config = df_cust_config
        self.tz_local = tz_local

    def get_control(self, ts, c, **kwargs):
        """
        Computes the control actions.

        :param ts: timestamp
        :param c: customer object
        :return: signal for the EWH, HP, and EV
        """
        # Ripple control signal depends on the local time
        ts_local = ts.astimezone(tz=self.tz_local)
        if c.has_wh:
            # Determine whether EWH is blocked or not
            WH_u = 0  # blocked is the default
            WH_cmds = ast.literal_eval(self.df_cust_config.loc[c.id, "cmds_nov-mar_WH"]) \
                if ((ts_local.month <= 3) or (ts_local.month >= 11)) \
                else ast.literal_eval(self.df_cust_config.loc[c.id, "cmds_apr-oct_WH"])  # the time window is different in summer and winter
            for times in WH_cmds:
                start, end = [datetime.datetime.strptime(time, "%H:%M").time() for time in times]
                if (ts_local.time() >= start) & (ts_local.time() < end):  # forward-looking time convention (timestamp 00:00 --> time interval 00:00 - 00:15)
                    WH_u = 1
        else:
            WH_u = None

        if c.has_hp:
            # Determine whether HP is blocked or not
            # If flag_HP_blocked is True, the HP is blocked on weekdays between 11 am and noon
            HP_u = 1  # unblocked is the default
            if (self.df_cust_config.loc[c.id, "flag_HP_blocked"]) \
                    & ((ts_local.time() >= datetime.time(11, 0))
                    & (ts_local.time() < datetime.time(12, 0))
                    & (ts_local.weekday() < 5)):
                HP_u = 0
        else:
            HP_u = None

        EV_u = 1 if c.has_ev else None

        return WH_u, HP_u, EV_u


class DynamicPriceThreshold:
    """
    Rule-based controller: The price for the next time step is compared to the price forecast for the remaining 23.75 hours.
    The WH is unblocked if the price is among the (K - K_block_day) lowest prices.
    The HP is unblocked if the price is among the (K - K_block_day) lowest prices.
    The EV charges if the price is among the (number of time steps required to reach desired SOC) lowest prices within the (number of time steps until departure) next time steps.
    """
    def __init__(self):
        pass

    def get_control(self, c, ts, price_next, price_24h_fc, DT, **kwargs):
        """
        :param c: customer object
        :param datetime.datetime ts: timestamp
        :param float price_next: price for the next time step (between -1 and 1)
        :param list price_fc: price for the remaining 23.75 h (between -1 and 1)
        :return:
        """
        price_next = round(price_next, 4)  # just in case there are inaccuracies
        # WH
        limit = sorted(price_24h_fc)[c.K - c.wh.K_block_day - 1]  # -1 bc index starts at 0
        WH_u = 1 if price_next <= round(limit, 4) else 0  # 1 - not blocked, 0 - blocked; unblock as early as possible if values are the same

        # HP
        limit = sorted(price_24h_fc)[c.K - c.hp.K_block_day - 1]  # -1 bc index starts at 0
        HP_u = 1 if price_next <= round(limit, 4) else 0  # 1 - not blocked, 0 - blocked; unblock as early as possible if values are the same

        # EV
        if c.ev.is_home[ts]:
            # Compute number of time steps the EV needs to charge to reach the desired SOC
            n_ts_charge = int(math.ceil((c.ev.desired_SOC[ts] - c.ev.SOC_prev) * c.ev.capacity / (c.ev.P_rated * c.ev.efficiency * DT)))

            # The EV does not have to charge anymore
            if n_ts_charge == 0:
                EV_u = 0
            else:
                # Number of time steps until departure
                n_ts_departure = int(min(c.ev.time_steps_until_departure[ts], c.K))

                # Determine whether to charge or not
                # TODO implement minimum_soc_factor if needed
                limit = sorted(price_24h_fc[:n_ts_departure])[n_ts_charge - 1]  # -1 bc index starts at 0
                EV_u = 1 if price_next <= round(limit, 4) or n_ts_charge == n_ts_departure else 0  # 1 - charge, 0 - do not charge; charges as early as possible if values are the same
        else:
            EV_u = 0

        return WH_u, HP_u, EV_u
