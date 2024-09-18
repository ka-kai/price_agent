import numpy as np
import pandas as pd


class ProportionalPriceInflPerDay:
    """
    Rule-based controller:
    The price for the next time interval scales linearly with respect to the inflexible load in the next time interval,
    scaling is done per day in local time.
    """

    def __init__(self, p_infl, config):
        # Scale the inflexible load to [-1, 1] on a daily basis in local time
        p_infl_local = p_infl.copy()  # copy to avoid changing the original data; original index used for final pd.Series
        p_infl_local.index = p_infl_local.index.tz_convert(config.tz_local)
        # TODO: possible improvement: combine first and last day; when time shift on last day is, e.g., 2 hours --> currently only those two hours are treated as a day
        for date in np.unique(p_infl_local.index.date):
            data = p_infl_local[p_infl_local.index.date == date]
            p_infl_local[p_infl_local.index.date == date] = ((data - data.min()) / (data.max() - data.min()) * 2 - 1)

        # Discretized price values can be {0, 1, ... , self.config.a["n_prices"] - 1};
        vals = np.linspace(-1, 1, config.a["n_prices"] + 1)[:-1]
        actions = [idx for idx in np.digitize(x=p_infl_local, bins=vals) - 1]  # indices returned by np.digitize start with 1 --> that's why we have -1
        # E.g., for n_prices = 2:
        # vals = [-1, 0]; then the actions are:
        # 0 if -1 <= p_infl_local < 0
        # 1 if  0 <= p_infl_local <= 1

        self.actions = pd.Series(data=actions, index=p_infl.index)

        return

    def predict(self, ts):
        """
        The method is called 'predict' to be in line with the RL method to determine the price.

        :param ts: timestamp at the beginning of the time interval for which the price is determined
        :return: action
        :rtype: int
        """

        return self.actions[ts]
