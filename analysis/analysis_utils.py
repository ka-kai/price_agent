import json
import pandas as pd
import warnings


def read_time_series_from_file(path, tz):
    """
    :param pathlib.Path path:   file to be read
    :param str tz:              desired timezone for the output (string or pytz object)
    :return:    dataframe with datetime index and the specified timezone
    :rtype:     pd.DataFrame
    """
    df = pd.read_csv(path, sep=";", index_col="time")
    df.index = pd.DatetimeIndex(df.index).tz_convert(tz)
    dt = df.index[1] - df.index[0]
    try:
        df.index.freq = dt
    except:
        warnings.warning("The sample frequency of the data is not consistent throughout the whole dataframe.")
    return df


def print_metrics(path_res, dict_res, list_metrics):
    """
    :param pathlib.Path path_res: path to the results folder
    :param dict dict_res: dictionary with the directories of the results
    :param list_metrics: which metrics to print
    :return:
    """
    # Get max value for no control case
    with open(path_res / dict_res["none"]["dir"] / "metrics.txt", "r") as f:
        dict_metrics_none = json.load(f)
    max_p_tot_none = dict_metrics_none["max_p_tot_eval"]

    for k, v in dict_res.items():
        # Read the metrics
        with open(path_res / v["dir"] / "metrics.txt", "r") as f:
            dict_metrics = json.load(f)

        # Print
        print(f"\nMetrics for {k}:")
        list_metrics_ = []
        for m in list_metrics:  # complete the key with threshold
            for key in dict_metrics.keys():
                if m in key:
                    list_metrics_.append(key)
                    continue
        for m in list_metrics_:
            print(f"{m}: {dict_metrics[m]}")

        # Print overall consumption
        df = pd.read_csv(path_res / v["dir"] / "df_evaluation.csv", sep=";", index_col="time")
        df.index = pd.DatetimeIndex(df.index)
        print(f"overall consumption: {round(df['p_tot'].sum() * ((df.index[1] - df.index[0]) / pd.Timedelta(hours=1)), 2)} kWh")

        # Comparison to no control
        if k != "none":
            print(f"-----\nComparison with no control case:")
            print(f"diff max_p_tot: {round(dict_metrics['max_p_tot_eval'] - max_p_tot_none,2)} kW")
            print(f"diff max_p_tot in %: {round((dict_metrics['max_p_tot_eval'] - max_p_tot_none) / max_p_tot_none * 100,2)} %")

    return
