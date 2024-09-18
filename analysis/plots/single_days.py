import matplotlib.dates as mdates
import pandas as pd

from .utils import *
from .time_series import subplots


def plot_single_days(path_res, dict_res, path_output, tz_plots, flag_cust_res,
                     flag_hp=True,
                     file_res="df_evaluation.csv",
                     list_days=None,
                     list_colors=list(dict_colors.values()),
                     lw_legend_frame=None,
                     ts_start=None,
                     ts_end=None
                     ):
    """
    Creates a plot for the day with the highest absolute total power, the lowest inflexible load, and the highest inflexible load,
    and saves it at the location <path_output>.

    :param pathlib.Path path_res: path to the results folder
    :param dict dict_res: dictionary with the directories of the results
    :param pathlib.Path path_output: path to the output folder
    :param str tz_plots: timezone used for plotting
    :param bool flag_cust_res:  whether to plot percentage of unblocked devices;
                                requires that individual customer results have been saved
    :param bool flag_hp: whether to plot heat pump results or not
    :param str file_res: csv file with simulation results
    :param list list_days: option to pass specific days to plot; format [(day in year, year, desired plot name), (...), ...]
    :param list list_colors: colors to be used
    :param float lw_legend_frame: linewidth of the legend frame
    :param datetime.datetime ts_start: start of time period to consider
    :param datetime.datetime ts_end: end of time period to consider (not included)
    :return: (fig, axs)
    """
    # Read the RL results
    path_rl = path_res / dict_res["rl"]["dir"]
    df = pd.read_csv(path_rl / file_res, sep=";", index_col="time")
    df.index = pd.DatetimeIndex(df.index).tz_convert(tz_plots)
    if ts_start is not None:
        df = df.loc[df.index >= ts_start]
    if ts_end is not None:
        df = df.loc[df.index < ts_end]
    if flag_cust_res:
        if (path_rl / "df_whhps.csv").is_file():  # Results from vectorized implementation
            df_whhps = pd.read_csv(path_rl / "df_whhps.csv", sep=";", index_col=0, header=[0, 1])
            df_whhps.index = pd.DatetimeIndex(df_whhps.index).tz_convert(tz_plots)
            # Not all customers necessarily have an EWH and HP
            # if a customer does not have an EWH or HP, the corresponding columns do not exist
            df_wh = df_whhps.loc[df.index, (slice(None), "WH_u")]
            df["EWHs unblocked (%)"] = df_wh.sum(axis=1) / df_wh.shape[1] * 100
            df_hp = df_whhps.loc[df.index, (slice(None), "HP_u")]
            df["HPs unblocked (%)"] = df_hp.sum(axis=1) / df_hp.shape[1] * 100
        else:  # Results from individual customer implementation; assumption is that all customers have an EWH and HP
            df_ = df.copy()
            list_c_ids = []
            for f in (path_rl).rglob("*.csv"):
                if "evaluation" in f.name:  # Skip df_evaluation.csv and df_evaluation_full.csv
                    pass
                else:
                    c_id = f.name[f.name.find("_") + 1:f.name.find(".")]  # Customer id
                    list_c_ids.append(c_id)
                    df_cust =pd.read_csv(path_rl / f, sep=";", index_col="time")
                    df_cust.index = pd.DatetimeIndex(df_cust.index).tz_convert(tz_plots)
                    df_ = df_.join(df_cust[["WH_u", "HP_u"]].rename(columns={"WH_u": f"WH_u_{c_id}",
                                                                             "HP_u": f"HP_u_{c_id}"}),
                                   how="left")
            df["EWHs unblocked (%)"] = df_[[f"WH_u_{c_id}" for c_id in list_c_ids]].sum(axis=1) / len(list_c_ids) * 100
            df["HPs unblocked (%)"] = df_[[f"HP_u_{c_id}" for c_id in list_c_ids]].sum(axis=1) / len(list_c_ids) * 100

    # Read the results without any control
    if "none" in dict_res.keys() and dict_res["none"]["dir"] is not None:
        df_none = pd.read_csv(path_res / dict_res["none"]["dir"] / file_res, sep=";", index_col="time")
        df_none.index = pd.DatetimeIndex(df_none.index).tz_convert(tz_plots)
        df["$P^{\mathrm{tot}}_{\mathrm{none}}$"] = df_none["p_tot"]
        df["$P^{\mathrm{EWH}}_{\mathrm{none}}$"] = df_none["p_wh"]
        df["$P^{\mathrm{HP}}_{\mathrm{none}}$"] = df_none["p_hp"]
        df["$P^{\mathrm{EV}}_{\mathrm{none}}$"] = df_none["p_ev"]

    if list_days is None:
        # Plot extreme days; each plot also includes the previous and the following day
        list_days = []
        # Day with the highest absolute p_tot
        list_days.append([df["p_tot"].abs().idxmax().dayofyear, df["p_tot"].abs().idxmax().year, "highest_p_tot"])
        # Day with the lowest p_infl
        list_days.append([df["p_infl"].idxmin().dayofyear, df["p_infl"].idxmin().year, "lowest_p_infl"])
        # Day with the highest p_infl
        list_days.append([df["p_infl"].idxmax().dayofyear, df["p_infl"].idxmax().year, "highest_p_infl"])

    for day, year, name in list_days:
        df_ = df[(df.index.dayofyear >= day - 1) & (df.index.dayofyear <= day + 1) & (df.index.year == year)].copy()

        # Determine the y limits for the device power profiles
        cols = [col for col in df_.columns if "p_wh" in col or "P^{\mathrm{EWH}" in col
                                            or "p_hp" in col or "P^{\mathrm{HP}" in col
                                            or "p_ev" in col or "P^{\mathrm{EV}" in col]
        y_lim_p_dev = (min(min(df_[cols].min() * 1.1), -5), max(df_[cols].max()) * 1.1)

        # Rename the columns to the desired labels
        df_.rename(columns={"price_n": "Normalized $\lambda_t$",
                            "p_tot": "$P^{\mathrm{tot}}_{\lambda}$",
                            "p_infl": "$P^{\mathrm{infl}}$",
                            "p_wh": "$P^{\mathrm{EWH}}_{\lambda}$",
                            "p_hp": "$P^{\mathrm{HP}}_{\lambda}$",
                            "p_ev": "$P^{\mathrm{EV}}_{\lambda}$"}, inplace=True)

        # Define the plot settings
        date_form = "%y-%m-%d"  # %H:%M
        xaxis_locator = ["day", 1]
        if "none" in dict_res.keys() and dict_res["none"]["dir"] is not None:
            list_subplots_ = [
                [["Normalized $\lambda_t$"], "", (-1.3, 1.3), 1],
                [["$P^{\mathrm{infl}}$", "$P^{\mathrm{tot}}_{\mathrm{none}}$", "$P^{\mathrm{tot}}_{\lambda}$"], "", None, 3],
                [["$P^{\mathrm{EWH}}_{\mathrm{none}}$", "$P^{\mathrm{EWH}}_{\lambda}$"], "", y_lim_p_dev, 2],
                [["$P^{\mathrm{HP}}_{\mathrm{none}}$", "$P^{\mathrm{HP}}_{\lambda}$"], "", y_lim_p_dev, 2],
                [["$P^{\mathrm{EV}}_{\mathrm{none}}$", "$P^{\mathrm{EV}}_{\lambda}$"], "", y_lim_p_dev, 2]]
        else:
            list_subplots_ = [
                [["Normalized $\lambda_t$"], "", (-1.3, 1.3), 1],
                [["$P^{\mathrm{infl}}$", "$P^{\mathrm{tot}}_{\lambda}$"], "", None, 3],
                [["$P^{\mathrm{EWH}}_{\lambda}$"], "", y_lim_p_dev, 2],
                [["$P^{\mathrm{HP}}_{\lambda}$"], "", y_lim_p_dev, 2],
                [["$P^{\mathrm{EV}}_{\lambda}$"], "", y_lim_p_dev, 2]]
        if flag_cust_res and flag_hp:
            height_ratios = [1, 1, 1, 1, 1, 0.78, 0.78]
        elif flag_cust_res and not flag_hp:
            height_ratios = [1, 1, 1, 1, 0.78]
        else:
            height_ratios = None

        # Add blocking information
        if flag_cust_res:
            list_subplots_ += [[["EWHs unblocked (%)"], "", (-10, 110), 1],
                              [["HPs unblocked (%)"], "", (-10, 110), 1]]
        # Remove HP information
        if flag_hp:
            list_subplots = list_subplots_
        else:
            list_subplots = []
            for i, l in enumerate(list_subplots_):
                if "HP" in l[0][0]:
                    pass
                else:
                    list_subplots.append(l)

        # Plot
        fig, axs = subplots(df=df_,
                            date_form=date_form,
                            xaxis_locator=xaxis_locator,
                            figsize=(3.3, 2) if flag_hp else (3.3, 1.5),  # width, height in inches
                            list_subplots=list_subplots,
                            height_ratios=height_ratios,
                            list_colors=list_colors,
                            lw_legend_frame=lw_legend_frame)

        # Axis formatting; subplot function is more general, that's why more specific adjustments are done here
        for ax in axs:
            for _, spine in ax.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(0.2)
            ax.tick_params(width=0.2, length=2, which="major")
            ax.tick_params(width=0.2, length=1, grid_alpha=1, which="minor")
            ax.grid(True, axis="both", which="both", linewidth=0.2)
        # By default, it uses -1, 0, and 1 as y-ticks; we want -1 and 1 as major ticks, and 0 as minor tick
        axs[0].set_yticks([-1, 1])
        axs[0].set_yticks([0], minor=True)
        # Add minor ticks
        axs[0].xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18], tz=tz_plots))
        axs[0].xaxis.set_minor_formatter(mdates.DateFormatter("", tz=tz_plots))

        # Highlight the most relevant time of the day for extreme days
        idx = None
        if name == "highest_p_tot":
            idx = df["p_tot"].abs().idxmax()
        elif name == "lowest_p_infl":
            idx = df["p_infl"].idxmin()
        elif name == "highest_p_infl":
            idx = df["p_infl"].idxmax()
        if idx is not None:
            for ax in axs:
                ax.axvline(idx, color=dict_colors["red"], linestyle="-", linewidth=2, alpha=0.2)

        # Save the plot
        fig.savefig(path_output / f"{path_rl.parts[-2]}_profiles_{name}.png")

    return fig, axs
