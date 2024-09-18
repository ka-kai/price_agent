from matplotlib import dates as mdates
import pandas as pd

from .utils import *


def plot_mean_daily(dict_data,
                    list_cols,
                    flag_individual=False,
                    flag_max_abs=False,
                    fontsize_ylabel=None,
                    ylim_factor=1.1,
                    ylim_min_min=0,
                    ylim_max_min=0,
                    alpha=1,
                    args_legend=None,
                    lw_legend_frame=0.8,
                    figsize=(5, 2.5),
                    title=None,
                    path_output=None,
                    ):
    """
    Mean daily profiles plot.

    :param dict dict_data: dictionary with the label as key, and tuple(dataframe, color) as value
    :param list list_cols: list of columns to be plotted [[column name in the dataframe, ylabel], ...]
    :param bool flag_individual: if True, plot individual lines for each day
    :param bool flag_max_abs: if True, add a horizontal line for the maximum absolute value of the data
    :param fontsize_ylabel: fontsize of the y-axis labels
    :param float alpha: transparency of the lines
    :param float ylim_factor: factor y-axis limits
    :param float ylim_min_min: minimum minimum value for the y-axis
    :param float ylim_max_min: minimum maximum value for the y-axis
    :param dict args_legend: arguments for the legend
    :param float lw_legend_frame: linewidth of the legend frame
    :param tuple figsize: width, height in inches
    :param str title: title of the plot
    :param pathlib.Path path_output: output path, including the filename
    :return: (fig, axs)
    """
    # Timezone
    tz = next(iter(dict_data.values()))[0].index.tzinfo  # use information from first df

    # Initialize
    fig, axs = plt.subplots(figsize=figsize, nrows=len(list_cols), ncols=1, sharex=True)

    # Plot
    for label, (df, color) in dict_data.items():
        assert df.index.tzinfo == tz, f"Timezone info of {label} is {df.index.tzinfo}, but should be {tz}"
        # Compute mean power for each 15-min interval in a day
        df = df.copy()
        df["t"] = df.index.time
        df_mean = df.groupby("t").mean()
        dummy_daterange = pd.date_range(start="2000-01-01", end="2000-01-02", freq=f"15min", tz=tz, inclusive="left")  # random date, we only use the time
        df_mean.index = dummy_daterange
        # Iterate over the specified columns
        for i, (col, ylabel) in enumerate(list_cols):
            if col in df_mean.columns:  # potentially, not all dfs in dict_data have all columns
                axs[i].plot(df_mean[col], color=color, alpha=alpha, label=label)
                # Add individual lines for each day
                if flag_individual:
                    df["d"] = df.index.date
                    df_daily = pd.pivot_table(df, index="t", columns="d", values=col, dropna=False)
                    df_daily.index = dummy_daterange
                    axs[i].plot(df_daily, color=color, alpha=0.2, linewidth=0.5)
                # Add horizontal line for the maximum absolute value
                if flag_max_abs and "power" in col.lower():
                    max_abs = df[col].abs().max()
                    axs[i].axhline(y=max_abs, color=color, linestyle="--", alpha=alpha, linewidth=0.5)
                    axs[i].axhline(y=-max_abs, color=color, linestyle="--", alpha=alpha, linewidth=0.5)
            # Format axis
            axs[i].set_ylabel(ylabel, fontsize=fontsize_ylabel)
            axs[i].grid(True, axis="both", which="major")
            axs[i].tick_params(length=3, which="major")
            axs[i].tick_params(length=2, which="minor")

    # Format
    axs[-1].set_xlabel(f"Time of day (time zone: {tz})")
    axs[-1].xaxis.set_major_locator(mdates.HourLocator(interval=4, tz=tz))
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    axs[-1].xaxis.set_minor_locator(mdates.HourLocator(interval=1, tz=tz))
    axs[-1].xaxis.set_minor_formatter(mdates.DateFormatter("", tz=tz))
    for ax in axs:
        ylim_min, ylim_max = ax.get_ylim()
        ax.set_ylim(min(ylim_min_min, ylim_min * ylim_factor), max(ylim_max_min, ylim_max * ylim_factor))

    # Legend
    lines, labels = axs[0].get_legend_handles_labels()
    if args_legend is None:
        legend_frame = fig.legend(lines, labels, loc="upper left", bbox_to_anchor=(1.02, 1)).get_frame()
    else:
        legend_frame = fig.legend(lines, labels, **args_legend).get_frame()
    legend_frame.set_linewidth(lw_legend_frame)

    # Title
    fig.suptitle(title)

    # Save
    if path_output is not None:
        fig.savefig(path_output)

    return fig, axs
