from matplotlib import dates as mdates
import pandas as pd

from .utils import *


def subplots(df,
             list_subplots,
             list_colors=list(dict_colors.values()),
             date_form="%d.%m.%Y",
             xaxis_locator=["day", 1],
             height_ratios=None,
             alpha=1,
             lw_legend_frame=0.8,
             figsize=(6, 3),
             title=None,
             path_output=None
             ):
    """
    Time series plot with multiple subplots.

    :param pd.DataFrame df: dataframe
    :param list list_subplots: [
        [[column(s) to plot in subplot 1],
        y-label subplot 1,
        limits y-axis subplot 1,
        number of legend columns subplot 1],
        [...]]
    :param list list_colors: colors to be used
    :param str date_form: format of the x-axis tick labels, strftime format
    :param list xaxis_locator: where to place the major ticks, e.g. ["day", 2] for a label every other day
    :param list height_ratios: height ratios of individual subplots
    :param float alpha: transparency of the lines
    :param float lw_legend_frame: linewidth of the legend frame
    :param tuple figsize: width, height in inches
    :param str title: title of the plot
    :param pathlib.Path path_output: output path, including the filename
    :return: (fig, axs)
    """
    # Timezone
    tz = df.index.tzinfo

    # Initialize
    fig, axs = plt.subplots(len(list_subplots), 1, figsize=figsize, sharex=True, height_ratios=height_ratios)
    if len(list_subplots) == 1:
        axs = [axs]

    # Plot
    for i, (list_cols, label, ylim, ncol_legend) in enumerate(list_subplots):
        for j, col in enumerate(list_cols):
            axs[i].plot(df[col], color=list_colors[-len(list_cols):][j], alpha=alpha, label=col)
        # Legend
        legend_frame = axs[i].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=ncol_legend).get_frame()
        legend_frame.set_linewidth(lw_legend_frame)
        # Format axis
        axs[i].set_ylim(ylim)
        axs[i].set_ylabel(label)

    # Format
    axs[-1].set_xlabel(f"Date (time zone: {tz})")
    loc_type, loc_int = xaxis_locator
    if loc_type == "month":
        axs[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=loc_int, tz=tz))
    elif loc_type == "day":
        axs[-1].xaxis.set_major_locator(mdates.DayLocator(interval=loc_int, tz=tz))
    elif loc_type == "hour":
        axs[-1].xaxis.set_major_locator(mdates.HourLocator(interval=loc_int, tz=tz))
        # set x_min such that the x-labels will include 00:00, and are not, e.g., 23:00  01:00 ...
        hour_x_min = (df.index[0].hour // loc_int) * loc_int
        axs[-1].set_xlim([df.index[0].replace(hour=hour_x_min, minute=0), df.index[-1]])
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter(date_form, tz=tz))

    # Title
    fig.suptitle(title)

    # Save
    if path_output is not None:
        fig.savefig(path_output)

    return fig, axs
