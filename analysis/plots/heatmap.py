"""
This code is based on:
T. Brudermueller and M. Kreft, “Smart Meter Data Analytics: Practical Use-Cases and Best Practices of Machine Learning Applications for
Energy Data in the Residential Sector,” in ICLR 2023 Workshop on Tackling Climate Change with Machine Learning, 2023.
Original code: https://github.com/bitstoenergy/iclr-smartmeteranalytics
"""

import datetime as dt
import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from typing import Optional
import warnings

from .utils import *

COLOR_AXES = "#999999"
COLOR_FACE = "#999999"

DEFAULT_FIGRATIO = 1.618
DEFAULT_FIGWIDTH = 8
DEFAULT_FIGHEIGHT = DEFAULT_FIGWIDTH / DEFAULT_FIGRATIO
DEFAULT_FIGSIZE = (DEFAULT_FIGWIDTH, DEFAULT_FIGHEIGHT)

rc_params = {
    "figure.figsize": DEFAULT_FIGSIZE,
    "font.family": "sans-serif",
    "font.size": 9,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


class CustomFigure(plt.Figure):
    ax_cbar: Optional[matplotlib.axes.Axes] = None
    ax_heatmap: Optional[matplotlib.axes.Axes] = None
    ax_histx: Optional[matplotlib.axes.Axes] = None
    ax_histx_max: Optional[matplotlib.axes.Axes] = None
    ax_histy: Optional[matplotlib.axes.Axes] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def HeatmapFigure(
    series: pd.Series,
    figsize: Optional[tuple[int, int]] = None,
    flag_histx: Optional[bool] = True,
    histx_label: Optional[str] = "Daily Energy\n(kWh)",
    histx_max_label: Optional[str] = "Daily Peak (kW)",
    histx_ylim: Optional[int] = None,
    histx_max_ylim: Optional[int] = None,
    flag_histy: Optional[bool] = True,
    histy_label: Optional[str] = "Mean Demand\nProfile (kW)",
    histy_xlim: Optional[int] = None,
    cbar_label: Optional[str] = "Power (kW)",
    annotate_suntimes: Optional[bool] = True,
    title: Optional[str] = None,
    **kwargs,
) -> CustomFigure:
    """Makes a figure with heatmap, daily overview, profile, and annotations.

    Args:
        series: Series with **Power** values and timezone aware datetme index
                with a fixed frequency. Timestamps describe the beginning of
                the interval for wich the power value is valid.
        figsize: Tuple with width and height of the figure in inches.
        flag_histx: If True, show histx.
        histx_label: Label for the x-axis of the daily energy overview.
        histx_max_label: Label for the y-axis of the daily peak overview. If
                         omitted, the daily peaks are not drawn.
        histx_ylim: Upper axis limit for the daily energy overview.
        histx_max_ylim: Upper axis limit for the daily peak overview.
        flag_histy: If True, show histy.
        histy_label: Label for the y-axis of the mean demand profile.
        histy_xlim: Upper axis limit for the mean demand profile.
        cbar_label: Label for the colorbar.
        annotate_suntimes: If True, sunrise and sunset times are annotated.
        title: Figure title.
        **kwargs: Additional keyword arguments passed to the pcolormesh plot.

    Returns:
        Custom figure subclass with the heatmap and additional axes.
    """
    interval_minutes = series.index.freq.nanos / 60e9

    # Generate the pivoted heatmap and corresponding time and date range
    data, daterange, timerange = _heatmap_data(series)

    # Set up the figure and axes
    fig = CustomFigure(figsize=figsize)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(8, 1),
        height_ratios=(2, 7),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.01,
        hspace=0.01 * DEFAULT_FIGRATIO,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_histy = fig.add_subplot(gs[1, 1])

    if flag_histx or flag_histy:
        ax_histx_twin = _plot_hists(
            daterange,
            timerange,
            data,
            ax_histx,
            ax_histy,
            interval_minutes,
            histx_label=histx_label,
            histx_max_label=histx_max_label,
            histy_label=histy_label,
            histx_ylim=histx_ylim,
            histx_max_ylim=histx_max_ylim,
            histy_xlim=histy_xlim
        )

    if flag_histx and flag_histy:
        # location of the color bar
        ax_cbar = ax_histx.inset_axes([1.07, 0, 0.035, 1])  # [x0, y0, width, height] Lower-left corner of inset Axes, and its width and height.
    elif flag_histx and not flag_histy:
        ax_histy.set_visible(False)
        ax_cbar = ax_histx.inset_axes([1.07, 0, 0.035, 1])
    elif not flag_histx and flag_histy:
        ax_histx.set_visible(False)
        ax_histx_twin.set_visible(False)
        ax_cbar = ax.inset_axes([1.2, 0.6, 0.035, 0.4])
    else:
        ax_histx.set_visible(False)
        ax_histy.set_visible(False)
        ax_cbar = ax.inset_axes([1.1, 0.6, 0.04, 0.4])

    mesh = plot_pcolormesh(ax, daterange, timerange, data, **kwargs)

    cbar = fig.colorbar(mesh, cax=ax_cbar, label=cbar_label)
    cbar.outline.set_color(COLOR_AXES)
    cbar.outline.set_linewidth(0)
    ax_cbar.tick_params(color=COLOR_AXES, rotation=90)

    if annotate_suntimes:
        pass  # removed in this version

    fig.suptitle(title)
    fig.ax_cbar = ax_cbar
    fig.ax_heatmap = ax
    fig.ax_histx = ax_histx
    fig.ax_histy = ax_histy
    if flag_histx:
        fig.ax_histx_max = ax_histx_twin

    return fig


def _heatmap_data(series: pd.Series):
    """Get day x time-of-day matrix and date-/timeranges from series
    """

    data = series.copy()
    timezone = data.index.tz
    # TODO
    # Why does this not make it work with pivot without aggfunc?
    # df.drop_duplicates(subset='Timestamp', keep='first', inplace=True)
    # Use multiindices and unstack instead?
    # https://pandas.pydata.org/docs/reference/api/pandas.Series.unstack.html
    df = data.to_frame(name="values")
    df["d"] = df.index.date
    df["t"] = df.index.time
    # data = df.pivot(index="to_time", columns="to_date", values='A+')
    data = df.pivot_table(
        index="t",
        columns="d",
        values="values",
        aggfunc=lambda x: x.iloc[0],
        # aggfunc=lambda x: x.sum(skipna=False),
        dropna=False
    )
    # Construct daterange with timezone
    daterange = data.columns.astype("datetime64[ns]").tz_localize(timezone)
    # Add one day to the end because pcolormesh requires edges
    daterange = pd.date_range(start=daterange.min(), end=daterange.max() + dt.timedelta(days=1), tz=timezone)
    # Construct timerange with frequency and timezone
    timerange = pd.date_range(
        start="1970-01-01T00:00:00",
        end="1970-01-02T00:00:00",
        freq=f"{series.index.freq.nanos}ns",
        tz=timezone,
    )

    data = data.to_numpy()
    return data, daterange, timerange


def plot_pcolormesh(ax, daterange, timerange, data, **kwargs):
    """
    Plot the 2D demand profile
    Take a numpy matrix and indices and make a figure with heatmap and sum/avg
    """

    mesh = ax.pcolormesh(daterange, timerange, data, **kwargs)
    ax.set_xlim(daterange[0], daterange[-1])
    ax.invert_yaxis()

    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz=timerange.tz))
    # ax.xaxis.set_major_locator(mdates.MonthLocator((1, 4, 7, 10), tz=timerange.tz))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(tz=timerange.tz))

    # Alternative format: '%#H' if os.name == 'nt' else '%-H'
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%H", tz=timerange.tz))
    ax.yaxis.set_major_locator(mdates.HourLocator(tz=timerange.tz, interval=6))
    ax.yaxis.set_minor_locator(mdates.HourLocator(tz=timerange.tz))

    # Remove last x tick label, to avoid overlapp with histogram
    ax.xaxis.get_majorticklabels()[-1].set_visible(False)

    ax.set_xlabel("Date")
    ax.set_ylabel("Hour")

    # Hide axes frame lines and set color
    for pos in ["left", "bottom", "top", "right"]:
        ax.spines[pos].set_color(COLOR_AXES)
    ax.tick_params(axis="both", which="both", color=COLOR_AXES)
    for pos in ["top", "right"]:
        ax.spines[pos].set_visible(False)

    return mesh


def _plot_hists(
    daterange,
    timerange,
    data,
    ax_histx,
    ax_histy,
    interval_minutes,
    histx_label=None,
    histx_max_label=None,
    histy_label=None,
    histx_ylim=None,
    histx_max_ylim=None,
    histy_xlim=None,
):
    # Daily max
    if histx_max_label:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            daily_max_draw = np.nanmax(data, axis=0)
        twinx = ax_histx.twinx()
        twinx.set_ylabel(histx_max_label, labelpad=0)
        twinx.scatter(
            daterange[:-1] + dt.timedelta(hours=12),
            daily_max_draw,
            color="black",
            s=1,
            linewidths=0,
        )
        twinx.set_ylim(0, histx_max_ylim)

        for pos in ["top", "left", "bottom"]:
            twinx.spines[pos].set_visible(False)
        twinx.spines["right"].set_color(COLOR_AXES)
        # Rotate in case they are long
        twinx.tick_params(color=COLOR_AXES, rotation=90)
    else:
        twinx = None

    # Daily sum
    daily_demand = np.nansum(data, axis=0) * interval_minutes / 60
    ax_histx.fill_between(
        daterange[:-1] + dt.timedelta(hours=12),
        daily_demand,
        facecolor=COLOR_FACE,
        alpha=0.5,
    )
    ax_histx.set_xlim(daterange[0], daterange[-1])
    # Need to set the max here as well, else when removing the lower tick (below) the limits get extended, since the ticks have not been rendered yet.
    if histx_ylim is None:
        ax_histx.set_ylim(min(0, daily_demand.min()), daily_demand.max())
    else:
        ax_histx.set_ylim(min(0, daily_demand.min()), histx_ylim)

    ax_histx.set_ylabel(histx_label)
    ax_histx.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    # Remove fist label because it may overlap with heat map
    ax_histx.yaxis.get_majorticklabels()[0].set_visible(False)

    # Mean profile
    ax_histy.fill_betweenx(
        timerange[:-1] + dt.timedelta(minutes=interval_minutes) / 2,
        np.nanmean(data, axis=1),
        facecolor=COLOR_FACE,
        alpha=0.5,
    )
    ax_histy.axes.yaxis.set_ticklabels([])
    # If demand is larger than zero, always show from zero, else show from negative demand on
    ax_histy.set_xlim(min(0, np.nanmean(data, axis=1).min()), histy_xlim)
    ax_histy.set_ylim(timerange[0], timerange[-1])
    ax_histy.set_xlabel(histy_label)
    ax_histy.invert_yaxis()  # This has to be called after setting lims

    # Hide the ticks and labels
    ax_histx.get_xaxis().set_visible(False)
    ax_histy.get_yaxis().set_visible(False)
    # Hide axes frame lines
    for pos in ["top", "right", "bottom"]:
        ax_histx.spines[pos].set_visible(False)
    for pos in ["top", "right", "left"]:
        ax_histy.spines[pos].set_visible(False)
    ax_histx.spines["left"].set_color(COLOR_AXES)
    ax_histx.tick_params(color=COLOR_AXES)
    ax_histy.spines["bottom"].set_color(COLOR_AXES)
    ax_histy.tick_params(color=COLOR_AXES)

    return twinx
