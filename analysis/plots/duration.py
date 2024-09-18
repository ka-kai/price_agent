from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from .utils import *


def duration_curve(
        df,
        col,
        xlabel="Share of time (%)",
        ylabel="Power (kW)",
        fig=None,
        ax=None,
        color=None,
        label=None,
        flag_axins_max=False,
        flag_axins_min=False,
        axins_max=[0.48, 0.65, 0.48, 0.32],
        axins_min=[0.12, 0.1, 0.48, 0.32],
        axins_max_cords=None,
        axins_min_cords=None,
        axins_x_loc=0.4,
        axins_y_loc=20,
        figsize=(5, 2),
):
    """
    Plots the values in column <col> in descending order;
    How to read? E.g.: For 2 % of the time, the load is above 40 kW

    :param pd.DataFrame df: data
    :param str col: column for which the duration curve is plotted
    :param str xlabel: label for the x-axis
    :param str ylabel: label for the y-axis
    :param plt.figure.Figure fig: figure object
    :param ax: figure axis
    :param str color: color of the line
    :param str label: label for the line
    :param bool flag_axins_max: whether to zoom in on the left end of the duration curve
    :param bool flag_axins_min: whether to zoom in on the right end of the duration curve
    :param list axins_max: [x0, y0, width, height]; lower-left corner of inset axes, and its width and height for maximum load
    :param list axins_min: [x0, y0, width, height]; lower-left corner of inset axes, and its width and height for minimum load
    :param list axins_max_cords: [x1, x2, y1, y2]; x and y limits for the maximum load inset
    :param list axins_min_cords: [x1, x2, y1, y2]; x and y limits for the minimum load inset
    :param float axins_x_loc: major locator for the x-axis of the inset
    :param float axins_y_loc: major locator for the y-axis of the inset
    :param tuple figsize: width, height in inches
    :return: (fig, ax, axins_max, axins_min)
    """
    # Determine values
    sorted_load = df[col].sort_values(ascending=False).dropna().values  # sort in descending order
    pct = np.arange(1, len(sorted_load) + 1) / len(sorted_load) * 100  # x axis, in percent

    # Plot; a new figure is only created when the function is called the first time
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        # Format the axes
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        list_axins = []
        if flag_axins_max:
            axins_max = ax.inset_axes(axins_max)
            list_axins.append((axins_max, axins_max_cords))
        if flag_axins_min:
            axins_min = ax.inset_axes(axins_min)
            list_axins.append((axins_min, axins_min_cords))
        for axins, cords in list_axins:
            x1, x2, y1, y2 = cords
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xlabel("")
            axins.tick_params(axis="both", which="both", pad=0.001)
            axins.grid(True)

    ax.plot(pct, sorted_load, color=color, label=label)
    list_axins = []
    if flag_axins_max:
        list_axins.append((axins_max, 0))
    if flag_axins_min:
        list_axins.append((axins_min, -1))
    for axins, idx in list_axins:
        axins.plot(pct, sorted_load, color=color, linewidth=0.4)
        axins.scatter(pct[idx], sorted_load[idx], marker="x", color=color, s=5)  # s is the marker size
        axins.xaxis.set_major_locator(MultipleLocator(axins_x_loc))
        axins.yaxis.set_major_locator(MultipleLocator(axins_y_loc))
        axins.set_ylabel("")

    # Legend
    legend_frame = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1)).get_frame()
    legend_frame.set_linewidth(0.8)

    return fig, ax, axins_max, axins_min


def plot_multiple_duration_curves(dict_data,
                                  col,
                                  xlabel="Share of time (%)",
                                  ylabel="Power (kW)",
                                  flag_axins_max=False,
                                  flag_axins_min=False,
                                  axins_max=None,
                                  axins_min=None,
                                  axins_max_cords=None,
                                  axins_min_cords=None,
                                  axins_x_loc=0.4,
                                  axins_y_loc=20,
                                  xlim=(-2, 102),
                                  ylim=None,
                                  figsize=(5, 2),
                                  title=None,
                                  path_output=None
                                  ):
    """

    :param dict dict_data:  dictionary with the label as key, and tuple(dataframe, color) as value
    :param str col: column to plot
    :param str xlabel: label for the x-axis
    :param str ylabel: label for the y-axis
    :param bool flag_axins_max: whether to zoom in on the left end of the duration curve
    :param bool flag_axins_min: whether to zoom in on the right end of the duration curve
    :param list axins_max: [x0, y0, width, height]; lower-left corner of inset axes, and its width and height for maximum load
    :param list axins_min: [x0, y0, width, height]; lower-left corner of inset axes, and its width and height for minimum load
    :param list axins_max_cords: [x1, x2, y1, y2]; x and y limits for the maximum load inset
    :param list axins_min_cords: [x1, x2, y1, y2]; x and y limits for the minimum load inset
    :param float axins_x_loc: major locator for the x-axis of the inset
    :param float axins_y_loc: major locator for the y-axis of the inset
    :param tuple xlim: x-axis limits
    :param tuple ylim: y-axis limits
    :param tuple figsize: width, height in inches
    :param str title: title of the plot
    :param pathlib.Path path_output: output path, including the filename
    :return: (fig, ax)
    """
    # Timezone
    tz = next(iter(dict_data.values()))[0].index.tzinfo  # use information from first df

    # Initialize
    fig, ax, axins_max, axins_min = None, None, axins_max, axins_min
    args = {
        "col": col,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "flag_axins_max": flag_axins_max,
        "flag_axins_min": flag_axins_min,
        "axins_max_cords": axins_max_cords,
        "axins_min_cords": axins_min_cords,
        "axins_x_loc": axins_x_loc,
        "axins_y_loc": axins_y_loc,
        "figsize": figsize,
    }

    # Add the curve for each dataframe
    for label, (df, color) in dict_data.items():
        assert df.index.tzinfo == tz, f"Timezone info of {label} is {df.index.tzinfo}, but should be {tz}"
        fig, ax, axins_max, axins_min = duration_curve(df=df,
                                                       fig=fig,
                                                       ax=ax,
                                                       color=color,
                                                       label=label,
                                                       axins_max=axins_max,
                                                       axins_min=axins_min,
                                                       **args
                                                       )

    # Format
    ax.grid(True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Title
    fig.suptitle(title)

    # Save the plot
    if path_output is not None:
        fig.savefig(path_output)

    return fig, ax
