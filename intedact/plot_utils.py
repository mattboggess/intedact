import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import scipy.stats as stats
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from typing import List, Optional
import matplotlib.dates as mdates
from plotnine.stats.smoothers import predictdf
from dateutil.rrule import (
    rrule,
    YEARLY,
    MONTHLY,
    WEEKLY,
    DAILY,
    HOURLY,
    MINUTELY,
    SECONDLY,
)
import re


# Plotting Helpers


def add_barplot_annotations(
    ax: plt.Axes,
    data: pd.DataFrame,
    column: str,
    add_percent: bool,
    flip_axis: bool,
    label_fontsize: Optional[int] = None,
    height_threshold: float = 0.75,
    denominator: Optional[int] = None,
) -> plt.Axes:
    """
    Converts a conversational description of a period (e.g. 2 weeks) to a pandas frequency string (2W).

    Args:
        ax: matplotlib axes with barplot to annotate
        data: pandas Dataframe containing barplot data
        column: Column with the values to annotate
        add_percent: Whether to add percentages (assumes values are counts)
        flip_axis: Whether the axes have been flipped for barplot
        label_fontsize: Size of the annotations text. Default tries to infer a reasonable size based on the figure
         size and number of values.
        height_threshold: Threshold for how tall a bar must be before labels get placed within bar instead of
         on top of bar

    Returns:
        Axis with annotations added
    """
    if denominator is None:
        denominator = data[column].sum()

    # TODO: Handle default annotation size
    for i, value in enumerate(data[column]):

        label = f"{value}"
        if flip_axis:
            if add_percent:
                label += f" ({100 * value / denominator:.2f}%)"
            va = "center"
            if value > height_threshold * ax.get_xlim()[1]:
                ha = "right"
                color = "white"
                mod = -0.005
            else:
                ha = "left"
                color = "black"
                mod = 0.005
            y = i
            x = value + mod * ax.get_xlim()[1]
        else:
            if add_percent:
                label += f"\n{100 * value / denominator:.2f}%"
            ha = "center"
            if value > height_threshold * ax.get_ylim()[1]:
                va = "top"
                color = "white"
                mod = -0.01
            else:
                va = "bottom"
                color = "black"
                mod = 0.01
            x = i
            y = value + mod * ax.get_ylim()[1]

        ax.text(
            x,
            y,
            label,
            fontsize=label_fontsize,
            color=color,
            va=va,
            ha=ha,
        )
    return ax


def add_percent_axis(ax: plt.Axes, data_size, flip_axis: bool = False) -> plt.Axes:
    """
    Adds a twin axis with percentages to a count plot.

    Args:
        ax: Plot axes figure to add percentage axis to
        data_size: Total count to use to normalize percentages
        flip_axis: Whether the countplot had its axes flipped

    Returns:
        Twin axis that percentages were added to
    """
    if flip_axis:
        ax_perc = ax.twiny()
        ax_perc.set_xticks(100 * ax.get_xticks() / data_size)
        ax_perc.set_xlim(
            (
                100.0 * (float(ax.get_xlim()[0]) / data_size),
                100.0 * (float(ax.get_xlim()[1]) / data_size),
            )
        )
        ax_perc.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_perc.xaxis.set_tick_params(labelsize=10)
    else:
        ax_perc = ax.twinx()
        ax_perc.set_yticks(100 * ax.get_yticks() / data_size)
        ax_perc.set_ylim(
            (
                100.0 * (float(ax.get_ylim()[0]) / data_size),
                100.0 * (float(ax.get_ylim()[1]) / data_size),
            )
        )
        ax_perc.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax_perc.yaxis.set_tick_params(labelsize=10)
    ax_perc.grid(False)
    return ax_perc


def set_fontsize(ax, fontsize):
    for item in [ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(float(fontsize) + 1)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fontsize)


def transform_axis(
    ax: plt.Axes,
    column: str,
    transform: str = "identity",
    xaxis: bool = False,
) -> plt.Axes:
    """
    Modifies an axis object to use a log transformation

    TODO: Better format for axis tick labels

    Args:
        ax: The axes object to modify scale for
        column: The dataframe column that is being transformed
        transform: Transformation to apply to the data for plotting:

         - **'identity'**: no transformation
         - **'log'**: apply a logarithmic transformation to the data
        clip: Value to clip zero values to for log transformation. If 0 (default), zero values are simply removed.
        xaxis: Whether to transform the x or the y axis
    Returns:
        Tuple containing the modified ggplot object with updated scale and modified axis label denoting
        the scale change.
    """
    if transform == "log":
        label = f"{column} (log10 scale)"
        if xaxis:
            ax.set_xlabel(label)
            ax.set_xscale("log", base=10)
        else:
            ax.set_ylabel(label)
            ax.set_yscale("log", base=10)
    return ax


def add_trendline(
    data: pd.DataFrame,
    x: str,
    y: str,
    ax: plt.Axes,
    method: str,
    span: float = 0.75,
    level: float = 0.95,
) -> plt.Axes:
    """
    Adds a trendline to a line or scatter plot. This is a modified version of plotnine's
    stat_smooth since plotnine can't interface with existing matplotlib axes.

    plotnine is amazing and I wish I could have used it for this project!

    Args:
        data: pandas Dataframe to add trend line for
        x: x axis variable column name
        y: y axis variable column name
        ax: matplotlib axis to add trend line to
        method: smoothing method, see plotnine's [stat_smooth](https://plotnine.readthedocs.io/en/stable/generated/plotnine.stats.stat_smooth.html#plotnine.stats.stat_smooth) for options
        span: span parameter for loess
        level: confidence level to use for drawing confidence interval

    Returns:
        matplotlib axes object with trend line added
    """

    params = {
        "geom": "smooth",
        "position": "identity",
        "na_rm": False,
        "method": method,
        "se": True,
        "n": 80,
        "formula": None,
        "fullrange": False,
        "level": level,
        "span": span,
        "method_args": {},
    }

    if params["method"] == "auto":
        max_group = data[x].value_counts().max()
        if max_group < 1000:
            try:
                from skmisc.loess import loess  # noqa: F401

                params["method"] = "loess"
            except ImportError:
                params["method"] = "lowess"
        else:
            params["method"] = "glm"

    if params["method"] == "mavg":
        if "window" not in params["method_args"]:
            window = len(data) // 10
            params["method_args"]["window"] = window

    if params["formula"]:
        allowed = {"lm", "ols", "wls", "glm", "rlm", "gls"}
        if params["method"] not in allowed:
            raise ValueError(
                "You can only use a formula with `method` is "
                "one of {}".format(allowed)
            )

    # convert datetime to numeric values
    if data[x].dtype.kind == "M":
        data["x"] = (data[x] - data[x].min()).dt.total_seconds()
    else:
        data["x"] = data[x]
    data["y"] = data[y]

    data = data.sort_values("x")
    n = data.shape[0]
    x_unique = data["x"].unique()

    # Not enough data to fit
    if len(x_unique) < 2:
        print("Need 2 or more points to smooth")
        return pd.DataFrame()

    if data["x"].dtype.kind == "i":
        xseq = np.sort(x_unique)
    else:
        rangee = [data["x"].min(), data["x"].max()]
        xseq = np.linspace(rangee[0], rangee[1], n)

    df = predictdf(data, xseq, **params)

    ax.plot(data[x], df["y"], color="black", linewidth=2)
    ax.fill_between(
        data[x],
        df["ymin"],
        df["ymax"],
        alpha=0.2,
        color="black",
        edgecolor=None,
    )
    return ax
