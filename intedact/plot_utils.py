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


# Data Helpers


def preprocess_transform(
    data: pd.DataFrame,
    column: str,
    transform: str = "identity",
    clip: float = 0,
) -> pd.DataFrame:
    """
    Preprocesses a pandas dataframe column for applying a data transformation

    Args:
        data: Data to be transformed
        column: The dataframe column that is being transformed
        transform: Transformation to apply to the data for plotting:

         - **'identity'**: no transformation
         - **'log'**: apply a logarithmic transformation to the data
        clip: Value to clip zero values to for log transformation. If 0 (default), zero values are simply removed.
    Returns:
        pandas DataFrame with preprocessed column data
    """
    if transform == "log":
        data.loc[data[column] < clip, column] = clip
        data = data[data[column] > 0]
    return data


def order_levels(
    data: pd.DataFrame,
    column1: str,
    column2: Optional[str] = None,
    order: str = "auto",
    max_levels: int = 30,
    include_missing: bool = False,
) -> List[str]:
    """
    Orders the levels of a discrete data column and condenses excess levels into Other category.

    Args:
        data: pandas DataFrame with data columns
        column1: A string matching a column whose levels we want to order
        column2: A string matching a second optional column whose values we can use to order column1 instead of using
         counts
        order_method: Order in which to sort the levels.
         - 'auto' sorts ordinal variables by provided ordering, nominal variables by
            descending frequency, and numeric variables in sorted order.
         - 'descending' sorts in descending frequency.
         - 'ascending' sorts in ascending frequency.
         - 'sorted' sorts according to sorted order of the levels themselves.
         - 'random' produces a random order. Useful if there are too many levels for one plot.
         Or can just pass in a list of levels directly.
        max_levels: Maximum number of levels to attempt to plot on a single plot. If exceeded, only the
         max_level - 1 levels will be plotted and the remainder will be grouped into an 'Other' category.
        include_missing: Whether to include missing values as an additional level in the data to be plotted

    Returns:
        Pandas series of column1 that has been converted into a Categorical type with the new level ordering
    """

    if type(order) == str:
        # determine order to plot levels
        if column2:
            value_counts = data.groupby(column1)[column2].median()
        else:
            value_counts = data[column1].value_counts()

        if order == "auto":
            if data[column1].dtype.name == "category" and data[column1].cat.ordered:
                order = list(data[column1].cat.categories)
            elif is_numeric_dtype(data[column1]):
                order = sorted(list(value_counts.index))
            else:
                order = list(value_counts.sort_values(ascending=False).index)
        elif order == "ascending":
            order = list(value_counts.sort_values(ascending=True).index)
        elif order == "descending":
            order = list(value_counts.sort_values(ascending=False).index)
        elif order == "sorted":
            order = sorted(list(value_counts.index))
        elif order == "random":
            order = list(value_counts.sample(frac=1).index)
        else:
            raise ValueError(f"Unknown level order specification: {order_method}")

    # restrict to max_levels levels (condense rest into Other)
    num_levels = len(data[column1].unique())
    if num_levels > max_levels:
        other_levels = order[max_levels - 1 :]
        order = order[: max_levels - 1] + ["Other"]
        if data[column1].dtype.name == "category":
            data[column1].cat.add_categories(["Other"], inplace=True)
        data[column1][data[column1].isin(other_levels)] = "Other"

    if include_missing:
        if data[column1].dtype.name == "category":
            data[column1].cat.add_categories(["NA"], inplace=True)
        data[column1] = data[column1].fillna("NA")
        order.append("NA")

    # convert to ordered categorical variable
    data[column1] = pd.Categorical(data[column1], categories=order, ordered=True)

    return data[column1]


def preprocess_transformations(
    data: pd.DataFrame, column: str, transform: str = "identity"
) -> pd.DataFrame:
    """
    Preprocesses a data column to be compatible with the provided transformation.

    Args:
        data: pandas DataFrame holding the data
        column: The dataframe column that is being transformed
        transform: Transformation to apply to the column:
         - 'identity': no transformation
         - 'log': apply a logarithmic transformation with small constant added in case of zero values
         - 'log_exclude0': apply a logarithmic transformation with zero values removed
         - 'sqrt': apply a square root transformation
    Returns:
        Modified dataframe with the column data updated according to the transform specified
    """
    if transform == "log":
        data[column] += 1e-6
    elif transform == "log_exclude0":
        data = data[data[column] > 0]
    return data


def freedman_diaconis_bins(a, log=False):
    """
    Calculate number of hist bins using Freedman-Diaconis rule.
    https://github.com/has2k1/plotnine/blob/bcb93d6cc4ff266565c32a095e40b0127d3d3b7c/plotnine/stats/binning.py
    Ceiling at 100 for default efficiency purposes.
    """
    # From http://stats.stackexchange.com/questions/798/
    a = np.asarray(a)
    if log:
        a = np.log(a)

    h = 2 * iqr(a) / (len(a) ** (1 / 3))

    # fall back to sqrt(a) bins if iqr is 0
    if h == 0:
        bins = np.ceil(np.sqrt(a.size))
    else:
        bins = np.ceil((np.nanmax(a) - np.nanmin(a)) / h)

    return min(np.int(bins), 100)


def convert_to_freq_string(date_str: str) -> str:
    """
    Converts a conversational description of a period (e.g. 2 weeks) to a pandas frequency string (2W).

    Args:
        date_str: A period description of the form "{number} {period}"

    Returns:
        A corresponding pandas frequency string
    """
    # Column type groupings
    DATE_CONVERSION = {
        "year": "AS",
        "month": "MS",
        "week": "W",
        "day": "D",
        "hour": "H",
        "minute": "T",
        "second": "S",
    }
    split = date_str.split()
    if len(split) != 2:
        return date_str
    quantity, period = split
    period = period.lower()
    if period.endswith("s"):
        period = period[:-1]
    period = DATE_CONVERSION[period]
    return f"{quantity}{period}"


def convert_date_breaks(breaks_str: str) -> mdates.DateLocator:
    """
    Converts a conversational description of a period (e.g. 2 weeks) to a matplotlib date tick Locator.

    Args:
        breaks_str: A period description of the form "{interval} {period}"

    Returns:
        A corresponding mdates.Locator
    """
    # Column type groupings
    DATE_CONVERSION = {
        "year": YEARLY,
        "month": MONTHLY,
        "week": WEEKLY,
        "day": DAILY,
        "hour": HOURLY,
        "minute": MINUTELY,
        "second": SECONDLY,
    }
    interval, period = breaks_str.split()
    period = period.lower()
    if period.endswith("s"):
        period = period[:-1]
    period = DATE_CONVERSION[period]

    return mdates.RRuleLocator(mdates.rrulewrapper(period, interval=int(interval)))


# Old


def match_axes(fig, ax, gg):
    upper = max(ax.get_xlim()[1], ax.get_ylim()[1])
    lower = min(ax.get_xlim()[0], ax.get_ylim()[0])
    gg += p9.coord_fixed(ratio=1, xlim=(lower, upper), ylim=(lower, upper))
    _ = gg._draw_using_figure(fig, [ax])
    return fig, ax, gg


def iqr(a):
    """
    Calculate the IQR for an array of numbers.
    https://github.com/has2k1/plotnine/blob/bcb93d6cc4ff266565c32a095e40b0127d3d3b7c/plotnine/stats/binning.py
    """
    a = np.asarray(a)
    q1 = stats.scoreatpercentile(a, 25)
    q3 = stats.scoreatpercentile(a, 75)
    return q3 - q1


def _rotate_labels(gg, rotate=True):
    if rotate:
        gg += theme(axis_text_x=element_text(rotation=90))
    else:
        gg += theme(axis_text_x=element_text(rotation=0))
    return gg


def trim_quantiles(
    data: pd.DataFrame,
    column: str,
    lower_quantile: float = 0,
    upper_quantile: float = 1,
) -> pd.DataFrame:
    """
    Filters a dataframe by removing rows where the values for a specified column are above or below
    provided quantile limits.

    Args:
        data: pandas DataFrame to perform EDA on
        column: A string matching a column in the data to visualize
        lower_quantile: Lower quantile of column data to remove below
        upper_quantile: Upper quantile of column data to remove above

    Returns:
        pandas Dataframe filtered to remove rows where column values are beyond the specified quantiles
    """
    lower_quantile = data[column].quantile(lower_quantile)
    upper_quantile = data[column].quantile(upper_quantile)
    query = (data[column] >= lower_quantile) & (data[column] <= upper_quantile)
    return data[query]


def detect_column_type(col_data, discrete_limit=50):

    if is_datetime64_any_dtype(col_data):
        return "datetime"
    elif is_numeric_dtype(col_data):
        if len(col_data.unique()) <= discrete_limit:
            return "discrete"
        else:
            return "continuous"
    elif col_data.dtype.name == "category":
        return "discrete"
    elif col_data.dtype.name == "string":
        return "text"
    elif col_data.dtype.name == "object":
        test_value = col_data.dropna().iat[0]
        if isinstance(test_value, (list, tuple, set)):
            return "list"
        # TODO: Probably need smarter detection
        elif type(test_value) == str:
            num_levels = col_data.nunique()
            if num_levels > len(col_data) / 2:
                if col_data.apply(lambda x: len(x.split(" "))).max() <= 3:
                    return "discrete"
                else:
                    return "text"
            else:
                return "discrete"
        else:
            return "discrete"
    else:
        raise ValueError(f"Unsupported data type {col_data.dtype.name}")


def coerce_column_type(col_data, col_type):

    if not is_datetime64_any_dtype(col_data) and col_type == "datetime":
        return pd.to_datetime(col_data)
    elif col_data.dtype.name == "category" and col_type == "text":
        return col_data.astype("string")
    else:
        return col_data


def fmt_counts(counts, percentages):
    """
    https://plotnine.readthedocs.io/en/stable/tutorials/miscellaneous-show-counts-and-percentages-for-bar-plots.html
    """
    fmt = "{} ({:.1f}%)".format
    return [fmt(c, p) for c, p in zip(counts, percentages)]
