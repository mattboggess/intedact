import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import scipy.stats as stats
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from typing import List
import plotnine as p9
import re


def match_axes(fig, ax, gg):
    upper = max(ax.get_xlim()[1], ax.get_ylim()[1])
    lower = min(ax.get_xlim()[0], ax.get_ylim()[0])
    gg += p9.coord_fixed(ratio=1, xlim=(lower, upper), ylim=(lower, upper))
    _ = gg._draw_using_figure(fig, [ax])
    return fig, ax, gg


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


def add_count_annotations(gg, data, num_levels, flip_axis):
    """"""
    # Determine annotation placements based on axis flip
    # TODO: probably need to handle location placement better
    value_counts = data["Count"]
    nudge = value_counts.max() / 100
    mid = value_counts.max() / 4 * 3
    if flip_axis:
        va = ["center"] * len(value_counts)
        ha = ["right" if x > mid else "left" for x in value_counts]
        nudge_y = [-nudge if x > mid else nudge for x in value_counts]
    else:
        va = ["top" if x > mid else "bottom" for x in value_counts]
        ha = ["center"] * len(value_counts)
        nudge_y = [-nudge if x > mid else nudge for x in value_counts]

    # Determine annotation size
    # TODO: probably need better automatic sizing determination
    if num_levels > 10:
        size = 8
    else:
        size = 11

    # add count/percentage annotations
    data["label"] = [
        f"{x} ({100 * x / data['Count'].sum():.1f}%)" for x in data["Count"]
    ]
    gg += p9.geom_text(
        p9.aes(label="label", group=1, ha=ha, va=va),
        nudge_y=nudge_y,
        color="black",
        size=size,
    )
    return gg


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


def transform_axis(
    gg: p9.ggplot, column: str, transform: str = "identity", xaxis: bool = False
) -> (p9.ggplot, str):
    """
    Modifies a plotnine axis scale according to a specified data transformation.

    Args:
        gg: The plotnine ggplot object to modify the axis scale for
        column: The dataframe column that is being transformed
        transform: Transformation to apply to the column:
         - 'identity': no transformation
         - 'log': apply a logarithmic transformation with small constant added in case of zero values
         - 'log_exclude0': apply a logarithmic transformation with zero values removed
         - 'sqrt': apply a square root transformation
        xaxis: Whether to transform the x or the y axis
    Returns:
        Tuple containing the modified ggplot object with updated scale and modified axis label denoting
        the scale change.
    """
    if transform in ["log", "log_exclude0"]:
        if xaxis:
            gg += p9.scale_x_log10()
        else:
            gg += p9.scale_y_log10()
        label = f"{column} (log10 scale)"
    elif transform == "sqrt":
        if xaxis:
            gg += p9.scale_x_sqrt()
        else:
            gg += p9.scale_y_sqrt()
        label = f"{column} (square root scale)"
    else:
        label = column

    return gg, label


def iqr(a):
    """
    Calculate the IQR for an array of numbers.
    https://github.com/has2k1/plotnine/blob/bcb93d6cc4ff266565c32a095e40b0127d3d3b7c/plotnine/stats/binning.py
    """
    a = np.asarray(a)
    q1 = stats.scoreatpercentile(a, 25)
    q3 = stats.scoreatpercentile(a, 75)
    return q3 - q1


def freedman_diaconis_bins(a, transform="identity"):
    """
    Calculate number of hist bins using Freedman-Diaconis rule.
    https://github.com/has2k1/plotnine/blob/bcb93d6cc4ff266565c32a095e40b0127d3d3b7c/plotnine/stats/binning.py
    """
    # From http://stats.stackexchange.com/questions/798/
    a = np.asarray(a)
    if "log" in transform:
        a = np.log10(a)
    elif transform == "sqrt":
        a = np.sqrt(a)

    h = 2 * iqr(a) / (len(a) ** (1 / 3))

    # fall back to sqrt(a) bins if iqr is 0
    if h == 0:
        bins = np.ceil(np.sqrt(a.size))
    else:
        bins = np.ceil((np.nanmax(a) - np.nanmin(a)) / h)

    return min(np.int(bins), 100)


def add_percent_axis(ax: plt.Axes, data_size, flip_axis: bool = False) -> plt.Axes:
    """
    Adds a twin axis with percentages to a count plot.

    Args:
        ax: Plot axes figure to add percentage axis to
        data_size: Total count of ex
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


def order_levels(
    data: pd.DataFrame,
    column1: str,
    column2: str = None,
    level_order: str = "auto",
    max_levels: int = 20,
) -> List[str]:
    """
    Orders the levels of a discrete data column and condenses excess levels into Other category.

    Args:
        data: pandas DataFrame with data columns
        column1: A string matching a column whose levels we want to order
        column2: A string matching a second optional column whose values we can use to order column1
        level_order: Order in which to sort the levels.
         - 'auto' sorts ordinal variables by provided ordering, nominal variables by
            descending frequency, and numeric variables in sorted order.
         - 'descending' sorts in descending frequency.
         - 'ascending' sorts in ascending frequency.
         - 'sorted' sorts according to sorted order of the levels themselves.
         - 'random' produces a random order. Useful if there are too many levels for one plot.
        max_levels: Maximum number of levels to attempt to plot on a single plot. If exceeded, only the
         max_level - 1 levels will be plotted and the remainder will be grouped into an 'Other' category.

    Returns:
        Pandas series of column1 that has been converted into a Categorical type with the new level ordering
    """

    # determine order to plot levels
    if column2:
        value_counts = data.groupby(column1)[column2].median()
    else:
        value_counts = data[column1].value_counts()

    if level_order == "auto":
        if data[column1].dtype.name == "category" and data[column1].cat.ordered:
            order = list(data[column1].cat.categories)
        elif is_numeric_dtype(data[column1]):
            order = sorted(list(value_counts.index))
        else:
            order = list(value_counts.sort_values(ascending=False).index)
    elif level_order == "ascending":
        order = list(value_counts.sort_values(ascending=True).index)
    elif level_order == "descending":
        order = list(value_counts.sort_values(ascending=False).index)
    elif level_order == "sorted":
        order = sorted(list(value_counts.index))
    elif level_order == "random":
        order = list(value_counts.sample(frac=1).index)
    else:
        raise ValueError(f"Unknown level order specification: {level_order}")

    # restrict to max_levels levels (condense rest into Other)
    num_levels = len(data[column1].unique())
    if num_levels > max_levels:
        other_levels = order[max_levels - 1 :]
        order = order[: max_levels - 1] + ["Other"]
        if data[column1].dtype.name == "category":
            data[column1].cat.add_categories(["Other"], inplace=True)
        data[column1][data[column1].isin(other_levels)] = "Other"

    # convert to ordered categorical variable
    data[column1] = pd.Categorical(data[column1], categories=order, ordered=True)

    return data[column1]


def fmt_counts(counts, percentages):
    """
    https://plotnine.readthedocs.io/en/stable/tutorials/miscellaneous-show-counts-and-percentages-for-bar-plots.html
    """
    fmt = "{} ({:.1f}%)".format
    return [fmt(c, p) for c, p in zip(counts, percentages)]
