import itertools
import re
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import scipy.stats as stats
from dateutil.rrule import DAILY
from dateutil.rrule import HOURLY
from dateutil.rrule import MINUTELY
from dateutil.rrule import MONTHLY
from dateutil.rrule import rrule
from dateutil.rrule import SECONDLY
from dateutil.rrule import WEEKLY
from dateutil.rrule import YEARLY
from pandas.api.types import is_datetime64_any_dtype
from pandas.api.types import is_numeric_dtype

from .config import TIME_UNITS


def format_bytes(bytes):
    if bytes // 1e9 > 1:
        return f"{bytes / 1e9:.1f}. GB"
    elif bytes // 1e6 > 1:
        return f"{bytes / 1e6:.1f} MB"
    elif bytes // 1e3 > 1:
        return f"{bytes / 1e3:.1f} KB"
    else:
        return f"{bytes} Bytes"


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
    add_other: bool = True,
) -> List[str]:
    """
    Orders the levels of a discrete data column and condenses excess levels into Other category.

    Args:
        data: pandas DataFrame with data columns
        column1: A string matching a column whose levels we want to order
        column2: A string matching a second optional column whose values we can use to order column1 instead of using
         counts
        order: Order in which to sort the levels.
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
            raise ValueError(f"Unknown level order specification: {order}")

    # restrict to max_levels levels (condense rest into Other)
    num_levels = data[column1].nunique()
    if num_levels > max_levels:
        if add_other:
            other_levels = order[max_levels - 1 :]
            order = order[: max_levels - 1] + ["Other"]
            if data[column1].dtype.name == "category":
                data[column1].cat.add_categories(["Other"], inplace=True)
            data[column1][data[column1].isin(other_levels)] = "Other"
        else:
            order = order[:max_levels]

    if include_missing:
        if data[column1].dtype.name == "category":
            data[column1].cat.add_categories(["NA"], inplace=True)
        data[column1] = data[column1].fillna("NA")
        order.append("NA")

    # convert to ordered categorical variable
    data[column1] = pd.Categorical(data[column1], categories=order, ordered=True)

    return data[column1]


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


def trim_values(
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
        lower_quantile: Lower quantile to filter data above
        upper_quantile: Upper quantile to filter data below

    Returns:
        pandas Dataframe filtered to remove rows where column values are beyond the specified quantiles
    """
    lower_quantile = data[column].quantile(lower_quantile)
    upper_quantile = data[column].quantile(upper_quantile)
    data = data[(data[column] >= lower_quantile) & (data[column] <= upper_quantile)]
    return data


def coerce_column_type(col_data, col_type):

    if not is_datetime64_any_dtype(col_data) and col_type == "datetime":
        return pd.to_datetime(col_data)
    elif col_data.dtype.name == "category" and col_type == "text":
        return col_data.astype("string")
    else:
        return col_data


def iqr(a):
    """
    Calculate the IQR for an array of numbers.
    https://github.com/has2k1/plotnine/blob/bcb93d6cc4ff266565c32a095e40b0127d3d3b7c/plotnine/stats/binning.py
    """
    a = np.asarray(a)
    q1 = stats.scoreatpercentile(a, 25)
    q3 = stats.scoreatpercentile(a, 75)
    return q3 - q1


def compute_time_deltas(
    col_data: pd.Series, delta_units: str, unit_th: float = 0.5
) -> Tuple[pd.Series, str]:
    """
    Compute time deltas between successive observations for a datetime series.

    Args:
        col_data: Datetime series to compute time differences for
        delta_units: Units to compute time deltas in. 'auto' attempts to guess the best units.
        unit_th: Threshold for median of units when guessing best units

    Returns:
        Computed time deltas series and units
    """

    dts = col_data.sort_values(ascending=True)
    deltas = dts - dts.shift(1)

    def td_str(units):
        if units == "months":
            return "30 days"
        elif units == "weeks":
            return "7 days"
        elif units == "years":
            return "365 days"
        else:
            return f"1 {units}"

    if delta_units == "auto":
        for unit in TIME_UNITS[::-1]:
            delta_units = unit
            unit_str = td_str(delta_units)
            med = (deltas / pd.Timedelta(unit_str)).median()
            if med >= unit_th:
                break
    else:
        unit_str = td_str(delta_units)
    deltas = deltas / pd.Timedelta(unit_str)
    return deltas, delta_units


def agg_time_series(data, column, agg_freq):
    """
    Aggregate a time series at the provided aggregation frequency.

    Args:
        data: pandas Dataframe to agg
        column: datetime column to aggregate
        agg_freq: Frequency at which to aggregate. Either a '{quantity} {unit} string such as '1 month' or a
         pandas frequency string.

    Returns:
        Tuple containing aggregated dataframe and ylabel with counts
    """

    # Automatically set an intelligent aggregation frequency
    if agg_freq == "auto":
        range_secs = (data[column].max() - data[column].min()).total_seconds()
        freqs = [
            "1 second",
            "1 month",
            "1 hour",
            "1 day",
            "1 week",
            "1 month",
            "1 year",
        ]
        vals = [
            1,
            60,
            3600,
            3600 * 24,
            3600 * 24 * 7,
            3600 * 24 * 7 * 30,
            3600 * 24 * 7 * 30 * 365,
        ]
        ix = np.argmin(np.abs([range_secs / 10 - v for v in vals]))
        agg_freq = freqs[ix]
    ylabel = f"Count (aggregated every {agg_freq})"
    agg_freq = convert_to_freq_string(agg_freq)

    # Resample and aggregate time series counts
    agg_df = (
        data.set_index(column)
        .resample(agg_freq)
        .agg("size")
        .reset_index()
        .rename({0: "Count"}, axis="columns")
    )
    return agg_df, ylabel


def detect_column_type(col_data, discrete_limit=50):
    col_data = col_data.dropna()

    if is_datetime64_any_dtype(col_data):
        return "datetime"
    elif is_numeric_dtype(col_data):
        if len(col_data.unique()) <= discrete_limit:
            return "categorical"
        else:
            return "numeric"
    elif col_data.dtype.name == "category":
        return "categorical"
    elif col_data.dtype.name == "string":
        test_value = col_data.dropna().iat[0]
        if re.search(
            "(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}",
            test_value,
        ):
            return "url"
        return "text"
    elif col_data.dtype.name == "object":
        test_value = col_data.dropna().iat[0]
        if isinstance(test_value, (list, tuple, set)):
            return "collection"
        # TODO: Probably need smarter detection
        elif type(test_value) == str:
            if re.search(
                "(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}",
                test_value,
            ):
                return "url"
            num_levels = col_data.nunique()
            if num_levels > len(col_data) / 2:
                if col_data.apply(lambda x: len(x.split(" "))).max() <= 3:
                    return "categorical"
                else:
                    return "text"
            else:
                return "categorical"
        else:
            return "categorical"
    else:
        raise ValueError(f"Unsupported data type {col_data.dtype.name}")


# Old


def match_axes(fig, ax, gg):
    upper = max(ax.get_xlim()[1], ax.get_ylim()[1])
    lower = min(ax.get_xlim()[0], ax.get_ylim()[0])
    gg += p9.coord_fixed(ratio=1, xlim=(lower, upper), ylim=(lower, upper))
    _ = gg._draw_using_figure(fig, [ax])
    return fig, ax, gg
