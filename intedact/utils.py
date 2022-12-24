import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from plotnine.stats.smoothers import predictdf

TIME_UNITS = [
    "nanoseconds",
    "microseconds",
    "milliseconds",
    "seconds",
    "months",
    "hours",
    "days",
    "weeks",
    "months",
    "years",
]


def bin_data(
    data: pd.DataFrame,
    column: str,
    num_intervals: int = 4,
    interval_type: str = "quantile",
) -> List[str]:
    """
    Bin a numeric column into a discrete set of intervals.

    Args:
        data: Dataframe with data
        column: Numeric column in data to bin into intervals
        num_intervals: Number of intervals to bin the data into
        interval_type: Type of intervals to make. Either quantile or equal width intervals.

    Returns:
        List of intervals in increasing order
    """
    if interval_type == "quantile":
        data["interval"] = pd.qcut(
            data[column], num_intervals, duplicates="drop"
        ).astype(str)
    elif interval_type == "equal_width":
        data["interval"] = pd.cut(data[column], num_intervals).astype(str)
    else:
        raise ValueError(f"Unknown interval_type {interval_type}")
    intervals = data.interval.unique()
    if len(intervals) < num_intervals and interval_type == "quantile":
        print(
            f"Dropped {num_intervals- len(intervals)} intervals due to duplicate quantiles."
        )
    interval_order = sorted(
        data.interval.unique(), key=lambda x: float(x.split(",")[0][1:])
    )
    data["interval"] = pd.Categorical(
        data["interval"], categories=interval_order, ordered=True
    )
    return interval_order


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
        add_other: Whether to include 'Other' in the plot or not when condensing levels

    Returns:
        List of new levels in order. column1 is converted into a Categorical type with the new level ordering in place.
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
        if data[column1].isnull().sum() > 0:
            if data[column1].dtype.name == "category":
                data[column1].cat.add_categories(["NA"], inplace=True)
            data[column1] = data[column1].fillna("NA")
            order.append("NA")
        else:
            print(f"No missing values for column: {column1}")

    data[column1] = pd.Categorical(data[column1], categories=order, ordered=True)

    return order


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


def coerce_column_type(col_data: pd.Series, col_type: str) -> pd.Series:
    if not is_datetime64_any_dtype(col_data) and col_type == "datetime":
        return pd.to_datetime(col_data)
    elif col_data.dtype.name == "category" and col_type == "text":
        return col_data.astype("string")
    else:
        return col_data


def agg_time_series(
    data: pd.DataFrame, column: str, agg_freq: str
) -> Tuple[pd.DataFrame, str]:
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


def detect_column_type(col_data: pd.Series, discrete_limit: int = 50) -> str:
    """
    Tries to infer the intedact variable type for the provided data.

    Args:
        col_data: Data to infer type for
        discrete_limit: Number of distinct values below which a column with a numerical data type will be treated as categorical

    Returns:
        Type of intedact variable that will be used to generate summaries
    """
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


def compute_trendline(
    data: pd.DataFrame,
    x: str,
    y: str,
    method: str,
    span: float = 0.75,
    level: float = 0.95,
) -> pd.DataFrame:
    """
    Computes data to use to plot a trend line. This is a modified version of plotnine's stat_smooth since plotly
    doesn't have support for custom subplot trend lines.

    Args:
        data: pandas Dataframe to add trend line for
        x: x axis variable column name
        y: y axis variable column name
        method: smoothing method, see plotnine's [stat_smooth](https://plotnine.readthedocs.io/en/stable/generated/plotnine.stats.stat_smooth.html#plotnine.stats.stat_smooth) for options
        span: span parameter for loess
        level: confidence level to use for drawing confidence interval

    Returns:
        Dataframe where x and y columns hold trend line data which can be plotted
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
    date_min = data[x].min()
    if data[x].dtype.kind == "M":
        data["x"] = (data[x] - date_min).dt.total_seconds()
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
    if data[x].dtype.kind == "M":
        df["x"] = date_min + pd.to_timedelta(data["x"], unit="S")
    return df
