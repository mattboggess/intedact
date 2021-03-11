import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, List, Optional
import seaborn as sns
from .utils import (
    freedman_diaconis_bins,
    trim_quantiles,
    order_levels,
    add_percent_axis,
    transform_axis,
    preprocess_transform,
    convert_to_freq_string,
    add_barplot_annotations,
)
from .config import FLIP_LEVEL_COUNT
from .bivariate_plots import time_series_plot


def boxplot(
    data: pd.DataFrame,
    column: str,
    ax: Optional[plt.Axes] = None,
    lower_quantile: int = 0,
    upper_quantile: int = 1,
    transform: str = "identity",
    clip: float = 0,
    flip_axis: bool = True,
    **kwargs,
) -> plt.Axes:
    """
    Plots a boxplot of a numerical data column in a pandas DataFrame.

    Wraps seaborn's boxplot adding additional data handling arguments such as log transformations and trimming
    quantiles to ignore outliers.

    Args:
        data: pandas DataFrame containing data to be plotted
        column: name of column to plot histogram of
        ax: matplotlib axes generated from blank ggplot to plot onto. If specified, must also specify fig
        lower_quantile: Lower quantile of data to remove before plotting for ignoring outliers
        upper_quantile: Upper quantile of data to remove before plotting for ignoring outliers
        transform: Transformation to apply to the data for plotting:

         - **'identity'**: no transformation
         - **'log'**: apply a logarithmic transformation to the data
        clip: Value to clip zero values to for log transformation. If 0 (default), zero values are simply removed.
        flip_axis: Whether to flip the plot so boxplot is horizontal.
        kwargs: Additional keyword arguments passed through to [sns.boxplot](https://seaborn.pydata.org/generated/seaborn.boxplot.html)

    Returns:
        The axes plot was drawn to

    Examples:
        .. plot::

            import seaborn as sns
            import intedact
            data = sns.load_dataset('tips')
            intedact.boxplot(data, 'total_bill')
    """
    data = data.copy()

    # Remove upper and lower quantiles
    data = trim_quantiles(
        data, column, lower_quantile=lower_quantile, upper_quantile=upper_quantile
    )

    # Clip/remove zeros for log transformation
    data = preprocess_transform(data, column, transform, clip=clip)

    # Plot boxplot
    if flip_axis:
        ax = sns.boxplot(x=column, data=data, ax=ax, **kwargs)
    else:
        ax = sns.boxplot(y=column, data=data, ax=ax, **kwargs)

    # Log transform axis
    ax = transform_axis(ax, column, transform=transform, xaxis=flip_axis)

    return ax


def countplot(
    data: pd.DataFrame,
    column: str,
    ax: Optional[plt.Axes] = None,
    order: Union[str, List] = "auto",
    max_levels: int = 30,
    flip_axis: Optional[bool] = None,
    label_rotation: Optional[int] = None,
    percent_axis: bool = True,
    label_counts: bool = True,
    label_fontsize: Optional[float] = None,
    include_missing: bool = False,
    **kwargs,
) -> plt.Axes:
    """
    Plots a bar plot of counts/percentages across the levels of a discrete data column in a pandas DataFrame.

    Wraps seaborn's countplot adding annotations, twin axis for percents, and a few other nice argument controls
    useful for EDA.

    Args:
        data: pandas DataFrame with data to be plotted
        column: column in the dataframe to plot
        ax: matplotlib axes to plot to. Defaults to current axis.
        order: Order in which to sort the levels of the variable for plotting:

         - **'auto'**: sorts ordinal variables by provided ordering, nominal variables by descending frequency, and numeric variables in sorted order.
         - **'descending'**: sorts in descending frequency.
         - **'ascending'**: sorts in ascending frequency.
         - **'sorted'**: sorts according to sorted order of the levels themselves.
         - **'random'**: produces a random order. Useful if there are too many levels for one plot.
         Or you can pass a list of level names in directly for your own custom order.
        max_levels: Maximum number of levels to attempt to plot on a single plot. If exceeded, only the
         max_level - 1 levels will be plotted and the remainder will be grouped into an 'Other' category.
        percent_axis: Whether to add a twin y axis with percentages
        label_counts: Whether to add exact counts and percentages as text annotations on each bar in the plot.
        label_fontsize: Size of the annotations text. Default tries to infer a reasonable size based on the figure
         size and number of levels.
        flip_axis: Whether to flip the plot so labels are on y axis. Useful for long level names or lots of levels.
         Default tries to infer based on number of levels and label_rotation value.
        label_rotation: Amount to rotate level labels. Useful for long level names or lots of levels.
        include_missing: Whether to include missing values as an additional level in the data to be plotted
        kwargs: Additional keyword arguments passed through to [sns.barplot](https://seaborn.pydata.org/generated/seaborn.barplot.html)

    Returns:
        The axes plot was drawn to

    Examples:
        .. plot::

            import seaborn as sns
            import intedact
            data = sns.load_dataset('tips')
            intedact.countplot(data, 'day')
    """
    data = data.copy()

    # Handle axis flip default
    # TODO: Make more intelligent
    num_plot_levels = min(max_levels, data[column].nunique())
    if flip_axis is None:
        flip_axis = num_plot_levels >= FLIP_LEVEL_COUNT and label_rotation == 0

    # Reorder column levels
    data[column] = order_levels(
        data,
        column,
        None,
        order=order,
        max_levels=max_levels,
        include_missing=include_missing,
    )
    order = list(data[column].cat.categories)

    # Make the countplot
    x = "Count" if flip_axis else column
    y = column if flip_axis else "Count"
    count_data = (
        data.groupby(column).size().reset_index().rename({0: "Count"}, axis="columns")
    )
    ax = sns.barplot(x=x, y=y, data=count_data, ax=ax, order=order, **kwargs)

    # Add annotations
    if label_counts:
        ax = add_barplot_annotations(
            ax,
            count_data,
            "Count",
            add_percent=True,
            flip_axis=flip_axis,
            label_fontsize=label_fontsize,
        )

    # Add label rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=label_rotation)

    # Add a twin axis for percentage
    if percent_axis:
        add_percent_axis(ax, count_data["Count"].sum(), flip_axis=flip_axis)

    return ax


def continuous_summary_stats(
    data: pd.DataFrame,
    column: str,
    lower_quantile: float = 0,
    upper_quantile: float = 1,
) -> pd.DataFrame:
    """
    Computes summary statistics for a numerical pandas DataFrame column.

    Computed statistics include:

      - mean and median
      - min and max
      - 25% percentile
      - 75% percentile
      - standard deviation and interquartile range
      - count and percentage of missing values

    Args:
        data: The dataframe with the column to summarize
        column: The column in the dataframe to summarize
        lower_quantile: Lower quantile of data to remove before plotting for ignoring outliers
        upper_quantile: Upper quantile of data to remove before plotting for ignoring outliers

    Returns:
        pandas DataFrame with one row containing the summary statistics for the provided column
    """
    data = trim_quantiles(
        data, column, lower_quantile=lower_quantile, upper_quantile=upper_quantile
    )

    num_missing = data[column].isnull().sum()
    perc_missing = num_missing / data.shape[0]

    table = pd.DataFrame(data[column].describe()).T
    table["iqr"] = data[column].quantile(0.75) - data[column].quantile(0.25)
    table["missing_count"] = num_missing
    table["missing_percent"] = perc_missing
    table = pd.DataFrame(table)
    return table


def histogram(
    data: pd.DataFrame,
    column: str,
    ax: plt.Axes = None,
    bins: Optional[int] = None,
    transform: str = "identity",
    clip: float = 0,
    lower_quantile: int = 0,
    upper_quantile: int = 1,
    kde: bool = False,
    **kwargs,
) -> plt.Axes:
    """
    Plots a histogram of a data column in a pandas DataFrame.

    Args:
        data: pandas DataFrame containing data to be plotted
        column: name of column to plot histogram of
        ax: matplotlib axes to draw plot onto
        bins: Number of bins to use for the time delta histogram. Default is 0 which translates to
         automatically determined bins.
        transform: Transformation to apply to the data for plotting:

         - **'identity'**: no transformation
         - **'log'**: apply a logarithmic transformation to the data
        clip: Value to clip zero values to for log transformation. If 0 (default), zero values are simply removed.
        lower_quantile: Lower quantile of data to remove before plotting for ignoring outliers
        upper_quantile: Upper quantile of data to remove before plotting for ignoring outliers
        kde: Whether to overlay a KDE plot on the histogram
        kwargs: Additional keyword arguments passed through to [sns.histplot](https://seaborn.pydata.org/generated/seaborn.histplot.html)

    Returns:
        Matplotlib axes object with histogram drawn

    Example:
        .. plot::

            import seaborn as sns
            import intedact
            data = sns.load_dataset('tips')
            intedact.histogram(data, 'total_bill')
    """
    data = data.copy()

    # Remove upper and lower quantiles
    data = trim_quantiles(
        data, column, lower_quantile=lower_quantile, upper_quantile=upper_quantile
    )

    # Clip/remove zeros for log transformation
    data = preprocess_transform(data, column, transform, clip=clip)

    # Plot histogram
    if bins is None:
        bins = freedman_diaconis_bins(data[column], log=transform == "log")
    ax = sns.histplot(
        x=column,
        data=data,
        ax=ax,
        kde=kde,
        log_scale=transform == "log",
        bins=bins,
        **kwargs,
    )

    return ax


def time_series_countplot(
    data: pd.DataFrame,
    column: str,
    ax: Optional[plt.Axes] = None,
    ts_type: str = "point",
    ts_freq: str = "auto",
    trend_line: Optional[str] = None,
    date_labels: Optional[str] = None,
    date_breaks: Optional[str] = None,
    span: float = 0.75,
    ci_level: float = 0.95,
    **kwargs,
) -> plt.Axes:
    """
    Plots a times series plot of datetime column where the y axis is counts of observations aggregated at a provided
    temporal frequency. Assumes that each row in the dataframe is a single event and not already aggregated.

    Args:
        data: pandas DataFrame to perform EDA on
        column: A string matching a column in the data
        ax: matplotlib axes generated from blank ggplot to plot onto. If specified, must also specify fig
        ts_type: 'line' plots a line graph while 'point' plots points for observations
        ts_freq: String describing the frequency at which to aggregate data in one of two formats:

            - A `pandas offset string <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_.
            - A human readable string in the same format passed to date breaks (e.g. "4 months")
            Default is to attempt to intelligently determine a good aggregation frequency.
        trend_line: Trend line to plot over data. Default is to plot no trend line. Other options available are
            same as those available in plotnine's `stat_smooth <https://plotnine.readthedocs.io/en/stable/generated/plotnine.stats.stat_smooth.html#plotnine.stats.stat_smooth>`_
        date_labels: strftime date formatting string that will be used to set the format of the x axis tick labels
        date_breaks: Date breaks string in form '{interval} {period}'. Interval must be an integer and period must be
          a time period ranging from seconds to years. (e.g. '1 year', '3 minutes')
        span: Span parameter to determine amount of smoothing for loess
        ci_level: Confidence level determining how wide to plot confidence intervals for smoothing.

    Returns:
        Matplotlib axes with time series drawn

    Example:
        .. plot::

            import pandas as pd
            import intedact
            data = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/tidytuesday_tweets/data.csv")
            data['created_at'] = pd.to_datetime(data.created_at)
            intedact.time_series_countplot(data, 'created_at', ts_freq='1 week', trend_line='auto');
    """
    # TODO: Handle auto aggregating more intelligently
    if ts_freq == "auto":
        ts_freq = "1 month"
    ylabel = f"Count (aggregated every {ts_freq})"
    ts_freq = convert_to_freq_string(ts_freq)

    # Resample and aggregate time series counts
    tmp = (
        data.set_index(column)
        .resample(ts_freq)
        .agg("size")
        .reset_index()
        .rename({0: "Count"}, axis="columns")
    )

    # Draw the time series plot
    ax = time_series_plot(
        tmp,
        column,
        "Count",
        ax=ax,
        ts_type=ts_type,
        trend_line=trend_line,
        date_labels=date_labels,
        date_breaks=date_breaks,
        span=span,
        ci_level=ci_level,
        **kwargs,
    )

    # Add a twin axis for percentage
    add_percent_axis(ax, len(data[column]), flip_axis=False)

    ax.set_ylabel(ylabel)
    return ax
