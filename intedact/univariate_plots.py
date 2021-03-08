import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, List
import seaborn as sns
import plotnine as p9
from typing import Tuple
from .utils import (
    freedman_diaconis_bins,
    trim_quantiles,
    transform_axis,
    order_levels,
    add_percent_axis,
    convert_to_freq_string,
)
from .utils import add_count_annotations, preprocess_transformations, transform_axis
from .config import FLIP_LEVEL_COUNT, BAR_COLOR, THEME_DEFAULT
from .bivariate_plots import time_series_plot
import matplotlib.dates as mdates


def boxplot(
    data: pd.DataFrame,
    column: str,
    transform: str = "identity",
    lower_quantile: int = 0,
    upper_quantile: int = 1,
    flip_axis: bool = True,
    bar_color: str = None,
    theme: str = None,
    fig_width: int = None,
    fig_height: int = None,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
) -> plt.Figure:
    """
    Plots a boxplot of a data column in a pandas DataFrame.

    Args:
        data: pandas DataFrame containing data to be plotted
        column: name of column to plot histogram of
        fig: matplotlib Figure generated from blank ggplot to plot onto. If specified, must also specify ax
        ax: matplotlib axes generated from blank ggplot to plot onto. If specified, must also specify fig
        fig_width: figure width in inches
        fig_height: figure height in inches
        transform: Transformation to apply to the data for plotting:

         - **'identity'**: no transformation
         - **'log'**: apply a logarithmic transformation with small constant added in case of zero values
         - **'log_exclude0'**: apply a logarithmic transformation with zero values removed
         - **'sqrt'**: apply a square root transformation
        lower_quantile: Lower quantile of data to remove before plotting for ignoring outliers
        upper_quantile: Upper quantile of data to remove before plotting for ignoring outliers
        flip_axis: Whether to flip the plot so boxplot is horizontal.
        bar_color: Color to use for histogram bars
        theme: plotnine theme to use for the plot, str must match available theme listed `here <https://plotnine.readthedocs.io/en/stable/api.html#themes>`_

    Returns:
        Tuple containing matplotlib figure and axes along with the plotnine ggplot object

    Example:
        .. plot::

            import seaborn as sns
            import intedact
            data = sns.load_dataset('tips')
            intedact.boxplot(data, 'total_bill')
    """
    data = trim_quantiles(
        data, column, lower_quantile=lower_quantile, upper_quantile=upper_quantile
    )
    data = preprocess_transformations(data, column, transform=transform)

    # plot boxplot
    if bar_color is not None:
        args = {"fill": bar_color, "color": "black"}
    else:
        args = {"color": "black", "fill": BAR_COLOR}
    gg = p9.ggplot(data, p9.aes(x=[""], y=column)) + p9.geom_boxplot(**args)

    if flip_axis:
        gg += p9.coord_flip()

    # handle axes transforms
    gg, label = transform_axis(gg, column, transform, xaxis=False)

    if theme is not None:
        gg += eval(f"p9.{theme}()")
    else:
        gg += eval(f"p9.{THEME_DEFAULT}()")

    if fig and ax is not None:
        _ = gg._draw_using_figure(fig, [ax])
    else:
        fig = gg.draw()

    # set the figure size
    if fig_width is not None and fig_height is not None:
        fig.set_size_inches(fig_width, fig_height)

    return fig, ax, gg


def countplot_old(
    data: pd.DataFrame,
    column: str,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    fig_width: int = None,
    fig_height: int = None,
    level_order: str = "auto",
    max_levels: int = 30,
    label_counts: bool = True,
    flip_axis: bool = None,
    label_rotation: int = 0,
    bar_color: str = None,
    theme: str = None,
) -> Tuple[plt.Figure, plt.Axes, p9.ggplot]:
    """
    Plots a bar plot of counts/percentages across the levels of a discrete data column in a pandas DataFrame.

    Args:
        data: pandas DataFrame with data to be plotted
        column: column in the dataframe to plot
        fig: matplotlib Figure generated from blank ggplot to plot onto. If specified, must also specify ax
        ax: matplotlib axes generated from blank ggplot to plot onto. If specified, must also specify fig
        fig_width: figure width in inches
        fig_height: figure height in inches
        level_order: Order in which to sort the levels of the variable for plotting:

         - **'auto'**: sorts ordinal variables by provided ordering, nominal variables by descending frequency, and numeric variables in sorted order.
         - **'descending'**: sorts in descending frequency.
         - **'ascending'**: sorts in ascending frequency.
         - **'sorted'**: sorts according to sorted order of the levels themselves.
         - **'random'**: produces a random order. Useful if there are too many levels for one plot.
        max_levels: Maximum number of levels to attempt to plot on a single plot. If exceeded, only the
         max_level - 1 levels will be plotted and the remainder will be grouped into an 'Other' category.
        label_counts: Whether to add exact counts and percentages as text annotations on each bar in the plot.
        flip_axis: Whether to flip the plot so labels are on y axis. Useful for long level names or lots of levels.
        label_rotation: Amount to rotate level labels. Useful for long level names or lots of levels.
        bar_color: Color to use for bar fills
        theme: plotnine theme to use for the plot, str must match available theme listed `here <https://plotnine.readthedocs.io/en/stable/api.html#themes>`_

    Returns:
        Tuple containing matplotlib figure and axes along with the plotnine ggplot object

    Example:
        .. plot::

            import seaborn as sns
            import intedact
            data = sns.load_dataset('tips')
            intedact.countplot(data, 'day')

    """
    # TODO: Get more intelligent flip axis determination
    num_levels = data[column].nunique()
    if flip_axis is None:
        flip_axis = num_levels >= FLIP_LEVEL_COUNT and label_rotation == 0

    # reorder column levels
    data[column] = order_levels(
        data,
        column,
        None,
        level_order=level_order,
        max_levels=max_levels,
        flip_axis=flip_axis,
    )

    # make the barplot
    count_data = (
        data.groupby(column).size().reset_index().rename({0: "Count"}, axis="columns")
    )
    if bar_color is not None:
        args = {"fill": bar_color, "color": "black"}
    else:
        args = {"fill": BAR_COLOR, "color": "black"}
    gg = p9.ggplot(count_data, p9.aes(x=column, y="Count")) + p9.geom_col(**args)

    if theme is not None:
        gg += eval(f"p9.{theme}()")
    else:
        gg += eval(f"p9.{THEME_DEFAULT}()")

    # rotate labels
    gg += p9.theme(axis_text_x=p9.element_text(rotation=label_rotation, ha="center"))

    # flip axis
    if flip_axis:
        gg += p9.coord_flip()
        xlabel = "Count"
        ylabel = column
    else:
        xlabel = column
        ylabel = "Count"

    # add annotations
    if label_counts:
        gg = add_count_annotations(gg, count_data, num_levels, flip_axis)

    if fig is None and ax is None:
        gg.draw()
        fig = plt.gcf()
        ax = fig.axes[0]
    elif fig is not None and ax is not None:
        _ = gg._draw_using_figure(fig, [ax])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

    # add a twin axis for percentage
    add_percent_axis(ax, len(data[column]), flip_axis=flip_axis)

    # set the figure size
    if fig_width is not None and fig_height is not None:
        fig.set_size_inches(fig_width, fig_height)

    return fig, ax, gg


def countplot(
    data: pd.DataFrame,
    column: str,
    order: Union[str, List] = "auto",
    max_levels: int = 30,
    label_counts: bool = True,
    flip_axis: bool = None,
    label_rotation: int = 0,
    ax: plt.Axes = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes, p9.ggplot]:
    """
    Plots a bar plot of counts/percentages across the levels of a discrete data column in a pandas DataFrame.

    Args:
        data: pandas DataFrame with data to be plotted
        column: column in the dataframe to plot
        fig: matplotlib Figure generated from blank ggplot to plot onto. If specified, must also specify ax
        ax: matplotlib axes generated from blank ggplot to plot onto. If specified, must also specify fig
        fig_width: figure width in inches
        fig_height: figure height in inches
        level_order: Order in which to sort the levels of the variable for plotting:

         - **'auto'**: sorts ordinal variables by provided ordering, nominal variables by descending frequency, and numeric variables in sorted order.
         - **'descending'**: sorts in descending frequency.
         - **'ascending'**: sorts in ascending frequency.
         - **'sorted'**: sorts according to sorted order of the levels themselves.
         - **'random'**: produces a random order. Useful if there are too many levels for one plot.
        max_levels: Maximum number of levels to attempt to plot on a single plot. If exceeded, only the
         max_level - 1 levels will be plotted and the remainder will be grouped into an 'Other' category.
        label_counts: Whether to add exact counts and percentages as text annotations on each bar in the plot.
        flip_axis: Whether to flip the plot so labels are on y axis. Useful for long level names or lots of levels.
        label_rotation: Amount to rotate level labels. Useful for long level names or lots of levels.
        bar_color: Color to use for bar fills
        theme: plotnine theme to use for the plot, str must match available theme listed `here <https://plotnine.readthedocs.io/en/stable/api.html#themes>`_

    Returns:
        Tuple containing matplotlib figure and axes along with the plotnine ggplot object

    Example:
        .. plot::

            import seaborn as sns
            import intedact
            data = sns.load_dataset('tips')
            intedact.countplot(data, 'day')

    """
    # TODO: Get more intelligent flip axis determination
    num_levels = data[column].nunique()
    if flip_axis is None:
        flip_axis = num_levels >= FLIP_LEVEL_COUNT and label_rotation == 0

    # reorder column levels
    if type(order) == str:
        order = order_levels(
            data,
            column,
            None,
            level_order=order,
            max_levels=max_levels,
            flip_axis=flip_axis,
        )

    # make the barplot
    count_data = (
        data.groupby(column).size().reset_index().rename({0: "Count"}, axis="columns")
    )
    if bar_color is not None:
        args = {"fill": bar_color, "color": "black"}
    else:
        args = {"fill": BAR_COLOR, "color": "black"}
    gg = p9.ggplot(count_data, p9.aes(x=column, y="Count")) + p9.geom_col(**args)

    if theme is not None:
        gg += eval(f"p9.{theme}()")
    else:
        gg += eval(f"p9.{THEME_DEFAULT}()")

    # rotate labels
    gg += p9.theme(axis_text_x=p9.element_text(rotation=label_rotation, ha="center"))

    # flip axis
    if flip_axis:
        gg += p9.coord_flip()
        xlabel = "Count"
        ylabel = column
    else:
        xlabel = column
        ylabel = "Count"

    # add annotations
    if label_counts:
        gg = add_count_annotations(gg, count_data, num_levels, flip_axis)

    if fig is None and ax is None:
        gg.draw()
        fig = plt.gcf()
        ax = fig.axes[0]
    elif fig is not None and ax is not None:
        _ = gg._draw_using_figure(fig, [ax])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

    # add a twin axis for percentage
    add_percent_axis(ax, len(data[column]), flip_axis=flip_axis)

    # set the figure size
    if fig_width is not None and fig_height is not None:
        fig.set_size_inches(fig_width, fig_height)

    return fig, ax, gg


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
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    fig_width: int = None,
    fig_height: int = None,
    hist_bins: int = 0,
    transform: str = "identity",
    lower_quantile: int = 0,
    upper_quantile: int = 1,
    kde: bool = False,
    bar_color: str = None,
    theme: str = None,
) -> Tuple[plt.Figure, plt.Axes, p9.ggplot]:
    """
    Plots a histogram of a data column in a pandas DataFrame.

    Args:
        data: pandas DataFrame containing data to be plotted
        column: name of column to plot histogram of
        fig: matplotlib Figure generated from blank ggplot to plot onto. If specified, must also specify ax
        ax: matplotlib axes generated from blank ggplot to plot onto. If specified, must also specify fig
        fig_width: figure width in inches
        fig_height: figure height in inches
        hist_bins: Number of bins to use for the time delta histogram. Default is 0 which translates to
         automatically determined bins.
        transform: Transformation to apply to the data for plotting:

         - **'identity'**: no transformation
         - **'log'**: apply a logarithmic transformation with small constant added in case of zero values
         - **'log_exclude0'**: apply a logarithmic transformation with zero values removed
         - **'sqrt'**: apply a square root transformation
        lower_quantile: Lower quantile of data to remove before plotting for ignoring outliers
        upper_quantile: Upper quantile of data to remove before plotting for ignoring outliers
        kde: Whether to overlay a KDE plot on the histogram
        bar_color: Color to use for histogram bars
        theme: plotnine theme to use for the plot, str must match available theme listed `here <https://plotnine.readthedocs.io/en/stable/api.html#themes>`_

    Returns:
        Tuple containing matplotlib figure and axes along with the plotnine ggplot object

    Example:
        .. plot::

            import seaborn as sns
            import intedact
            data = sns.load_dataset('tips')
            intedact.histogram(data, 'total_bill')
    """

    # calculate bin number from data if not provided
    if hist_bins == 0:
        hist_bins = freedman_diaconis_bins(data[column], transform)

    data = trim_quantiles(
        data, column, lower_quantile=lower_quantile, upper_quantile=upper_quantile
    )
    data = preprocess_transformations(data, column, transform=transform)

    # plot histogram
    if bar_color is not None:
        args = {"fill": bar_color, "color": "black"}
    else:
        args = {"fill": BAR_COLOR, "color": "black"}
    if kde:
        ylabel = "Density"
        gg = (
            p9.ggplot(data, p9.aes(x=column, y="..density.."))
            + p9.geom_histogram(bins=hist_bins, **args)
            + p9.geom_density()
        )
    else:
        ylabel = "Count"
        gg = p9.ggplot(data, p9.aes(x=column)) + p9.geom_histogram(
            bins=hist_bins, **args
        )

    # handle axes transforms
    gg, xlabel = transform_axis(gg, column, transform, xaxis=True)

    if theme is not None:
        gg += eval(f"p9.{theme}()")
    else:
        gg += eval(f"p9.{THEME_DEFAULT}()")

    if fig and ax is not None:
        _ = gg._draw_using_figure(fig, [ax])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        gg += p9.labs(x=xlabel, y=ylabel)
        fig = gg.draw()

    # set the figure size
    if fig_width is not None and fig_height is not None:
        fig.set_size_inches(fig_width, fig_height)

    return fig, ax, gg


def time_series_countplot(
    data: pd.DataFrame,
    column: str,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    fig_width: int = None,
    fig_height: int = None,
    ts_type: str = "line",
    ts_freq: str = "auto",
    trend_line: str = "none",
    date_labels: str = None,
    date_breaks: str = None,
    theme: str = None,
) -> Tuple[plt.Figure, plt.Axes, p9.ggplot]:
    """
    Plots a times series plot of datetime column where the y axis is counts of observations aggregated a provided
    temporal frequency. Assumes that each row in the dataframe is a single event and not already aggregated.

    Args:
        data: pandas DataFrame to perform EDA on
        column: A string matching a column in the data
        fig: matplotlib Figure generated from blank ggplot to plot onto. If specified, must also specify ax
        ax: matplotlib axes generated from blank ggplot to plot onto. If specified, must also specify fig
        fig_width: Width of the plot in inches
        fig_height: Height of the plot in inches
        ts_type: 'line' plots a line graph while 'point' plots points for observations
        ts_freq: String describing the frequency at which to aggregate data in one of two formats:

            - A `pandas offset string <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_.
            - A human readable string in the same format passed to date breaks (e.g. "4 months")
            Default is to attempt to intelligently determine a good aggregation frequency.
        trend_line: Trend line to plot over data. Default is to plot no trend line. Other options are passed
            to `geom_smooth <https://plotnine.readthedocs.io/en/stable/generated/plotnine.geoms.geom_smooth.html>`_.
        date_labels: Date formatting string that will be passed to the labels argument
            of `scale_datetime <https://plotnine.readthedocs.io/en/stable/generated/plotnine.scales.scale.scale_datetime.html#plotnine.scales.scale.scale_datetime>`_.
        date_breaks: Date breaks string that will be passed to the breaks argument
            of `scale_datetime <https://plotnine.readthedocs.io/en/stable/generated/plotnine.scales.scale.scale_datetime.html#plotnine.scales.scale.scale_datetime>`_.
        theme: plotnine theme to use for the plot, str must match available theme listed `here <https://plotnine.readthedocs.io/en/stable/api.html#themes>`_.

    Returns:
        Tuple containing matplotlib figure and axes along with the plotnine ggplot object

    Example:
        .. plot::

            import pandas as pd
            import intedact
            data = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/tidytuesday_tweets/data.csv")
            data['created_at'] = pd.to_datetime(data.created_at)
            intedact.time_series_countplot(data, 'created_at', ts_freq='1W', trend_line='auto');
    """
    # TODO: Handle color specification
    if ts_freq == "auto":
        ts_freq = "1 month"
    ts_freq = convert_to_freq_string(ts_freq)

    # resample and aggregate time series counts
    tmp = (
        data.set_index(column)
        .resample(ts_freq)
        .agg("size")
        .reset_index()
        .rename({0: "Count"}, axis="columns")
    )

    fig, ax, gg = time_series_plot(
        tmp,
        column,
        "Count",
        fig_width=fig_width,
        fig_height=fig_height,
        fig=fig,
        ax=ax,
        ts_type=ts_type,
        trend_line=trend_line,
        date_labels=date_labels,
        date_breaks=date_breaks,
        theme=theme,
    )

    # add a twin axis for percentage
    add_percent_axis(ax, len(data[column]), flip_axis=False)

    return fig, ax, gg
