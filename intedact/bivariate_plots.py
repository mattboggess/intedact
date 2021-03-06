import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotnine as p9
from typing import Tuple, Optional
from .data_utils import (
    convert_to_freq_string,
    match_axes,
    convert_date_breaks,
    trim_values,
)

from .plot_utils import add_trendline, transform_axis
import seaborn as sns


def histogram2d(
    data: pd.DataFrame,
    column1: str,
    column2: str,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    fig_width: int = 6,
    fig_height: int = 6,
    trend_line: str = "auto",
    lower_quantile1: float = 0,
    upper_quantile1: float = 1,
    lower_quantile2: float = 0,
    upper_quantile2: float = 1,
    transform1: str = "identity",
    transform2: str = "identity",
    equalize_axes: bool = False,
    reference_line: bool = False,
    plot_density: bool = False,
) -> Tuple[plt.Figure, plt.Axes, p9.ggplot]:
    """
    Creates an EDA plot for two continuous variables.

    Args:
        data: pandas DataFrame containing data to be plotted
        column1: name of column to plot on the x axis
        column2: name of column to plot on the y axis
        fig: matplotlib Figure generated from blank ggplot to plot onto. If specified, must also specify ax
        ax: matplotlib axes generated from blank ggplot to plot onto. If specified, must also specify fig
        fig_width: figure width in inches
        fig_height: figure height in inches
        trend_line: Trend line to plot over data. Default is to plot no trend line. Other options are passed
            to `geom_smooth <https://plotnine.readthedocs.io/en/stable/generated/plotnine.geoms.geom_smooth.html>`_.
        lower_quantile1: Lower quantile of column1 data to remove before plotting for ignoring outliers
        upper_quantile1: Upper quantile of column1 data to remove before plotting for ignoring outliers
        lower_quantile2: Lower quantile of column2 data to remove before plotting for ignoring outliers
        upper_quantile2: Upper quantile of column2 data to remove before plotting for ignoring outliers
        transform1: Transformation to apply to the column1 data for plotting:

         - **'identity'**: no transformation
         - **'log'**: apply a logarithmic transformation with small constant added in case of zero values
         - **'log_exclude0'**: apply a logarithmic transformation with zero values removed
         - **'sqrt'**: apply a square root transformation
        transform2: Transformation to apply to the column2 data for plotting. Same options as for column1.
        equalize_axes: Square the aspect ratio and match the axis limits
        reference_line: Add a y = x reference line
        plot_density: Overlay a 2d density on the given plot

    Returns:
        Tuple containing matplotlib figure and axes along with the plotnine ggplot object

    Examples:
        .. plot::

            import pandas as pd
            import intedact
            data = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2018/2018-09-11/cats_vs_dogs.csv")
            intedact.histogram2d(data, 'n_dog_households', 'n_cat_households', equalize_axes=True, reference_line=True);
    """
    data = trim_quantiles(
        data, column1, lower_quantile=lower_quantile1, upper_quantile=upper_quantile1
    )
    data = trim_quantiles(
        data, column2, lower_quantile=lower_quantile2, upper_quantile=upper_quantile2
    )
    data = preprocess_transformations(data, column1, transform=transform1)
    data = preprocess_transformations(data, column2, transform=transform2)

    # draw the scatterplot
    gg = p9.ggplot(data, p9.aes(x=column1, y=column2)) + p9.geom_bin2d()

    # overlay density
    if plot_density:
        gg += p9.geom_density_2d()

    # add reference line
    if reference_line:
        gg += p9.geom_abline(color="black")

    # add trend line
    if trend_line != "none":
        gg += p9.geom_smooth(method=trend_line, color="red")

    gg += p9.labs(fill="")

    # handle axes transforms
    gg, xlabel = transform_axis(gg, column1, transform1, xaxis=True)
    gg, ylabel = transform_axis(gg, column2, transform2, xaxis=False)

    if fig is None and ax is None:
        gg.draw()
        fig = plt.gcf()
        ax = fig.axes[0]
    else:
        _ = gg._draw_using_figure(fig, [ax])

    if equalize_axes:
        fig, ax, gg = match_axes(fig, ax, gg)
        fig.set_size_inches(fig_width, fig_width)
    else:
        fig.set_size_inches(fig_width, fig_height)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    return fig, ax, gg


def scatterplot(
    data: pd.DataFrame,
    column1: str,
    column2: str,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    fig_width: int = 6,
    fig_height: int = 6,
    trend_line: str = "auto",
    alpha: float = 1,
    lower_quantile1: float = 0,
    upper_quantile1: float = 1,
    lower_quantile2: float = 0,
    upper_quantile2: float = 1,
    transform1: str = "identity",
    transform2: str = "identity",
    equalize_axes: bool = False,
    reference_line: bool = False,
    plot_density: bool = False,
) -> Tuple[plt.Figure, plt.Axes, p9.ggplot]:
    """
    Creates an EDA plot for two continuous variables.

    Args:
        data: pandas DataFrame containing data to be plotted
        column1: name of column to plot on the x axis
        column2: name of column to plot on the y axis
        fig: matplotlib Figure generated from blank ggplot to plot onto. If specified, must also specify ax
        ax: matplotlib axes generated from blank ggplot to plot onto. If specified, must also specify fig
        fig_width: figure width in inches
        fig_height: figure height in inches
        trend_line: Trend line to plot over data. Default is to plot no trend line. Other options are passed
            to `geom_smooth <https://plotnine.readthedocs.io/en/stable/generated/plotnine.geoms.geom_smooth.html>`_.
        alpha: The amount of alpha to apply to points for the scatter plot type (0 - 1)
        lower_quantile1: Lower quantile of column1 data to remove before plotting for ignoring outliers
        upper_quantile1: Upper quantile of column1 data to remove before plotting for ignoring outliers
        lower_quantile2: Lower quantile of column2 data to remove before plotting for ignoring outliers
        upper_quantile2: Upper quantile of column2 data to remove before plotting for ignoring outliers
        transform1: Transformation to apply to the column1 data for plotting:

         - **'identity'**: no transformation
         - **'log'**: apply a logarithmic transformation with small constant added in case of zero values
         - **'log_exclude0'**: apply a logarithmic transformation with zero values removed
         - **'sqrt'**: apply a square root transformation
        transform2: Transformation to apply to the column2 data for plotting. Same options as for column1.
        equalize_axes: Square the aspect ratio and match the axis limits
        reference_line: Add a y = x reference line
        plot_density: Overlay a 2d density on the given plot

    Returns:
        Tuple containing matplotlib figure and axes along with the plotnine ggplot object

    Examples:
        .. plot::

            import pandas as pd
            import intedact
            data = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2018/2018-09-11/cats_vs_dogs.csv")
            intedact.scatterplot(data, 'n_dog_households', 'n_cat_households', equalize_axes=True, reference_line=True);
    """
    data = trim_quantiles(
        data, column1, lower_quantile=lower_quantile1, upper_quantile=upper_quantile1
    )
    data = trim_quantiles(
        data, column2, lower_quantile=lower_quantile2, upper_quantile=upper_quantile2
    )
    data = preprocess_transformations(data, column1, transform=transform1)
    data = preprocess_transformations(data, column2, transform=transform2)

    # draw the scatterplot
    gg = p9.ggplot(data, p9.aes(x=column1, y=column2)) + p9.geom_point(alpha=alpha)

    # overlay density
    if plot_density:
        gg += p9.geom_density_2d()

    # add reference line
    if reference_line:
        gg += p9.geom_abline(color="black")

    # add trend line
    if trend_line != "none":
        gg += p9.geom_smooth(method=trend_line, color="red")

    # handle axes transforms
    gg, xlabel = transform_axis(gg, column1, transform1, xaxis=True)
    gg, ylabel = transform_axis(gg, column2, transform2, xaxis=False)

    if fig is None and ax is None:
        gg.draw()
        fig = plt.gcf()
        ax = fig.axes[0]
    else:
        _ = gg._draw_using_figure(fig, [ax])

    if equalize_axes:
        fig, ax, gg = match_axes(fig, ax, gg)
        fig.set_size_inches(fig_width, fig_width)
    else:
        fig.set_size_inches(fig_width, fig_height)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    return fig, ax, gg


def time_series_plot(
    data: pd.DataFrame,
    column1: str,
    column2: str,
    ax: Optional[plt.Axes] = None,
    ts_type: str = "point",
    trend_line: Optional[str] = None,
    date_labels: Optional[str] = None,
    date_breaks: Optional[str] = None,
    span: float = 0.75,
    ci_level: float = 0.95,
    **kwargs,
) -> plt.Axes:
    """
    Plots a times series plot of a datetime column and a numerical column.

    Args:
        data: pandas DataFrame to perform EDA on
        column1: A string matching a datetime column in the data
        column2: A string matching a numerical column in the data
        ax: matplotlib axes to draw plot onto
        ts_type: 'line' plots a line graph, 'point' plots points for observations
        trend_line: Trend line to plot over data. Default is to plot no trend line. Other options are passed
            to `geom_smooth <https://plotnine.readthedocs.io/en/stable/generated/plotnine.geoms.geom_smooth.html>`_.
        date_labels: strftime date formatting string that will be used to set the format of the x axis tick labels
        date_breaks: Date breaks string in form '{interval} {period}'. Interval must be an integer and period must be
          a time period ranging from seconds to years. (e.g. '1 year', '3 minutes')
        span: span parameter for loess
        ci_level: confidence level to use for drawing confidence interval

    Returns:
        matplotlib Axes to plot time series on

    Example:
        .. plot::

            import pandas as pd
            import intedact
            data = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/tidytuesday_tweets/data.csv")
            data['created_at'] = pd.to_datetime(data.created_at)
            intedact.time_series(data, 'created_at', trend_line='auto');
    """

    if ts_type == "point":
        ax = sns.scatterplot(data=data, ax=ax, x=column1, y=column2)
    else:
        ax = sns.lineplot(data=data, ax=ax, x=column1, y=column2)

    if trend_line is not None:
        ax = add_trendline(
            data, column1, column2, ax, method=trend_line, span=span, level=ci_level
        )

    # Set the date axis tick breaks
    if date_breaks is None:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    else:
        locator = convert_date_breaks(date_breaks)
    ax.xaxis.set_major_locator(locator)

    # Set the date axis tick label formats
    if date_labels is None:
        formatter = mdates.ConciseDateFormatter(locator)
    else:
        formatter = mdates.DateFormatter(date_labels)
    ax.xaxis.set_major_formatter(formatter)

    return ax
