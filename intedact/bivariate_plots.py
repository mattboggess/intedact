import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotnine as p9
from typing import Tuple
from .config import THEME_DEFAULT
from .utils import (
    convert_to_freq_string,
    trim_quantiles,
    preprocess_transformations,
    transform_axis,
    match_axes,
)


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
    Plots a times series plot of a datetime column and a numerical column.

    Args:
        data: pandas DataFrame to perform EDA on
        column1: A string matching a datetime column in the data
        column2: A string matching a numerical column in the data
        fig: matplotlib Figure generated from blank ggplot to plot onto. If specified, must also specify ax
        ax: matplotlib axes generated from blank ggplot to plot onto. If specified, must also specify fig
        fig_width: Width of the plot in inches
        fig_height: Height of the plot in inches
        ts_type: 'line' plots a line graph while 'point' plots points for observations
        ts_freq: pandas offset string that denotes the frequency at which to aggregate for the time series plot.
            Default is to attempt to automatically determine a reasonable time frequency to aggregate at.
            See `here <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_ for pandas offset string options.
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

    # TODO: make this intelligent
    ts_freq = convert_to_freq_string(ts_freq)
    if ts_freq == "auto":
        ts_freq = "1AS"

    # plot the line or point plot
    if ts_type == "line":
        gg = p9.ggplot(data, p9.aes(x=column1, y=column2)) + p9.geom_line()
    else:
        gg = p9.ggplot(data, p9.aes(x=column1, y=column2)) + p9.geom_point()

    # add smooth line if specified
    if trend_line != "none":
        gg += p9.geom_smooth(method=trend_line, color="red")

    # use provided breaks and labels if specified
    if date_breaks is not None and date_labels is not None:
        gg += p9.scale_x_datetime(date_breaks=date_breaks, date_labels=date_labels)
    elif date_breaks is not None:
        gg += p9.scale_x_datetime(date_breaks=date_breaks)
    elif date_labels is not None:
        gg += p9.scale_x_datetime(date_labels=date_labels)

    # use provided theme
    if theme is not None:
        gg += eval(f"p9.{theme}()")
    else:
        gg += eval(f"p9.{THEME_DEFAULT}()")

    # draw the figure
    # xlabel = f"{column1} (aggregated at {ts_freq})"
    xlabel = column1
    if fig and ax is not None:
        _ = gg._draw_using_figure(fig, [ax])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(column2)
    else:
        gg += p9.labs(x=xlabel, y=column2)
        fig = gg.draw()
        ax = fig.axes[0]

    # use matplotlib default date labels if non specified
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    if date_breaks is None:
        ax.xaxis.set_major_locator(locator)
    if date_labels is None:
        ax.xaxis.set_major_formatter(formatter)

    # set the figure size
    if fig_width is not None and fig_height is not None:
        fig.set_size_inches(fig_width, fig_height)

    return fig, ax, gg
