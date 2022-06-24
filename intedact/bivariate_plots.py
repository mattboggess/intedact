from typing import Optional
from typing import Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data_utils import convert_date_breaks
from .data_utils import freedman_diaconis_bins
from .data_utils import preprocess_transform
from .data_utils import trim_values
from .plot_utils import add_trendline
from .plot_utils import transform_axis


def numeric_2dplot(
    data: pd.DataFrame,
    column1: str,
    column2: str,
    plot_type: str = "scatter",
    trend_line: str = "auto",
    bins: Optional[int] = None,
    alpha: float = 1,
    lower_quantile1: float = 0,
    upper_quantile1: float = 1,
    lower_quantile2: float = 0,
    upper_quantile2: float = 1,
    transform1: str = "identity",
    transform2: str = "identity",
    clip: float = 0,
    ci_level=0.95,
    span=0.75,
    reference_line: bool = False,
    match_axes: bool = False,
) -> Tuple[plt.Axes, plt.Figure]:
    """
    Creates an EDA plot for two numeric variables that is a wrapper around seaborn's jointplot.

    Args:
        data: pandas DataFrame containing data to be plotted
        column1: name of column to plot on the x axis
        column2: name of column to plot on the y axis
        plot_type: One of ['auto', 'hist', 'hex', 'kde', 'scatter']
        trend_line: Trend line to plot over data. Default is to plot no trend line. Other options are passed
            to `geom_smooth <https://plotnine.readthedocs.io/en/stable/generated/plotnine.geoms.geom_smooth.html>`_.
        bins: Number of bins to use for the histogram/hexplot. Default is to determine # of bins from the data
        alpha: Amount of transparency to add to the scatter plot points [0, 1]
        lower_quantile1: Lower quantile to filter data above for column1
        upper_quantile1: Upper quantile to filter data below for column1
        lower_quantile2: Lower quantile to filter data above for column2
        upper_quantile2: Upper quantile to filter data below for column2
        transform1: Transformation to apply to the data for plotting:

         - **'identity'**: no transformation
         - **'log'**: apply a logarithmic transformation to the data
        transform2: Transformation to apply to the column2 data for plotting. Same options as for column1.
        clip: Value to clip zero values to for log transformation. If 0 (default), zero values are simply removed.
        ci_level: Confidence level determining how wide to plot confidence intervals for trend line smoothing.
        span: Span parameter to determine amount of smoothing for loess
        reference_line: Add a y = x reference line
        match_axes: Match the x and y axis limits

    Returns:
        Matplotlib axes and figure with plot drawn

    Examples:
        .. plot::

            import seaborn as sns
            import intedact
            data = sns.load_dataset("iris")
            intedact.numeric_2dplot(data, 'sepal_length', 'sepal_width');
    """
    data = data.copy()
    data = data.dropna(subset=[column1, column2])

    # Remove upper and lower values
    data = trim_values(data, column1, lower_quantile1, upper_quantile1)
    data = trim_values(data, column2, lower_quantile2, upper_quantile2)

    # Clip/remove zeros for log transformation
    data = preprocess_transform(data, column1, transform1, clip=clip)
    data = preprocess_transform(data, column2, transform2, clip=clip)

    kws = dict(alpha=alpha)
    if plot_type != "scatter":
        kws["alpha"] = 1.0
    if plot_type == "hist":
        if bins is None:
            bins1 = freedman_diaconis_bins(data[column1], log=(transform1 == "log"))
            bins2 = freedman_diaconis_bins(data[column2], log=(transform2 == "log"))
            bins = max(bins1, bins2)
        kws["bins"] = bins
    if plot_type == "hex":
        if bins is None:
            bins = 30
        kws["gridsize"] = bins
        kws["mincnt"] = 1

    g = sns.jointplot(
        data=data, x=column1, y=column2, kind=plot_type, joint_kws=kws, cmap="viridis"
    )
    ax = g.figure.axes[0]
    ax_marg_x = g.figure.axes[1]
    ax_marg_y = g.figure.axes[2]

    if match_axes:
        max_val = max(data[column1].max(), data[column2].max())
        min_val = min(data[column1].min(), data[column2].min())
        ax.set_xlim((min_val, max_val))
        ax.set_ylim((min_val, max_val))

    if trend_line != "none":
        ax = add_trendline(
            data, column1, column2, ax, method=trend_line, span=span, level=ci_level
        )

    if reference_line:
        x_vals = np.array(ax.get_xlim())
        y_vals = x_vals
        ax.plot(x_vals, y_vals, "--")

    ax = transform_axis(ax, column1, transform=transform1, xaxis=True)
    ax = transform_axis(ax, column2, transform=transform2, xaxis=False)
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    plt.setp(ax_marg_x.get_xticklabels(minor=True), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(minor=True), visible=False)

    return ax, g.figure


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
