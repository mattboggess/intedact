from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data_utils import convert_date_breaks
from .data_utils import freedman_diaconis_bins
from .data_utils import order_levels
from .data_utils import preprocess_transform
from .data_utils import trim_values
from .plot_utils import add_trendline
from .plot_utils import transform_axis


def categorical_categorical_summary(
    data: pd.DataFrame,
    column1: str,
    column2: str,
    ax: Optional[plt.Axes] = None,
    order1: Union[str, List] = "auto",
    order2: Union[str, List] = "auto",
    max_levels: int = 30,
    include_missing: bool = False,
    add_other: bool = True,
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
    # Reorder column levels
    data[column1] = order_levels(
        data,
        column1,
        None,
        order=order1,
        max_levels=max_levels,
        include_missing=include_missing,
        add_other=add_other,
    )
    order1 = list(data[column1].cat.categories)
    data[column2] = order_levels(
        data,
        column2,
        None,
        order=order2,
        max_levels=max_levels,
        include_missing=include_missing,
        add_other=add_other,
    )
    order2 = list(data[column2].cat.categories)

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(
        rows=4,
        cols=2,
        specs=[
            [{"colspan": 2, "rowspan": 2}, None],
            [None, None],
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
        ],
    )

    # Make the heatmap
    ct = pd.crosstab(data[column2], data[column1])
    annot = ct.applymap(lambda x: f"{x} ({100 * x / ct.sum().sum():.1f}%)")
    fig.add_trace(
        go.Heatmap(
            z=ct,
            x=[str(x) for x in order1],
            y=[str(x) for x in order2],
            hovertext=annot,
            text=annot,
            texttemplate="%{text}",
        ),
        row=1,
        col=1,
    )

    tmp = (
        data.groupby([column1, column2])
        .size()
        .reset_index()
        .rename({0: "Count"}, axis="columns")
    )
    tmp["fraction"] = tmp.groupby(column1).Count.apply(lambda x: x / x.sum())
    # Make the barchart
    for o in order2:
        fig.add_trace(
            go.Bar(x=order1, y=tmp[tmp[column2] == o].fraction, name=o),
            row=3,
            col=1,
        )
    fig.update_layout(barmode="group")

    # Make the line chart
    for o in order2:
        fig.add_trace(
            go.Line(x=order1, y=tmp[tmp[column2] == o].fraction, name=o),
            row=4,
            col=1,
        )

    fig.update_layout(height=1000, width=1000, title_text="Cat Cat")
    fig.show()


def categorical_heatmap(
    data: pd.DataFrame,
    column1: str,
    column2: str,
    ax: Optional[plt.Axes] = None,
    order1: Union[str, List] = "auto",
    order2: Union[str, List] = "auto",
    max_levels: int = 30,
    include_missing: bool = False,
    add_other: bool = True,
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
    # Reorder column levels
    data[column1] = order_levels(
        data,
        column1,
        None,
        order=order1,
        max_levels=max_levels,
        include_missing=include_missing,
        add_other=add_other,
    )
    order1 = list(data[column1].cat.categories)
    data[column2] = order_levels(
        data,
        column2,
        None,
        order=order2,
        max_levels=max_levels,
        include_missing=include_missing,
        add_other=add_other,
    )
    order2 = list(data[column2].cat.categories)

    # Make the heatmap
    ct = pd.crosstab(data[column1], data[column2])
    annot = ct.applymap(lambda x: f"{x} ({100 * x / ct.sum().sum():.1f}%)")
    # ax = sns.heatmap(data=ct, annot=annot, ax=ax, fmt="")

    # return ax
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Heatmap(
            z=ct,
            x=order2,
            y=order1,
            hovertext=annot,
            text=annot,
            texttemplate="%{text}",
        ),
        row=1,
        col=1,
    )

    fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")
    fig.show()


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
