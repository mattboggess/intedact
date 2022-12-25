from typing import List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from intedact.utils import bin_data, compute_trendline, order_levels, trim_values


def categorical_categorical_summary(
    data: pd.DataFrame,
    column1: str,
    column2: str,
    fig_height: int = 1000,
    fig_width: int = 1200,
    order1: Union[str, List] = "auto",
    order2: Union[str, List] = "auto",
    barmode: str = "stack",
    max_levels: int = 30,
    include_missing: bool = False,
    display_figure: bool = False,
) -> go.Figure:
    """
    Generates an EDA summary of two categorical variables

    Args:
        data: pandas DataFrame with data to be plotted
        column1: First categorical column in the data to plot as independent variable
        column2: Second categorical column in the data to plot as dependent variable
        fig_width: Figure width in pixels
        fig_height: Figure height in pixels
        order1: Order in which to sort the levels of the first variable:

         - **'auto'**: sorts ordinal variables by provided ordering, nominal variables by descending frequency, and numeric variables in sorted order.
         - **'descending'**: sorts in descending frequency.
         - **'ascending'**: sorts in ascending frequency.
         - **'sorted'**: sorts according to sorted order of the levels themselves.
         - **'random'**: produces a random order. Useful if there are too many levels for one plot.
         Or you can pass a list of level names in directly for your own custom order.
        order2: Same as order1 but for the second variable
        barmode: Type of bar plot aggregation. One of ['stack', 'group', 'overlay', 'relative']
        max_levels: Maximum number of levels to attempt to plot on a single plot. If exceeded, only the
         max_level - 1 levels will be plotted and the remainder will be grouped into an 'Other' category.
        include_missing: Whether to include missing values as an additional level in the data to be plotted
        display_figure: Whether to display the figure in addition to returning it
    """
    data = data.copy()
    order1 = order_levels(
        data,
        column1,
        None,
        order=order1,
        max_levels=max_levels,
        include_missing=include_missing,
        add_other=True,
    )
    order2 = order_levels(
        data,
        column2,
        None,
        order=order2,
        max_levels=max_levels,
        include_missing=include_missing,
        add_other=True,
    )

    colorway = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
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
    fig.update_layout(height=fig_height, width=fig_width)
    fig.update_traces(showscale=False)

    # Make the heatmap
    ct = pd.crosstab(data[column2], data[column1])
    annot = ct.applymap(lambda x: f"{100 * x / ct.sum().sum():.1f}%")
    fig.add_trace(
        go.Heatmap(
            z=ct,
            x=[str(x) for x in order1],
            y=[str(x) for x in order2],
            colorbar=None,
            hovertemplate=(
                f"{column1}"
                + ": %{x}<br>"
                + f"{column2}"
                + ": %{y}<br>"
                + "Count: %{z}<br>"
                + "Percent: %{text}"
                + "<extra></extra>"
            ),
            text=annot,
            texttemplate="%{z} (%{text})",
            coloraxis="coloraxis",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text=column2, row=1, col=1)
    fig.update_xaxes(row=1, col=1)

    tmp = (
        data.groupby([column1, column2])
        .size()
        .reset_index()
        .rename({0: "Count"}, axis="columns")
    )
    tmp["fraction"] = tmp.groupby(column1).Count.apply(lambda x: x / x.sum())
    # Make the barchart
    for i, o in enumerate(order2):
        fig.add_trace(
            go.Bar(
                x=order1,
                y=tmp[tmp[column2] == o].fraction,
                name=o,
                legendgroup=i,
                showlegend=False,
                marker={"color": colorway[i]},
                hovertemplate=(
                    f"{column1}"
                    + ": %{x}<br>"
                    + f"{column2}: {o}<br>"
                    + "Fraction: %{y}"
                    + "<extra></extra>"
                ),
            ),
            row=3,
            col=1,
        )
    fig.update_layout(barmode=barmode)

    for i, o in enumerate(order2):
        fig.add_trace(
            go.Scatter(
                x=order1,
                y=tmp[tmp[column2] == o].fraction,
                name=o,
                legendgroup=i,
                line=dict(color=colorway[i]),
                hovertemplate=(
                    f"{column1}"
                    + ": %{x}<br>"
                    + f"{column2}: {o}<br>"
                    + "Fraction: %{y}"
                    + "<extra></extra>"
                ),
            ),
            row=4,
            col=1,
        )
    fig.update_coloraxes(showscale=False)
    fig.update_layout(legend=dict(y=0.25, yanchor="middle", title=column2))
    fig.update_xaxes(title_text=column1, row=4, col=1)
    fig.update_yaxes(title_text="Fraction", row=3, col=1)
    fig.update_yaxes(title_text="Fraction", row=4, col=1)

    if display_figure:
        fig.show()
    return fig


def numeric_categorical_summary(
    data: pd.DataFrame,
    column1: str,
    column2: str,
    fig_height: int = 600,
    fig_width: int = 1200,
    order: Union[str, List] = "auto",
    num_intervals: int = 4,
    interval_type: str = "quantile",
    max_levels: int = 30,
    include_missing: bool = False,
    display_figure: bool = False,
) -> go.Figure:
    """
    Generates an EDA summary of the relationship of a numeric variable on a categorical variable.

    Args:
        data: pandas DataFrame with data to be plotted
        column1: Numeric column in the data to be plotted as independent variable
        column2: Categorical column in the data to be plotted as dependent variable
        fig_height: Height of the figure in pixels
        fig_width: Width of the figure in pixels
        order: Order in which to sort the levels of the categorical variable:

         - **'auto'**: sorts ordinal variables by provided ordering, nominal variables by descending frequency, and numeric variables in sorted order.
         - **'descending'**: sorts in descending frequency.
         - **'ascending'**: sorts in ascending frequency.
         - **'sorted'**: sorts according to sorted order of the levels themselves.
         - **'random'**: produces a random order. Useful if there are too many levels for one plot.
         Or you can pass a list of level names in directly for your own custom order.
        num_intervals: Number of intervals to bin column1 into
        interval_type: Type of intervals to bin column1 into.  'quantile' or 'equal width'
        max_levels: Maximum number of levels to attempt to plot on a single plot. If exceeded, only the
         max_level - 1 levels will be plotted and the remainder will be grouped into an 'Other' category.
        include_missing: Whether to include missing values as an additional level in the data to be plotted
        display_figure: Whether to display the figure in addition to returning it
    """
    data = data.copy()
    order = order_levels(
        data,
        column2,
        None,
        order=order,
        max_levels=max_levels,
        include_missing=include_missing,
        add_other=True,
    )
    colorway = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24

    bin_data(data, column1, num_intervals, interval_type)

    fig = make_subplots(
        rows=1,
        cols=1,
    )
    fig.update_layout(height=fig_height, width=fig_width)
    fig.update_traces(showscale=False)

    tmp = (
        data.groupby(["interval", column2])
        .size()
        .reset_index()
        .rename({0: "Count"}, axis="columns")
    )
    tmp["fraction"] = tmp.groupby("interval").Count.apply(lambda x: x / x.sum())

    for i, o in enumerate(order):
        fig.add_trace(
            go.Scatter(
                x=tmp[tmp[column2] == o].interval,
                y=tmp[tmp[column2] == o].fraction,
                name=o,
                legendgroup=i,
                line=dict(color=colorway[i]),
                hovertemplate=(
                    f"{column1}"
                    + ": %{x}<br>"
                    + f"{column2}: {o}<br>"
                    + "Fraction: %{y}"
                    + "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    fig.update_layout(legend=dict(title=column2))
    fig.update_xaxes(title_text=f"{column1} ({interval_type} bins)", row=1, col=1)
    fig.update_yaxes(title_text="Fraction", row=1, col=1)

    if display_figure:
        fig.show()
    return fig


def categorical_numeric_summary(
    data: pd.DataFrame,
    column1: str,
    column2: str,
    fig_height: int = 1000,
    fig_width: int = 1200,
    order: Union[str, List] = "auto",
    max_levels: int = 10,
    include_missing: bool = False,
    lower_quantile: float = 0,
    upper_quantile: float = 1,
    hist_bins: Optional[int] = None,
    dist_type: str = "kde_only",
    transform: str = "identity",
    display_figure: bool = False,
) -> go.Figure:
    """
    Generates an EDA summary of the relationship between a categorical variable as the independent variable and a
    numeric variable as the dependent variable.

    Args:
        data: pandas DataFrame with data to be plotted
        column1: Categorical column in the data to be used as independent variable
        column2: Numeric column in the data to be used as dependent variable
        fig_height: Height of the figure in pixels
        fig_width: Width of the figure in pixels
        order: Order in which to sort the levels of the categorical variable:

         - **'auto'**: sorts ordinal variables by provided ordering, nominal variables by descending frequency, and numeric variables in sorted order.
         - **'descending'**: sorts in descending frequency.
         - **'ascending'**: sorts in ascending frequency.
         - **'sorted'**: sorts according to sorted order of the levels themselves.
         - **'random'**: produces a random order. Useful if there are too many levels for one plot.
         Or you can pass a list of level names in directly for your own custom order.
        max_levels: Maximum number of levels to attempt to plot on a single plot. If exceeded, only the
         max_level - 1 levels will be plotted and the remainder will be grouped into an 'Other' category.
        include_missing: Whether to include missing values as an additional level in the data to be plotted
        lower_quantile: Lower quantile to filter numeric column above
        upper_quantile: Upper quantile to filter numeric column below
        hist_bins: Number of bins to use for the histogram. Default will use plotly defaults
        dist_type: Type of distribution to plot:

         - **'norm_hist+kde'**: Plots histograms with overlaid KDE normalized to be a probabililty density
         - **'norm_hist_only'**: Plots just histograms normalized to be a probabililty density
         - **'unnorm_hist_only'**: Plots just unnormalized histograms with counts
         - **'kde_only'**: Plots just KDEs normalized to be a probabililty density
        transform: Transformation to apply to the numeric column for plotting:

            - 'identity': no transformation
            - 'log': apply a logarithmic transformation (zero and negative values will be filtered out)
            - 'sqrt': apply a square root transformation
        display_figure: Whether to display the figure in addition to returning it
    """
    data = data.copy()
    if hist_bins == 0:
        hist_bins = None

    data = trim_values(data, column2, lower_quantile, upper_quantile)

    if transform == "log":
        label = f"log({column2})"
        data[label] = np.log(data[column2])
    elif transform == "sqrt":
        label = f"sqrt({column2})"
        data[label] = np.sqrt(data[column2])
    else:
        label = column2
    colorway = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24

    # Reorder categorical column levels
    order = order_levels(
        data,
        column1,
        label,
        order=order,
        max_levels=max_levels,
        include_missing=include_missing,
        add_other=True,
    )

    fig = px.histogram(
        data,
        x=label,
        color=column1,
        marginal="box",
        category_orders={column1: order},
        color_discrete_sequence=colorway,
        nbins=hist_bins,
        histnorm="probability density" if dist_type != "unnorm_hist_only" else None,
        barmode="overlay",
        opacity=0 if dist_type == "kde_only" else None,
    )

    if dist_type in ["norm_hist+kde", "kde_only"]:
        xs = [list(data[data[column1] == o][label]) for o in order]
        tmp_fig = ff.create_distplot(xs, order, show_hist=False, show_rug=False)
        for i, trace in enumerate(tmp_fig.data):
            trace["showlegend"] = False
            trace["line"]["color"] = colorway[i]
            fig.add_trace(trace, row=1, col=1)

    fig.update_layout(height=fig_height, width=fig_width)

    if display_figure:
        fig.show()
    return fig


def numeric_numeric_summary(
    data: pd.DataFrame,
    column1: str,
    column2: str,
    fig_height: int = 1200,
    fig_width: int = 1200,
    trend_line: str = "auto",
    opacity: float = 1.0,
    hist_bins: Optional[int] = None,
    lower_quantile1: float = 0,
    upper_quantile1: float = 1,
    lower_quantile2: float = 0,
    upper_quantile2: float = 1,
    num_intervals: int = 4,
    interval_type: str = "quantile",
    transform1: str = "identity",
    transform2: str = "identity",
    display_figure: bool = False,
) -> go.Figure:
    """
    Creates a bivariate EDA summary for two numeric data columns in a pandas DataFrame.

    Args:
        data: pandas DataFrame to perform EDA on
        column1: name of numeric column to plot as independent variable
        column2: name of numeric column to plot as dependent variable
        fig_height: Height of the plot in pixels
        fig_width: Width of the plot in pixels
        opacity: Level of opacity to apply to points in scatterplot (0 = fully transparent, 1 = fully opaque)
        trend_line: Trend line to plot over data. Default is to plot no trend line. Other options are passed
            to `geom_smooth <https://plotnine.readthedocs.io/en/stable/generated/plotnine.geoms.geom_smooth.html>`_.
        hist_bins: Number of bins to use for the histogram. Default will use plotly defaults
        lower_quantile1: Lower quantile to filter data above for column1
        upper_quantile1: Upper quantile to filter data below for column1
        lower_quantile2: Lower quantile to filter data above for column2
        upper_quantile2: Upper quantile to filter data below for column2
        num_intervals: Number of intervals to bin column1 into for the boxplots
        interval_type: Type of intervals to bin column1 into for the boxplots. 'quantile' or 'equal width'
        transform1: Transformation to apply to the column1 for plotting:

            - 'identity': no transformation
            - 'log': apply a logarithmic transformation (zero and negative values will be filtered out)
            - 'sqrt': apply a square root transformation
        transform2: Transformation to apply to the column2 data for plotting. Same options as for column1.
        display_figure: Whether to display the figure in addition to returning it
    """
    if hist_bins == 0:
        hist_bins = None
    data = data.copy()

    data = trim_values(data, column1, lower_quantile1, upper_quantile1)
    data = trim_values(data, column2, lower_quantile2, upper_quantile2)
    interval_order = bin_data(data, column1, num_intervals, interval_type)

    if transform1 == "sqrt":
        label1 = f"sqrt({column1})"
        data[label1] = np.sqrt(data[column1])
    else:
        label1 = column1

    if transform2 == "sqrt":
        label2 = f"sqrt({column2})"
        data[label2] = np.sqrt(data[column2])
    else:
        label2 = column2

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"colspan": 1, "rowspan": 1}, {"colspan": 1, "rowspan": 1}],
            [{"colspan": 2}, None],
        ],
    )
    fig.update_layout(height=fig_height, width=fig_width)

    fig.add_trace(
        go.Scatter(
            x=data[label1],
            y=data[label2],
            mode="markers",
            marker=dict(opacity=opacity),
        ),
        row=1,
        col=1,
    )
    if trend_line is not None:
        trend_data = compute_trendline(data, x=label1, y=label2, method=trend_line)
        fig.add_trace(
            go.Scatter(
                x=trend_data["x"],
                y=trend_data["y"],
                mode="lines",
                line=dict(color="red", width=2),
            ),
            row=1,
            col=1,
        )
    fig.update_xaxes(title_text=label1, row=1, col=1)
    fig.update_yaxes(title_text=label2, row=1, col=1)
    if transform1 == "log":
        fig.update_xaxes(type="log", row=1, col=1)
    if transform2 == "log":
        fig.update_yaxes(type="log", row=1, col=1)

    for i, o in enumerate(interval_order):
        fig.add_trace(
            go.Box(
                y=data[data["interval"] == o][label2],
                name=o,
                legendgroup=i,
            ),
            row=2,
            col=1,
        )
    fig.update_xaxes(title_text=f"{label1} ({interval_type} bins)", row=2, col=1)
    fig.update_yaxes(title_text=f"{label2}", row=2, col=1)
    if transform2 == "log":
        fig.update_yaxes(type="log", row=2, col=1)

    # Plotly doesn't support log histograms natively so we have to apply log to the data rather than use log axis
    if transform1 == "log":
        label1 = f"log({column1})"
        data[label1] = np.log(data[column1])
    if transform2 == "log":
        label2 = f"log({column2})"
        data[label2] = np.log(data[column2])

    fig.add_trace(
        go.Histogram2d(
            x=data[label1],
            y=data[label2],
            nbinsx=hist_bins,
            nbinsy=hist_bins,
            colorbar=dict(lenmode="fraction", len=0.5, yanchor="top", y=1),
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text=label1, row=1, col=2)
    fig.update_yaxes(title_text=label2, row=1, col=2)

    fig.update(layout_showlegend=False)
    if display_figure:
        fig.show()

    return fig
