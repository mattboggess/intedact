from typing import Tuple
from typing import Union

import plotly.express as px
import plotly.graph_objects as go
import scipy.stats
import seaborn as sns
from IPython.display import display
from plotly.subplots import make_subplots

from .bivariate_plots import (
    numeric_2dplot,
)
from .data_utils import order_levels
from .data_utils import trim_values
from .plot_utils import *


def categorical_categorical_summary(
    data: pd.DataFrame,
    column1: str,
    column2: str,
    order1: Union[str, List] = "auto",
    order2: Union[str, List] = "auto",
    barmode: str = "stack",
    max_levels: int = 30,
    include_missing: bool = False,
) -> plt.Axes:
    """
    Generates an EDA summary of two categorical variables. This includes:
      - Categorical heatmap with counts and percentages of each pair of levels
      - Bar chart with relative percentage of column2 levels for each column1 level
      - Line plot with relative percentage of column2 levels for each column1 level

    Args:
        data: pandas DataFrame with data to be plotted
        column1: First categorical column in the data
        column2: Second categorical column in the data
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
        add_other: Whether to include

    Returns:

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
        add_other=True,
    )
    order1 = list(data[column1].cat.categories)
    data[column2] = order_levels(
        data,
        column2,
        None,
        order=order2,
        max_levels=max_levels,
        include_missing=include_missing,
        add_other=True,
    )
    order2 = list(data[column2].cat.categories)
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
    fig.update_layout(height=1000, width=1000, title_text="Cat Cat")
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

    # Make the line chart
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
    fig.update_layout(legend=dict(y=0.25, yanchor="middle"))

    fig.show()
    return fig


def numeric_categorical_summary(
    data: pd.DataFrame,
    column1: str,
    column2: str,
    order2: Union[str, List] = "auto",
    bins: int = 4,
    bin_type: str = "quantiles",
    max_levels: int = 30,
    include_missing: bool = False,
    add_other: bool = True,
) -> plt.Axes:
    """
    Generates an EDA summary of the relationship of a numeric variable on a categorical variable

    Args:
        data: pandas DataFrame with data to be plotted
        column1: First categorical column in the data
        column2: Second categorical column in the data
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
        add_other: Whether to include

    Returns:

    Examples:
        .. plot::

            import seaborn as sns
            import intedact
            data = sns.load_dataset('tips')
            intedact.countplot(data, 'day')
    """
    # Reorder categorical column levels
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
    colorway = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24

    if bin_type == "quantiles":
        data["interval"] = pd.qcut(data[column1], bins).astype(str)
    else:
        data["interval"] = pd.cut(data[column1], bins).astype(str)

    fig = make_subplots(
        rows=2,
        cols=1,
    )
    fig.update_layout(height=1000, width=1000)
    fig.update_traces(showscale=False)

    tmp = (
        data.groupby(["interval", column2])
        .size()
        .reset_index()
        .rename({0: "Count"}, axis="columns")
    )
    tmp["fraction"] = tmp.groupby("interval").Count.apply(lambda x: x / x.sum())

    # Make the line chart
    for i, o in enumerate(order2):
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
    # fig.update_coloraxes(showscale=False)
    # fig.update_layout(legend=dict(y=.25, yanchor="middle"))

    fig.show()


def numeric_numeric_bivariate_summary(
    data: pd.DataFrame,
    column1: str,
    column2: str,
    fig_height: int = 6,
    fig_width: int = 6,
    fontsize: int = 15,
    color_palette: str = None,
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
    reference_line: bool = False,
    match_axes: bool = False,
    interactive: bool = False,
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Creates a bivariate EDA summary for two numeric data column in a pandas DataFrame.

    Summary consists of a scatterplot with correlation coefficients.

    Args:
        data: pandas DataFrame to perform EDA on
        column1: name of column to plot on the x axis
        column2: name of column to plot on the y axis
        fig_height: Height of the plot in inches
        fig_width: Width of the plot in inches
        fontsize: Font size of axis and tick labels
        color_palette: Seaborn color palette to use
        plot_type: One of ['auto', 'hist', 'hex', 'kde', 'scatter']
        trend_line: Trend line to plot over data. Default is to plot no trend line. Other options are passed
            to `geom_smooth <https://plotnine.readthedocs.io/en/stable/generated/plotnine.geoms.geom_smooth.html>`_.
        alpha: Amount of transparency to add to the scatterplot points [0, 1]
        lower_quantile1: Lower quantile to filter data above for column1
        upper_quantile1: Upper quantile to filter data below for column1
        lower_quantile2: Lower quantile to filter data above for column2
        upper_quantile2: Upper quantile to filter data below for column2
        transform1: Transformation to apply to the data for plotting:

         - **'identity'**: no transformation
         - **'log'**: apply a logarithmic transformation to the data
        transform2: Transformation to apply to the column2 data for plotting. Same options as for column1.
        clip: Value to clip zero values to for log transformation. If 0 (default), zero values are simply removed.
        reference_line: Add a y = x reference line
        match_axes: Match the x and y axis limits
        interactive: Whether to modify to be used with interactive for ipywidgets

    Returns:
        Tuple containing matplotlib Figure drawn and summary stats DataFrame

    """
    data = data.copy()

    if color_palette != "":
        sns.set_palette(color_palette)
    else:
        sns.set_palette("tab10")

    ax, fig = numeric_2dplot(
        data,
        column1,
        column2,
        plot_type=plot_type,
        trend_line=trend_line,
        bins=bins,
        alpha=alpha,
        lower_quantile1=lower_quantile1,
        lower_quantile2=lower_quantile2,
        upper_quantile1=upper_quantile1,
        upper_quantile2=upper_quantile2,
        transform1=transform1,
        transform2=transform2,
        clip=clip,
        reference_line=reference_line,
        match_axes=match_axes,
    )
    fig.set_size_inches(fig_height, fig_width)
    set_fontsize(ax, fontsize)

    data = trim_values(data, column1, lower_quantile1, upper_quantile1)
    data = trim_values(data, column2, lower_quantile2, upper_quantile2)
    spearman = scipy.stats.spearmanr(data[column1], data[column2])
    pearson = scipy.stats.pearsonr(data[column1], data[column2])
    table = pd.DataFrame({"Pearson": [pearson[0]], "Spearman": [spearman[0]]})

    if interactive:
        display(table)
        plt.show()

    return table, fig
