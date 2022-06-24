from typing import Tuple

import scipy.stats
import seaborn as sns
from IPython.display import display

from .bivariate_plots import (
    numeric_2dplot,
)
from .data_utils import trim_values
from .plot_utils import *


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
