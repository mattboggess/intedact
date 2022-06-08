# TODO: WIP
import calendar
from collections import Counter
from itertools import combinations
from typing import List
from typing import Tuple
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import tldextract
from IPython.display import display
from matplotlib import gridspec

from .bivariate_plots import (
    scatterplot,
)
from .bivariate_plots import time_series_plot
from .config import TIME_UNITS
from .data_utils import compute_time_deltas
from .data_utils import convert_to_freq_string
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
    trend_line: str = "auto",
    alpha: float = 1,
    lower_trim1: int = 0,
    upper_trim1: int = 0,
    lower_trim2: int = 0,
    upper_trim2: int = 0,
    transform1: str = "identity",
    transform2: str = "identity",
    clip: float = 0,
    reference_line: bool = False,
    plot_density: bool = False,
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
        trend_line: Trend line to plot over data. Default is to plot no trend line. Other options are passed
            to `geom_smooth <https://plotnine.readthedocs.io/en/stable/generated/plotnine.geoms.geom_smooth.html>`_.
        alpha: The amount of alpha to apply to points for the scatter plot type (0 - 1)
        lower_trim1: Number of values to trim from lower end of distribution for column1
        upper_trim1: Number of values to trim from upper end of distribution for column1
        lower_trim2: Number of values to trim from lower end of distribution for column1
        upper_trim2: Number of values to trim from upper end of distribution for column1
        transform1: Transformation to apply to the data for plotting:

         - **'identity'**: no transformation
         - **'log'**: apply a logarithmic transformation to the data
        transform2: Transformation to apply to the column2 data for plotting. Same options as for column1.
        clip: Value to clip zero values to for log transformation. If 0 (default), zero values are simply removed.
        reference_line: Add a y = x reference line
        plot_density: Overlay a 2d density on the given plot
        interactive: Whether to modify to be used with interactive for ipywidgets

    Returns:
        Tuple containing matplotlib Figure drawn and summary stats DataFrame

    """
    data = data.copy()

    if color_palette != "":
        sns.set_palette(color_palette)
    else:
        sns.set_palette("tab10")

    f, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax = scatterplot(
        data,
        column1,
        column2,
        ax=ax,
        trend_line=trend_line,
        alpha=alpha,
        lower_trim1=lower_trim1,
        lower_trim2=lower_trim2,
        upper_trim1=upper_trim1,
        upper_trim2=upper_trim2,
        transform1=transform1,
        transform2=transform2,
        clip=clip,
        reference_line=reference_line,
        plot_density=plot_density,
    )
    set_fontsize(ax, fontsize)

    data = trim_values(data, column1, lower_trim1, upper_trim1)
    data = trim_values(data, column2, lower_trim2, upper_trim2)
    spearman = scipy.stats.spearmanr(data[column1], data[column2])
    pearson = scipy.stats.pearsonr(data[column1], data[column2])
    table = pd.DataFrame({"Pearson": [pearson[0]], "Spearman": [spearman[0]]})

    if interactive:
        display(table)
        plt.show()

    return table, ax
