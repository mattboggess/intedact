import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from ipywidgets import interactive, fixed, Layout, Button, Output, HBox, VBox
from collections import Counter
from itertools import combinations
import matplotlib.ticker as mtick 
from plotnine import *
from matplotlib import gridspec
import warnings
from .utils import *
from .config import *


def continuous_continuous_bivariate_eda(
    data, column1, column2, fig_width=6, fig_height=6, plot_type='auto', trend_line='auto', 
    lower_quantile1=0, upper_quantile1=1, lower_quantile2=0, upper_quantile2=1, alpha=1,
    transform1='identity', transform2='identity', equalize_axes=False,  reference_line=False,
    plot_density=False):
    """
    Creates an EDA plot for two continuous variables.
    
    Parameters
    ----------
    data: pandas.DataFrame
        Dataset to perform EDA on 
    column1: str
        A string matching a column in the data to be used as the independent variable
    column2: str
        A string matching a column in the data to be used as the dependent variable
    fig_width: int
        Width of figure in inches
    fig_height: int
        Height of figure in inches
    plot_type: ['auto', 'scatter', 'bin2d', 'count']
        Type of plot to show:
        - 'auto': Defaults to scatter plot.
        - 'scatter': Draw a scatter plot using geom_scatter.
        - 'bin2d': Draw a 2d histogram using geom_bin2d.
        - 'count': Draw a 2d count plot using geom_count.
    trend_line: ['auto', 'none', 'loess', 'lm']
        Trend line to plot over data. 'none' will plot no trend line. Other options are passed
        to plotnine's geom_smooth.
    equalize_axes: bool
        Whether to square the aspect ratio and match the axis limits to create a directly comparable ratio
    reference_line: bool
        Whether to add a y = x reference line
    plot_density: bool
        Whether to overlay a 2d density on the given plot
    alpha: float, [0, 1]
        The amount of alpha to apply to points for the scatter plot type
    transform1: str, ['identity', 'log', 'log_exclude0', 'sqrt']
        Transformation to apply to the first column for plotting:
          - 'identity': no transformation
          - 'log': apply a logarithmic transformation with small constant added in case of 0
          - 'log_exclude0': apply a logarithmic transformation with zero removed
          - 'sqrt': apply a square root transformation
    transform2: str, ['identity', 'log', 'log_exclude0', 'sqrt']
        Transformation to apply to the second column (same as transform1)
    lower_quantile1: float, optional [0, 1]
        Lower quantile of column1 to remove before plotting for ignoring outliers
    upper_quantile1: float, optional [0, 1]
        Upper quantile of column2 to remove before plotting for ignoring outliers
    lower_quantile2: float, optional [0, 1]
        Same as lower_quantile1 but for column2
    upper_quantile2: float, optional [0, 1]
        Same as upper_quantile1 but for column2

    Returns 
    -------
    None
       Draws the plot to the current matplotlib figure
    """
    
    data = preprocess_numeric_variables(data, column1, column2, lq1=lower_quantile1, hq1=upper_quantile1,
                                        lq2=lower_quantile2, hq2=upper_quantile2, transform1=transform1,
                                        transform2=transform2)
    
    # make histogram and boxplot figure (empty figure hack for plotting with subplots/getting ax
    # handle): https://github.com/has2k1/plotnine/issues/373
    fig = (ggplot() + geom_blank(data=data) + theme_void()).draw()
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0])
    
    if plot_type == 'auto':
        plot_type = 'scatter'
    
    gg = ggplot(data, aes(x=column1, y=column2))
    if plot_type == 'scatter':
        gg += geom_point(alpha=alpha)
    elif plot_type == 'bin2d':
        gg += geom_bin2d()
    elif plot_type == 'count':
        gg += geom_count()
    else:
        raise ValueError(f"Unsupported plot type {plot_type}")
    
    # overlay density
    if plot_density:
        gg += geom_density_2d()
    
    # add reference line 
    if reference_line:
        gg += geom_abline(color='black')
        
    # add trend line
    if trend_line != 'none':
        gg += geom_smooth(method=trend_line, color='red')
        
    # handle transforms
    if transform1 in ['log', 'log_exclude0']:
        gg += scale_x_log10()
    elif transform1 == 'sqrt':
        gg += scale_x_sqrt()
    if transform2 in ['log', 'log_exclude0']:
        gg += scale_y_log10()
    elif transform2 == 'sqrt':
        gg += scale_x_sqrt()
        
    # handle aspect ratio
    _ = gg._draw_using_figure(fig, [ax])
    if equalize_axes:
        upper = max(ax.get_xlim()[1], ax.get_ylim()[1])
        lower = min(ax.get_xlim()[0], ax.get_ylim()[0])
        gg += coord_fixed(ratio=1, xlim=(lower, upper), ylim=(lower, upper))
        _ = gg._draw_using_figure(fig, [ax])
        fig.set_size_inches(fig_width, fig_width)
    else:
        fig.set_size_inches(fig_width, fig_height)
        
    ax.set_xlabel(column1)
    ax.set_ylabel(column2)
    
    plt.show()

def discrete_discrete_bivariate_eda(data, column1, column2, fig_width=12, fig_height=6, plot_type='auto',
                                    rotate_labels=False, level_order1='auto', level_order2='auto', 
                                    top_n=20, normalize=False, flip_axis=False):
    """ 
    Creates an EDA plot for two discrete variables.
    
    Parameters
    ----------
    data: pandas.DataFrame
        Dataset to perform EDA on 
    column1: str
        A string matching a column in the data to be used as the independent variable
    column2: str
        A string matching a column in the data to be used as the dependent variable
    fig_width: int
        Width of figure in inches
    fig_height: int
        Height of figure in inches
    plot: ['auto', 'clustered_bar', 'faceted_bar', 'freqpoly', 'count', 'heatmap']
        Type of plot to show. 
        - 'auto':  
        - 'clustered_bar': Bar plot where each level of column1 is broken into multiple clustered
          bars colored by the level of column2
        - 'faceted_bar': Bar plot faceted by levels of column2
        - 'freqpoly': Line plot with separate lines for counts in each level of column2 
        - 'count': 2d countplot using geom_count where counts are represented by areas of circles
        - 'heatmap': 2d heatmap where counts are represented using color in a heatmap
    rotate_labels: bool, optional
        Whether to rotate x axis labels to prevent overlap. 
    level_order1: ['auto', 'descending', 'ascending', 'sorted', 'random']
        Order in which to order the levels for the first column. 
         - 'auto' sorts ordinal variables by provided ordering, nominal variables by 
            descending frequency, and numeric variables in sorted order.
         - 'descending' sorts in descending frequency.
         - 'ascending' sorts in ascending frequency.
         - 'sorted' sorts according to sorted order of the levels themselves.
         - 'random' produces a random order. Useful if there are too many levels for one plot. 
    level_order2: ['auto', 'descending', 'ascending', 'sorted', 'random']
        Same as level_order1 but for the second column.
    top_n: int, optional 
        Maximum number of levels to attempt to plot on a single plot. If exceeded, only the 
        top_n - 1 levels will be plotted individually and the remainder will be grouped into an 
        'Other' category. 
    normalize: bool, optional
        - 'clustered_bar': Normalizes counts within levels of column1 so that each cluster sums to 100% 
        - 'faceted_bar': Normalizes counts within levels of column2 so that each facet sums to 100% 
        - 'freqpoly': Normalizes counts within levels of column2 so that each line sums to 100% 
        - 'count': No effect 
        - 'heatmap': Replaces counts with overall percentages across both levels 
    flip_axis: bool, optional
        Whether to flip axes for each plot.
        
    Returns 
    -------
    None
       Draws the plot to the current matplotlib figure
    """
    data = data.copy().dropna(subset=[column1, column2])
    
    if plot_type == 'auto':
        plot = 'clustered_bar'
        
    if plot_type in ['faceted_bar', 'freqpoly']:
        flip_axis = False
        
    data[column1] = order_categorical(data, column1, None, level_order1, top_n, flip_axis)
    data[column2] = order_categorical(data, column2, None, level_order2, top_n, flip_axis)
        
    if plot_type == 'clustered_bar':
        if normalize:
            data = (
              data
              .groupby([column1, column2])
              .size()
              .groupby(level=[0])
              .apply(lambda x: 100 * x / x.sum())
              .reset_index()
              .rename({0: 'percent'}, axis='columns')
            )
            gg = (
              ggplot(data, aes(x=column1, y='percent', fill=column2)) + 
              geom_col(position='dodge') + 
              scale_y_continuous(labels=lambda l: ["%d%%" % (v) for v in l]) +
              labs(y=f"Percent (normalized within {column1} levels)")
            )
        else:
            gg = (
              ggplot(data, aes(x=column1, fill=column2)) + 
              geom_bar(position='dodge')
            )
            
    elif plot_type == 'faceted_bar':
        if normalize:
            data = (
              data
              .groupby([column2, column1])
              .size()
              .groupby(level=[0])
              .apply(lambda x: 100 * x / x.sum())
              .reset_index()
              .rename({0: 'percent'}, axis='columns')
            )
            gg = (
              ggplot(data, aes(x=column1, y='percent')) + 
              geom_col(position='dodge', fill='steelblue', color='black') +
              facet_grid(f"{column2} ~ .") +
              scale_y_continuous(labels=lambda l: ["%d%%" % (v) for v in l]) +
              labs(y=f"Percent (normalized within {column2} levels)")
            )
        else:
            gg = (
              ggplot(data, aes(x=column1)) + 
              geom_bar(position='dodge') + 
              facet_grid(f"{column2} ~ .")
            )
            
    elif plot == 'freqpoly':
        if normalize:
            data = (
              data
              .groupby([column2, column1])
              .size()
              .groupby(level=[0])
              .apply(lambda x: 100 * x / x.sum())
              .reset_index()
              .rename({0: 'percent'}, axis='columns')
            )
            gg = (
              ggplot(data, aes(x=column1, y='percent', color=column2, group=column2)) + 
              geom_line() +
              geom_point() +
              scale_y_continuous(labels=lambda l: ["%d%%" % (v) for v in l]) +
              labs(y = f"Percent (normalized within {column2} levels)")
            )
        else:
            data = (
              data
              .groupby([column2, column1])
              .size()
              .reset_index()
              .rename({0: 'count'}, axis='columns')
            )
            gg = (
              ggplot(data, aes(x=column1, y='count', color=column2, group=column2)) + 
              geom_line() + 
              geom_point()
            )
            
    elif plot == 'count':
        gg = (
          ggplot(data, aes(x=column1, y=column2)) + 
          geom_count()
        )
    elif plot == 'heatmap':
        data = (
          data
          .groupby([column1, column2])
          .size()
          .reset_index()
          .rename({0: 'count'}, axis='columns')
          .assign(percent=lambda x: 100 * (x['count'] / x['count'].sum()))
          .assign(percent_label=lambda x: [f"{v:.1f}%" for v in x['percent']])
        )
        if normalize:
            gg = (
              ggplot(data, aes(x=column1, y=column2, fill='percent')) + 
              geom_tile() + 
              geom_text(aes(label='percent_label'))
            )
        else:
            gg = (
              ggplot(data, aes(x=column1, y=column2, fill='count')) + 
              geom_tile() + 
              geom_text(aes(label='count'))
            )
            
    if rotate_labels:
        gg += theme(axis_text_x=element_text(rotation=90, hjust=1))
    else:
        gg += theme(axis_text_x=element_text(rotation=0, hjust=1))
            
    if flip_axis and plot not in ['freqpoly', 'faceted_bar']:
        gg += coord_flip()
        
    f = gg.draw()
    f.set_size_inches(fig_width, fig_height)


def discrete_continuous_bivariate_eda(
    data, column1, column2, fig_width=10, fig_height=5, plot_type='auto', level_order='auto', top_n=20,
    alpha=.6, hist_bins=0, transform='identity', lower_quantile=0, upper_quantile=1,
    normalize_freqpoly=False, flip_axis=False, varwidth=True, ref_lines=True, rotate_labels=False):
    """ 
    Creates an EDA plot for a discrete and a continuous variable.
    
    Parameters
    ----------
    data: pandas.DataFrame
        Dataset to perform EDA on 
    column1: str
        A string matching a column in the data to be used as the independent variable
    column2: str
        A string matching a column in the data to be used as the dependent variable
    fig_width: int
        Width of figure in inches
    fig_height: int
        Height of figure in inches
    plot_type: ['auto', 'bar', 'point', 'histogram', 'density', 'freqpoly', 'boxplot', 'violin',
                'faceted_histogram', 'faceted_density']
        Type of plot to show. 
        - 'auto': Defaults to bar if single column2 value per level of column1. histogram if column1 has
          2 levels. boxplot otherwise.
        - 'bar': Bar plot where each level of column1 has a single column2 value plotted as a bar
        - 'point': Point plot where each level of column1 has a single column2 value plotted as a point
        - 'histogram': Plot two overlapped histograms with alpha
        - 'density': Plot two overlapped densities with alpha
        - 'freqpoly': Line plot with separate lines for density of column2 in each level of column1
        - 'boxplot': One boxplot per level of column1 side by side
        - 'violin': One violinplot per level of column1 side by side
        - 'faceted_histogram': Vertically stacked facets with one histogram of column2 for each level of column1 by facet
        - 'faceted_density': Vertically stacked facets with one density of column2 for each level of column1 by facet
    level_order: ['auto', 'descending', 'ascending', 'sorted', 'random']
        Order in which to order the levels for the first column.
         - 'auto' sorts ordinal variables by provided ordering, nominal variables by
            descending median of column2, and numeric variables in sorted order.
         - 'descending' sorts in descending median of column2.
         - 'ascending' sorts in ascending median of column2.
         - 'sorted' sorts according to sorted order of the column1 levels themselves.
         - 'random' produces a random order. Useful if there are too many levels for one plot.
    top_n: int, optional
        Maximum number of levels to attempt to plot on a single plot. If exceeded, only the
        top_n - 1 levels will be plotted individually. For bar and point plots, the excess levels will be
        discarded. For other plots, the remainder will be grouped into an '__Other__' category.
    alpha: float, [0, 1]
        The amount of alpha to apply for the overlapping histograms/densities
    hist_bins: int
        Number of bins to use for the histograms/freqpoly (0 uses geom_histogram default bins)",
    rotate_labels: bool, optional
        Whether to rotate x axis labels to prevent overlap. 
    level_order2: ['auto', 'descending', 'ascending', 'sorted', 'random']
        Same as level_order1 but for the second column.
    normalize: bool, optional
        - 'clustered_bar': Normalizes counts within levels of column1 so that each cluster sums to 100% 
        - 'faceted_bar': Normalizes counts within levels of column2 so that each facet sums to 100% 
        - 'freqpoly': Normalizes counts within levels of column2 so that each line sums to 100% 
        - 'count': No effect 
        - 'heatmap': Replaces counts with overall percentages across both levels 
    flip_axis: bool, optional
        Whether to flip axes for each plot.
        
    Returns 
    -------
    None
       Draws the plot to the current matplotlib figure
    """
    data = data.copy().dropna(subset=[column1, column2])
    # preprocess column for transforms and remove outlier quantiles
    data = preprocess_numeric_variables(data, column2, lq1=lower_quantile, hq1=upper_quantile,
                                        transform1=transform)
    if hist_bins == 0:
        hist_bins = freedman_diaconis_bins(data[column2], transform)

    if plot_type == 'auto':
        if data.groupby(column1).size().max() == 1:
            plot_type = 'bar'
        elif len(data[column1].unique()) == 2:
            plot_type = 'histogram'
        else:
            plot_type = 'boxplot'

    if plot_type in ['faceted_histogram', 'faceted_density', 'freqpoly', 'histogram', 'density']:
        flip_axis = False
        rotate_labels = False

    # TODO: Fix freqpoly ordering
    # hand level ordering and condensing extra levels into __OTHER__
    if level_order == 'auto' and plot_type in ['faceted_histogram', 'faceted_density'] and not is_numeric_dtype(data[column1]):
        level_order = 'ascending'
    if top_n < data[column1].nunique():
        ref_lines = False
        if plot_type in ['bar', 'point']:
            print(f"WARNING: {data[column1].nunique() - top_n} levels excluded from plot.")
        else:
            print(f"WARNING: {data[column1].nunique() - top_n} levels condensed into __OTHER__.")
    data[column1] = order_categorical(data, column1, column2, level_order, top_n, flip_axis)
    if plot_type in ['bar', 'point']:
        data = data[data[column1] != '__OTHER__']

    # compute mean and median summaries for displaying reference lines on plots
    if ref_lines:
        if plot_type in ['bar', 'point', 'boxplot', 'violin']:
            summary = pd.DataFrame({
                'measure': ['median', 'mean'],
                column2: [data[column2].median(), data[column2].mean()]})
        else:
            summary = (
                data
                .groupby(column1)[column2]
                .agg(['median', 'mean'])
                .reset_index()
                .melt(
                    id_vars=column1,
                    value_vars=['median', 'mean'],
                    var_name='measure',
                    value_name=column2
                )
            )
        summary['measure'] = summary['measure'].astype(pd.CategoricalDtype(['median', 'mean'], ordered=True))

    if plot_type == 'bar':
        gg = (
            ggplot(data, aes(x=column1, y=column2)) +
            geom_col(fill=BAR_COLOR)
        )
        if ref_lines:
            gg += geom_hline(data=summary, mapping=aes(yintercept=column2, linetype='measure'), color='red')

    if plot_type == 'point':
        gg = (
                ggplot(data, aes(x=column1, y=column2)) +
                geom_point()
        )
        if ref_lines:
            gg += geom_hline(data=summary, mapping=aes(yintercept=column2, linetype='measure'), color='red')

    elif plot_type == 'histogram':
        gg = (
          ggplot(data, aes(fill=column1, x=column2)) + 
          geom_histogram(alpha=alpha, position='identity', bins=hist_bins)
        )
        if ref_lines:
            gg += geom_vline(data=summary, mapping=aes(xintercept=column2, color=column1, linetype='measure'), size=1.2)

    elif plot_type == 'density':
        gg = (
          ggplot(data, aes(fill=column1, x=column2)) +
          geom_density(alpha=alpha)
        )
        if ref_lines:
            gg += geom_vline(data=summary, mapping=aes(xintercept=column2, color=column1, linetype='measure'), size=1.2)

    elif plot_type == 'boxplot':
        gg = ggplot(data, aes(x=column1, y=column2))
        if ref_lines:
            gg += geom_hline(data=summary, mapping=aes(yintercept=column2, linetype='measure'), color='red')
        gg += geom_boxplot(fill=BAR_COLOR, varwidth=varwidth)

    elif plot_type == 'violin':
        gg = ggplot(data, aes(x=column1, y=column2))
        if ref_lines:
            gg += geom_hline(data=summary, mapping=aes(yintercept=column2, linetype='measure'), color='red')
        gg += geom_violin(fill=BAR_COLOR, draw_quantiles=[.25, .5, .75])

    elif plot_type == 'freqpoly':
        gg = ggplot(data, aes(color=column1, x=column2))
        if normalize_freqpoly:
            gg += geom_freqpoly(aes(y='..density..'), bins=hist_bins)
        else:
            gg += geom_freqpoly(bins=hist_bins)

    elif plot_type == 'faceted_histogram':
        gg = (
          ggplot(data, aes(x=column2)) +
          geom_histogram(fill=BAR_COLOR, color='black', bins=hist_bins) +
          facet_grid(f"{column1} ~ .")
        )
        if ref_lines:
            gg += geom_vline(data=summary, mapping=aes(xintercept=column2, linetype='measure'), size=1.2, color='red')

    elif plot_type == 'faceted_density':
        gg = (
          ggplot(data, aes(x=column2)) + 
          geom_density(fill=BAR_COLOR) +
          facet_grid(f"{column1} ~ .")
        )
        if ref_lines:
            gg += geom_vline(data=summary, mapping=aes(xintercept=column2, linetype='measure'), size=1.2, color='red')

    if flip_axis:
        gg += coord_flip()

    if rotate_labels:
        gg += theme(axis_text_x=element_text(rotation=90, hjust=1))
    else:
        gg += theme(axis_text_x=element_text(rotation=0))

    f = gg.draw()
    f.set_size_inches(fig_width, fig_height)


def datetime_continuous_bivariate_eda(data, column1, column2, fig_width=10, fig_height=5, plot_type='auto',
                                      alpha=1):

    # scatterplot (with and without trend line, alpha)
    # boxplots (needs resample frequency)
    # line plot

    gg = (
        ggplot(data, aes(x=column1, y=column2)) +
        geom_point()
    )
    f = gg.draw()
    f.set_size_inches(fig_width, fig_height)

def bivariate_eda_interact(data):
    pd.set_option('precision', 2)
    sns.set_style('whitegrid')
    theme_set(theme_bw())
    warnings.simplefilter("ignore")
    
    widget = interactive(
        column_bivariate_eda_interact, 
        data=fixed(data), 
        column1=data.columns,
        col1_type=WIDGET_VALUES['col1_type']['widget_options'],
        column2=data.columns,
        col2_type=WIDGET_VALUES['col2_type']['widget_options']
    )

    # organize plot controls and adjust descriptions/widths
    widget.layout = Layout(flex_flow='row wrap')
    for ch in widget.children:
        if hasattr(ch, 'description') and ch.description in WIDGET_VALUES:
            ch.style = {'description_width': WIDGET_VALUES[ch.description]['width']}
            ch.description = WIDGET_VALUES[ch.description]['description']

    # set up column types to be inferred from current column selections
    def match_type1(*args):
        widget.children[1].value = detect_column_type(data[widget.children[0].value])
    def match_type2(*args):
        widget.children[3].value = detect_column_type(data[widget.children[2].value])
    widget.children[0].observe(match_type1, 'value')
    widget.children[2].observe(match_type2, 'value')
    widget.children[1].value = detect_column_type(data[data.columns[0]])
    widget.children[3].value = detect_column_type(data[data.columns[0]])

    display(widget)


def column_bivariate_eda_interact(data, column1, col1_type, column2, col2_type):

    # ranges for widgets
    fig_height_range = (1, 30, 1)
    fig_width_range = (1, 30, 1)
    hist_bin_range = (0, 50, 1)
    cutoff_range = (0, 1, .01)
    level_orders = ['auto', 'ascending', 'descending', 'sorted', 'random']
    kde_default = False
    flip_axis_default = False
    transforms = ['identity', 'log', 'log_exclude0', 'sqrt']
    top_n_range = (5, 100, 1)
    num_outliers = (0, data.shape[0], 1)

    data = data.copy()
    data[column1] = coerce_column_type(data[column1], col1_type)
    data[column2] = coerce_column_type(data[column2], col2_type)
    print("Plot Controls:")

    # continuous-continuous
    if col1_type == 'continuous' and col2_type == 'continuous':
        WIDGET_VALUES['plot_type'] = WIDGET_VALUES['plot_type_cc']
        widget = interactive(
            continuous_continuous_bivariate_eda,
            data=fixed(data),
            column1=fixed(column1),
            column2=fixed(column2),
            plot_type=WIDGET_VALUES['plot_type']['widget_options'],
            trend_line=WIDGET_VALUES['trend_line']['widget_options'],
            alpha=WIDGET_VALUES['alpha']['widget_options'],
            transform1=WIDGET_VALUES['transform1']['widget_options'],
            transform2=WIDGET_VALUES['transform2']['widget_options'],
            lower_quantile1=WIDGET_VALUES['lower_quantile1']['widget_options'],
            upper_quantile1=WIDGET_VALUES['upper_quantile1']['widget_options'],
            lower_quantile2=WIDGET_VALUES['lower_quantile2']['widget_options'],
            upper_quantile2=WIDGET_VALUES['upper_quantile2']['widget_options']
        )

    elif (col1_type == 'discrete' and col2_type == 'continuous') or \
         (col2_type == 'discrete' and col1_type == 'continuous'):
        if col2_type == 'discrete':
            column1, column2 = column2, column1

        print(data.groupby(column1).size().max())
        if data.groupby(column1).size().max() == 1:
            WIDGET_VALUES['plot_type'] = WIDGET_VALUES['plot_type_dc1']
        elif len(data[column1].unique()) == 2:
            WIDGET_VALUES['plot_type'] = WIDGET_VALUES['plot_type_dc2']
        else:
            WIDGET_VALUES['plot_type'] = WIDGET_VALUES['plot_type_dcn']

        widget = interactive(
            discrete_continuous_bivariate_eda,
            data=fixed(data),
            column1=fixed(column1),
            column2=fixed(column2),
            plot_type=WIDGET_VALUES['plot_type']['widget_options'],
            level_order=WIDGET_VALUES['level_order']['widget_options'],
            top_n=WIDGET_VALUES['top_n']['widget_options'],
            alpha=WIDGET_VALUES['alpha']['widget_options'],
            hist_bins=WIDGET_VALUES['hist_bins']['widget_options'],
            lower_quantile=WIDGET_VALUES['lower_quantile']['widget_options'],
            upper_quantile=WIDGET_VALUES['upper_quantile']['widget_options'],
            transform=WIDGET_VALUES['transform']['widget_options'],
        )

    elif col1_type == 'discrete' and col2_type == 'discrete':
        widget = interactive(
            discrete_discrete_bivariate_eda,
            data=fixed(data),
            column1=fixed(column1),
            column2=fixed(column2),
            fig_width=fig_width_range,
            fig_height=fig_height_range,
            plot=['auto', 'clustered_bar', 'faceted_bar', 'freqpoly', 'count', 'heatmap'],
            level_order1=level_orders,
            level_order2=level_orders,
            top_n=top_n_range
        )

    elif (col1_type == 'datetime' and col2_type == 'continuous') or \
         (col2_type == 'datetime' and col1_type == 'continuous'):
        if col2_type == 'datetime':
            column1, column2 = column2, column1
        widget = interactive(
            datetime_continuous_bivariate_eda,
            data=fixed(data),
            column1=fixed(column1),
            column2=fixed(column2),
            fig_width=WIDGET_VALUES['fig_width']['widget_options'],
            fig_height=WIDGET_VALUES['fig_height']['widget_options']
        )
    else:
        print("No EDA support for these variable types")
        return 

    for ch in widget.children[:-1]:
        if hasattr(ch, 'description') and ch.description in WIDGET_VALUES:
            ch.style = {'description_width': WIDGET_VALUES[ch.description]['width']}
            ch.description = WIDGET_VALUES[ch.description]['description']
    widget.update()
    controls = HBox(widget.children[:-1], layout=Layout(flex_flow='row wrap'))
    output = widget.children[-1]
    display(VBox([controls, output]))
