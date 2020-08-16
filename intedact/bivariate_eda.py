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
from .utils import _rotate_labels
from .config import *


def continuous_continuous_bivariate_eda(
    data, column1, column2, fig_width=6, fig_height=6, trend_line='auto', alpha=1,
    lower_quantile1=0, upper_quantile1=1, lower_quantile2=0, upper_quantile2=1,
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
    gs = gridspec.GridSpec(1, 2)

    ax_scatter = fig.add_subplot(gs[0])
    gg_scatter = ggplot(data, aes(x=column1, y=column2)) + geom_point(alpha=alpha)
    ax_hist = fig.add_subplot(gs[1])
    gg_hist = ggplot(data, aes(x=column1, y=column2)) + geom_bin2d()

    # overlay density
    if plot_density:
        gg_scatter += geom_density_2d()
        gg_hist += geom_density_2d()

    # add reference line 
    if reference_line:
        gg_scatter += geom_abline(color='black')
        gg_hist += geom_abline(color='black')

    # add trend line
    if trend_line != 'none':
        gg_scatter += geom_smooth(method=trend_line, color='red')
        gg_hist += geom_smooth(method=trend_line, color='red')

    # handle transforms
    if transform1 in ['log', 'log_exclude0']:
        gg_scatter += scale_x_log10()
        gg_hist += scale_x_log10()
    elif transform1 == 'sqrt':
        gg_scatter += scale_x_sqrt()
        gg_hist += scale_x_sqrt()
    if transform2 in ['log', 'log_exclude0']:
        gg_scatter += scale_y_log10()
        gg_hist += scale_y_log10()
    elif transform2 == 'sqrt':
        gg_scatter += scale_y_sqrt()
        gg_hist += scale_y_sqrt()

    # handle aspect ratio
    _ = gg_scatter._draw_using_figure(fig, [ax_scatter])
    _ = gg_hist._draw_using_figure(fig, [ax_hist])
    if equalize_axes:
        upper = max(ax_scatter.get_xlim()[1], ax_scatter.get_ylim()[1])
        lower = min(ax_scatter.get_xlim()[0], ax_scatter.get_ylim()[0])
        gg_scatter += coord_fixed(ratio=1, xlim=(lower, upper), ylim=(lower, upper))
        gg_hist += coord_fixed(ratio=1, xlim=(lower, upper), ylim=(lower, upper))
        _ = gg_scatter._draw_using_figure(fig, [ax_scatter])
        _ = gg_hist._draw_using_figure(fig, [ax_hist])
        fig.set_size_inches(fig_width * 2, fig_width)
    else:
        fig.set_size_inches(fig_width * 2, fig_height)

    ax_scatter.set_xlabel(column1)
    ax_scatter.set_ylabel(column2)
    ax_hist.set_xlabel(column1)

    plt.show()


def discrete_discrete_bivariate_eda(data, column1, column2, fig_width=10, fig_height=6, level_order1='auto',
                                    level_order2='auto', top_n=20, normalize=False, rotate_labels=False):
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
    
    data[column1] = order_categorical(data, column1, None, level_order1, top_n)
    data[column2] = order_categorical(data, column2, None, level_order2, top_n)

    # draw heatmap of frequency table
    data_heat = (
        data
            .groupby([column1, column2])
            .size()
            .reset_index()
            .rename({0: 'count'}, axis='columns')
            .assign(percent=lambda x: 100 * (x['count'] / x['count'].sum()))
            .assign(percent_label=lambda x: [f"{v:.1f}%" for v in x['percent']])
    )
    if normalize:
        gg_heat = (
                ggplot(data_heat, aes(x=column1, y=column2, fill='percent')) +
                geom_tile() +
                geom_text(aes(label='percent_label'))
        )
    else:
        gg_heat = (
                ggplot(data_heat, aes(x=column1, y=column2, fill='count')) +
                geom_tile() +
                geom_text(aes(label='count'))
        )
    gg_heat += scale_fill_cmap('Blues')

    # draw clustered bar plot
    if normalize:
        data_cluster = (
            data
                .groupby([column1, column2])
                .size()
                .groupby(level=[0])
                .apply(lambda x: 100 * x / x.sum())
                .reset_index()
                .rename({0: 'percent'}, axis='columns')
        )
        gg_cluster = (
                ggplot(data_cluster, aes(x=column1, y='percent', fill=column2)) +
                geom_col(position='dodge') +
                scale_y_continuous(labels=lambda l: ["%d%%" % (v) for v in l]) +
                labs(y=f"Percent (normalized within {column1} levels)")
        )
    else:
        gg_cluster = (
                ggplot(data, aes(x=column1, fill=column2)) +
                geom_bar(position='dodge')
        )

    # draw discrete freqpoly
    if normalize:
        data_poly = (
            data
                .groupby([column2, column1])
                .size()
                .groupby(level=[0])
                .apply(lambda x: 100 * x / x.sum())
                .reset_index()
                .rename({0: 'percent'}, axis='columns')
        )
        gg_freqpoly = (
                ggplot(data_poly, aes(x=column1, y='percent', color=column2, group=column2)) +
                geom_line() +
                geom_point() +
                scale_y_continuous(labels=lambda l: ["%d%%" % (v) for v in l]) +
                labs(y = f"Percent (normalized within {column2} levels)")
        )
    else:
        data_poly = (
            data
                .groupby([column2, column1])
                .size()
                .reset_index()
                .rename({0: 'count'}, axis='columns')
        )
        gg_freqpoly = (
                ggplot(data_poly, aes(x=column1, y='count', color=column2, group=column2)) +
                geom_line() +
                geom_point()
        )

    if rotate_labels:
        gg_heat += theme(axis_text_x=element_text(rotation=90, hjust=1))
        gg_cluster += theme(axis_text_x=element_text(rotation=90, hjust=1))
        gg_freqpoly += theme(axis_text_x=element_text(rotation=90, hjust=1))
    else:
        gg_heat += theme(axis_text_x=element_text(rotation=0, hjust=1))
        gg_cluster += theme(axis_text_x=element_text(rotation=0, hjust=1))
        gg_freqpoly += theme(axis_text_x=element_text(rotation=0, hjust=1))


    f = gg_heat.draw()
    f.set_size_inches(fig_width, fig_height)
    f = gg_cluster.draw()
    f.set_size_inches(fig_width, fig_height)
    f = gg_freqpoly.draw()
    f.set_size_inches(fig_width, fig_height)


def binary_continuous_bivariate_eda(
    data, column1, column2, fig_width=10, fig_height=5, level_order='auto', top_n=20,
    alpha=.6, hist_bins=0, transform='identity', lower_quantile=0, upper_quantile=1,
    ref_lines=True, normalize_dist=False):

    data = data.copy().dropna(subset=[column1, column2])
    # preprocess column for transforms and remove outlier quantiles
    data = preprocess_numeric_variables(data, column2, lq1=lower_quantile, hq1=upper_quantile,
                                        transform1=transform)
    if hist_bins == 0:
        hist_bins = freedman_diaconis_bins(data[column2], transform)

    data[column1] = order_categorical(data, column1, column2, level_order, top_n, False)

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

    if normalize_dist:
        gg_pair = (
                ggplot(data, aes(fill=column1, x=column2)) +
                geom_density(alpha=alpha)
        )
    else:
        gg_pair = (
                ggplot(data, aes(fill=column1, x=column2)) +
                geom_histogram(alpha=alpha, position='identity', bins=hist_bins)
        )
    if ref_lines:
        gg_pair += geom_vline(data=summary, mapping=aes(xintercept=column2, color=column1, linetype='measure'), size=1.2)

    f = gg_pair.draw()
    f.set_size_inches(fig_width, fig_height)

def discrete_single_bivariate_eda(
    data, column1, column2, fig_width=10, fig_height=5, level_order='auto', top_n=20, transform='identity',
    lower_quantile=0, upper_quantile=1, ref_lines=True, flip_axis=False, rotate_labels=False):

    data = data.copy().dropna(subset=[column1, column2])
    # preprocess column for transforms and remove outlier quantiles
    data = preprocess_numeric_variables(data, column2, lq1=lower_quantile, hq1=upper_quantile,
                                        transform1=transform)
    if hist_bins == 0:
        hist_bins = freedman_diaconis_bins(data[column2], transform)

    summary = pd.DataFrame({
        'measure': ['median', 'mean'],
        column2: [data[column2].median(), data[column2].mean()]})

    data[column1] = order_categorical(data, column1, column2, level_order, top_n, flip_axis)
    data = data[data[column1] != '__OTHER__']
    if top_n < data[column1].nunique():
        print(f"WARNING: {data[column1].nunique() - top_n} levels excluded from plot.")

    gg_bar = (
            ggplot(data, aes(x=column1, y=column2)) +
            geom_col(fill=BAR_COLOR)
    )
    gg_point = (
            ggplot(data, aes(x=column1, y=column2)) +
            geom_point()
    )
    if ref_lines:
        gg_point += geom_hline(data=summary, mapping=aes(yintercept=column2, linetype='measure'), color='red')
        gg_bar += geom_hline(data=summary, mapping=aes(yintercept=column2, linetype='measure'), color='red')

    if flip_axis:
        gg_bar += coord_flip()
        gg_point += coord_flip()

    gg_bar = _rotate_labels(gg_bar, rotate_labels)
    gg_point = _rotate_labels(gg_point, rotate_labels)

    f = gg_bar.draw()
    f.set_size_inches(fig_width, fig_height)
    f = gg_point.draw()
    f.set_size_inches(fig_width, fig_height)


def nlevels_continuous_bivariate_eda(
    data, column1, column2, fig_width=10, fig_height=5, level_order='auto', top_n=20,
    hist_bins=0, transform='identity', lower_quantile=0, upper_quantile=1,
    ref_lines=True, normalize_dist=False, varwidth=True, flip_axis=False, rotate_labels=False):
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

    # only one numeric value per level
    if top_n < data[column1].nunique():
        print(f"WARNING: {data[column1].nunique() - top_n} levels condensed into __OTHER__.")

    summary = pd.DataFrame({
        'measure': ['median', 'mean'],
        column2: [data[column2].median(), data[column2].mean()]})

    data[column1] = order_categorical(data, column1, column2, level_order, top_n, flip_axis)
    gg_box = ggplot(data, aes(x=column1, y=column2))
    gg_violin = ggplot(data, aes(x=column1, y=column2))
    if ref_lines:
        gg_box += geom_hline(data=summary, mapping=aes(yintercept=column2, linetype='measure'), color='red')
        gg_violin += geom_hline(data=summary, mapping=aes(yintercept=column2, linetype='measure'), color='red')

    gg_box += geom_boxplot(fill=BAR_COLOR, varwidth=varwidth)
    gg_violin += geom_violin(fill=BAR_COLOR, draw_quantiles=[.25, .5, .75])

    gg_box = _rotate_labels(gg_box, rotate_labels)
    gg_violin = _rotate_labels(gg_violin, rotate_labels)

    if flip_axis:
        gg_box += coord_flip()
        gg_violin += coord_flip()

    data[column1] = order_categorical(data, column1, column2, level_order, top_n, False)
    gg_freq = ggplot(data, aes(color=column1, x=column2))
    if normalize_dist:
        gg_freq += geom_freqpoly(aes(y='..density..'), bins=hist_bins)
    else:
        gg_freq += geom_freqpoly(bins=hist_bins)

    f = gg_box.draw()
    f.set_size_inches(fig_width, fig_height)
    f = gg_violin.draw()
    f.set_size_inches(fig_width, fig_height)
    f = gg_freq.draw()
    f.set_size_inches(fig_width, fig_height)


def datetime_continuous_bivariate_eda(data, column1, column2, fig_width=10, fig_height=5, ts_freq='1M', delta_freq='1D'):

    # scatterplot (with and without trend line, alpha)
    # boxplots (needs resample frequency)
    # line plot
    data['month'] = data[column1].dt.month_name()
    data['day of month'] = data[column1].dt.day
    data['year'] = data[column1].dt.year
    data['hour'] = data[column1].dt.hour
    data['minute'] = data[column1].dt.minute
    data['second'] = data[column1].dt.second
    data['day of week'] = data[column1].dt.day_name()

    # compute time deltas
    dts = data[column1].sort_values(ascending=True)
    data['deltas'] = (dts - dts.shift(1)) / pd.Timedelta(delta_freq)

    # make histogram and boxplot figure (empty figure hack for plotting with subplots)
    # https://github.com/has2k1/plotnine/issues/373
    fig = (ggplot() + geom_blank(data=data) + theme_void()).draw()
    fig.set_size_inches(fig_width, fig_height * 4)
    gs = gridspec.GridSpec(4, 2)

    ax_line = fig.add_subplot(gs[0, :])
    gg_line = (
        ggplot(data, aes(x=column1, y=column2)) +
        geom_point() +
        geom_smooth()
    )
    _ = gg_line._draw_using_figure(fig, [ax_line])

    data[column1] = data[column1].dt.to_period(ts_freq)
    ax_box = fig.add_subplot(gs[1, :])
    print(data[column1].unique())
    gg_box = (
            ggplot(data, aes(x=column1, y=column2, group=column1)) +
            geom_boxplot()
    )
    _ = gg_box._draw_using_figure(fig, [ax_box])

    plt.show()

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
        col2_type=WIDGET_VALUES['col2_type']['widget_options'],
        manual_update=False
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
    widget.children[2].value = data.columns[1]
    widget.children[3].value = detect_column_type(data[data.columns[1]])

    display(widget)


def column_bivariate_eda_interact(data, column1, col1_type, column2, col2_type, manual_update=False):

    data = data.copy()
    data[column1] = coerce_column_type(data[column1], col1_type)
    data[column2] = coerce_column_type(data[column2], col2_type)
    print("Plot Controls:")

    # continuous-continuous
    if col1_type == 'continuous' and col2_type == 'continuous':
        WIDGET_VALUES['plot_type'] = WIDGET_VALUES['plot_type_cc']
        widget = interactive(
            continuous_continuous_bivariate_eda,
            {'manual': manual_update},
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

        if data.groupby(column1).size().max() == 1:
            widget = interactive(
                discrete_single_bivariate_eda,
                {'manual': manual_update},
                data=fixed(data),
                column1=fixed(column1),
                column2=fixed(column2),
                level_order=WIDGET_VALUES['level_order']['widget_options'],
                top_n=WIDGET_VALUES['top_n']['widget_options'],
                lower_quantile=WIDGET_VALUES['lower_quantile']['widget_options'],
                upper_quantile=WIDGET_VALUES['upper_quantile']['widget_options'],
                transform=WIDGET_VALUES['transform']['widget_options'],
            )
        elif len(data[column1].unique()) == 2:
            widget = interactive(
                binary_continuous_bivariate_eda,
                {'manual': manual_update},
                data=fixed(data),
                column1=fixed(column1),
                column2=fixed(column2),
                level_order=WIDGET_VALUES['level_order']['widget_options'],
                top_n=WIDGET_VALUES['top_n']['widget_options'],
                lower_quantile=WIDGET_VALUES['lower_quantile']['widget_options'],
                upper_quantile=WIDGET_VALUES['upper_quantile']['widget_options'],
                transform=WIDGET_VALUES['transform']['widget_options'],
            )
        else:
            widget = interactive(
                nlevels_continuous_bivariate_eda,
                {'manual': manual_update},
                data=fixed(data),
                column1=fixed(column1),
                column2=fixed(column2),
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
            {'manual': manual_update},
            data=fixed(data),
            column1=fixed(column1),
            column2=fixed(column2),
            fig_width=fig_width_range,
            fig_height=fig_height_range,
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
            {'manual': manual_update},
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
