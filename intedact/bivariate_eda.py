import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from ipywidgets import interactive, fixed, Layout, Button, Output
from collections import Counter
from itertools import combinations
import matplotlib.ticker as mtick 
from plotnine import *
from matplotlib import gridspec
import warnings
from utils import order_categorical, categorize_column_type, preprocess_numeric_variables

DESCRIPTIONS = {
    'column1': ("column1: Column to be plotted as independent variable", '23%'),
    'column2': ("column2: Column to be plotted as dependent variable", '24%'),
    'discrete_limit': (("discrete_limit: # of unique values a variable must have before it is considered "
                        "continuous rather than discrete"), '35%'),
    'fig_width': ("fig_width: width of figure in inches", '25%'),
    'fig_height': ("fig_height: height of figure in inches (multiplied if multiple subplots)", '26%'),
    'plot_type': ("plot_type: type of plot to display (see docs for details)", '25%')
    #'level1_order': "level1_order: Order for arranging column1 levels on plot",
    #'level2_order': "level2_order: Order for arranging column2 levels on plot",
    #'normalize': "normalize: Whether to normalize counts to relative proportions (normalization varies by plot type)",
    #'flip_axis': "flip_axis: Whether to flip y and x axes for horizontal display",
    #'top_n': "top_n: Maximum number of levels to display before condensing remaining into 'Other'"
}


def continuous_continuous_bivariate_eda(
    data, column1, column2, fig_width=6, fig_height=6, plot_type='auto', trend_line='auto', 
    aspect_ratio='manual',  reference_line=False, plot_density=False, alpha=1,
    lower_cutoff1=0, upper_cutoff1=1, lower_cutoff2=0, upper_cutoff2=1, 
    transform1='identity', transform2='identity'):
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
    plot_type: ['auto', 'scatter', 'bin2d', 'boxplot']
        Type of plot to show. 
        - 'auto':  
        - 'scatter': 
        - 'bin2d':
        - 'boxplot':
    trend_line: ['auto', 'none', 'loess', 'lm']
        Trend line to plot over data. 'none' will plot no trend line. Other options are passed
        to plotnine's geom_smooth.
    aspect_ratio: ['manual', 'equal_axes', 'bank_to_45']
    reference_line: bool
    plot_density: bool
        Whether to overlay a 2d density on the given plot
        
    Returns 
    -------
    None
       Draws the plot to the current matplotlib figure
    """
    
    data = preprocess_numeric_variables(data, column1, column2, lq1=lower_cutoff1, hq1=upper_cutoff1,
                                        lq2=lower_cutoff2, hq2=upper_cutoff2, transform1=transform1,
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
    if aspect_ratio == 'equal_axes':
        upper = max(ax.get_xlim()[1], ax.get_ylim()[1])
        lower = min(ax.get_xlim()[0], ax.get_ylim()[0])
        gg += coord_fixed(ratio=1, xlim=(lower, upper), ylim=(lower, upper))
        _ = gg._draw_using_figure(fig, [ax])
        fig.set_size_inches(fig_width, fig_width)
    elif aspect_ratio == 'bank_to_45':
        slope, _ = np.polyfit(data[column1], data[column2], deg=1)
        fig.set_size_inches(fig_width, fig_width / slope)
    else:
        fig.set_size_inches(fig_width, fig_height)
        
    ax.set_xlabel(column1)
    ax.set_ylabel(column2)
    
    plt.show()

def discrete_discrete_bivariate_eda(data, column1, column2, fig_width=12, fig_height=6, plot='auto', 
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
    
    if plot == 'auto':
        plot = 'clustered_bar'
        
    if plot in ['faceted_bar', 'freqpoly']:
        flip_axis = False
        
    data[column1] = order_categorical(data, column1, None, level_order1, top_n, flip_axis)
    data[column2] = order_categorical(data, column2, None, level_order2, top_n, flip_axis)
        
    if plot == 'clustered_bar':
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
              labs(y = f"Percent (normalized within {column1} levels)")
            )
        else:
            gg = (
              ggplot(data, aes(x=column1, fill=column2)) + 
              geom_bar(position='dodge')
            )
            
    elif plot == 'faceted_bar':
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
              labs(y = f"Percent (normalized within {column2} levels)")
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
    
def discrete_continuous_bivariate_eda(data, column1, column2, fig_width=12, fig_height=6, 
                                      plot='auto', level_order='auto', flip_axis=False, top_n=20, 
                                      alpha=.6):
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
    data[column1] = order_categorical(data, column1, column2, level_order, top_n, flip_axis)
    
    if plot == 'auto':
        if len(data[column2].unique()) == len(data[column1].unique()):
            plot = 'bar' 
        elif len(data[column1].unique()) == 2:
            plot = 'histogram' 
        else:
            plot = 'boxplot'
    
    if plot == 'histogram':
        gg = (
          ggplot(data, aes(fill=column1, x=column2)) + 
          geom_histogram(alpha=alpha, position='identity')
        )
    elif plot == 'density':
        gg = (
          ggplot(data, aes(fill=column1, x=column2)) + 
          geom_density(alpha=alpha)
        )
    elif plot == 'boxplot':
        gg = (
          ggplot(data, aes(x=column1, y=column2)) + 
          geom_boxplot(varwidth=True, fill='steelblue')
        )
    elif plot == 'freqpoly':
        gg = (
          ggplot(data, aes(color=column1, x=column2)) + 
          geom_freqpoly()
        )
    elif plot == 'violin':
        gg = (
          ggplot(data, aes(x=column1, y=column2)) + 
          geom_violin(fill='steelblue', draw_quantiles=[.25, .5, .75])
        )
    elif plot == 'faceted_histogram':
        gg = (
          ggplot(data, aes(x=column2)) + 
          geom_histogram(fill='steelblue', color='black') + 
          facet_grid(f"{column1} ~ .")
        )
    elif plot == 'faceted_density':
        gg = (
          ggplot(data, aes(x=column2)) + 
          geom_density(fill='steelblue') + 
          facet_grid(f"{column1} ~ .")
        )
        
    f = gg.draw()
    f.set_size_inches(fig_width, fig_height)
    
def bivariate_eda_interact(data):
    sns.set_style('whitegrid')
    theme_set(theme_bw())
    warnings.simplefilter("ignore")
    
    widget = interactive(
        column_bivariate_eda_interact, 
        data=fixed(data), 
        column1=data.columns,
        column2=data.columns,
        discrete_limit=(10, 100, 1)
    ) 
    
    widget.layout = Layout(flex_flow='row wrap')
    for ch in widget.children:
        if hasattr(ch, 'description') and ch.description in DESCRIPTIONS:
            #ch.layout = Layout(description_width=DESCRIPTIONS[ch.description][1])
            ch.style = {'description_width': DESCRIPTIONS[ch.description][1]}
            ch.description = DESCRIPTIONS[ch.description][0]
    display(widget)
    return widget

def column_bivariate_eda_interact(data, column1, column2, discrete_limit=20):
    discrete_types = ['discrete_numeric', 'unordered_categorical', 'ordered_categorical',
                      'unordered_categorical (inferred from object)']
    
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
    
    col1_type = categorize_column_type(data[column1], discrete_limit)
    col2_type = categorize_column_type(data[column2], discrete_limit)
    print(f"Detected Column1 Type: {col1_type}")
    print(f"Detected Column2 Type: {col2_type}")
    
    # continuous-continuous
    if col1_type == 'continuous_numeric' and col2_type == 'continuous_numeric':
        print("Calling continuous_continuous_bivariate_eda:")
        widget = interactive(
            continuous_continuous_bivariate_eda,
            data=fixed(data),
            column1=fixed(column1),
            column2=fixed(column2),
            plot_type=['auto', 'scatter', 'bin2d', 'count'],
            trend_line=['auto', 'none', 'loess', 'lowess', 'glm', 'lm'],
            alpha=(0, 1, .05),
            transform1=transforms,
            transform2=transforms,
            lower_cutoff1=cutoff_range,
            upper_cutoff1=cutoff_range,
            lower_cutoff2=cutoff_range,
            upper_cutoff2=cutoff_range,
            aspect_ratio=['manual', 'bank_to_45', 'equal_axes']
        )
    # discrete-continuous
    # only one continuous value = dot/bar plot
    # distributions: boxplot, violinplot, histograms (2), freqpoly, faceted histograms
    elif (col1_type in discrete_types and col2_type == 'continuous_numeric') or \
         (col2_type in discrete_types and col1_type == 'continuous_numeric'):
        if col1_type == 'continuous_numeric':
            column1, column2 = column2, column1
        
        print(column1, column2)
        print(len(data[column1].unique()))
        if len(data[column2].unique()) == len(data[column1].unique()):
            plots = ['auto', 'bar', 'dot']
        elif len(data[column1].unique()) == 2:
            plots = ['auto', 'histogram', 'density']
        else:
            plots = ['auto', 'freqpoly', 'boxplot', 'violin', 'faceted_histogram', 'faceted_density']
        # check for singleton value
        widget = interactive(
            discrete_continuous_bivariate_eda,
            data=fixed(data),
            column1=fixed(column1),
            column2=fixed(column2),
            plot=plots,
            level_order=level_orders,
            top_n=top_n_range,
            alpha=(0, 1, .05)
        )
    elif col1_type in discrete_types and col2_type in discrete_types:
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
    else:
        print("No EDA support for these variable types")
        return 
    
    widget.layout = Layout(flex_flow='row wrap')
    for ch in widget.children:
        if hasattr(ch, 'description') and ch.description in DESCRIPTIONS:
            ch.style = {'description_width': DESCRIPTIONS[ch.description][1]}
            ch.description = DESCRIPTIONS[ch.description][0]
    display(widget)
    return widget
    