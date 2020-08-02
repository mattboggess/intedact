import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from ipywidgets import interactive, fixed, Layout
from collections import Counter
from itertools import combinations
import matplotlib.ticker as mtick 
from plotnine import *
from matplotlib import gridspec
import warnings

def _eda_countplot(data, column, fig, ax, flip_axis=False, percent=True, label_counts=True):
    """ 
    Creates a countplot for a categorical variable column. 
    
    Parameters
    ----------
    data: pandas.DataFrame
        Dataset to perform EDA on 
    column: str
        A string matching a column in the data 
    fig: plotnine ggplot
        plotnine ggplot figure object that plot is being drawn onto
    ax: matplotlib Axes
        Axes object to draw the plot onto. 
    flip_axis: bool, optional
        Whether to flip the countplot so labels are on y axis. Useful for long level names
        or lots of levels.
    percent: bool, optional
        Whether to include a twin axis denoting the relative percentages in addition to counts
        
    Returns 
    -------
    matplotlib Axes 
        Returns the Axes object with the plot drawn onto it. 
    """
    def combine(counts, percentages):
        fmt = '{} ({:.1f}%)'.format
        return [fmt(c, p) for c, p in zip(counts, percentages)]
    
    p1 = (
      ggplot(data, aes(x = column)) +
      geom_bar(fill='steelblue', color='black')
    )
    
    if flip_axis:
        p1 += coord_flip()
        va = 'center'
        ha = 'right'
    else:
        va = 'bottom'
        ha = 'center'
    
    if data[column].nunique() > 10:
        size = 8
    else:
        size = 11
    
    if label_counts:
      p1 += geom_text(
        aes(label='stat(combine(count, 100*prop))', group=1),
        stat='count',
        va=va,
        ha=ha,
        color='black',
        size=size
      )
        
    _ = p1._draw_using_figure(fig, [ax])
            
    if percent:
        if flip_axis:
            ax_perc = ax.twiny()
            ax_perc.set_xticks(100 * ax.get_xticks() / len(data[column]))
            ax_perc.set_xlim((100.0 * (float(ax.get_xlim()[0]) / len(data[column])),
                              100.0 * (float(ax.get_xlim()[1]) / len(data[column]))))
            ax_perc.xaxis.set_major_formatter(mtick.PercentFormatter())
            ax_perc.grid(None)
        else:
            ax_perc = ax.twinx()
            ax_perc.set_yticks(100 * ax.get_yticks() / len(data[column]))
            ax_perc.set_ylim((100.0 * (float(ax.get_ylim()[0]) / len(data[column])),
                              100.0 * (float(ax.get_ylim()[1]) / len(data[column]))))
            ax_perc.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax_perc.grid(None)
        
    return ax

def _eda_histogram(data, column, fig, ax, hist_bins=None, kde=False, transform='identity', 
                   lower_cutoff=0, upper_cutoff=1):
    """ 
    Creates a histogram for a numeric variable column. 
    
    Parameters
    ----------
    data: pandas.DataFrame
        Dataset to perform EDA on 
    column: str
        A string matching a column in the data 
    hist_bins: int, optional 
        Number of bins to use for the histogram. Default is automatically infer a reasonable value. 
    kde: bool, optional 
        Whether to plot a gaussian kernel density estimate. Will also normalize histogram.
    transform: ['identity', 'log', 'log_exclude0'] 
        Transformation to apply to the data for plotting:
          - 'identity': no transformation
          - 'log': apply a logarithmic transformation with small constant added in case of 0 
          - 'log_exclude0': apply a logarithmic transformation with zero removed
    lower_cutoff: float, optional [0, 1]
        Lower quantile of data to remove before plotting for ignoring outliers
    upper_cutoff: float, optional [0, 1]
        Upper quantile of data to remove before plotting for ignoring outliers
    ax: matplotlib Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes. 
        
    Returns 
    -------
    matplotlib Axes 
        Returns the Axes object with the plot drawn onto it. 
    """
    # cut out upper and lower percentiles in case of outliers
    lq = data[column].quantile(lower_cutoff)
    hq = data[column].quantile(upper_cutoff)
    data = data.query(f"{column} >= {lq} and {column} <= {hq}")
    
    # modify data for transforms 
    if transform == 'log_exclude0':
        data = data[data[column] > 0]
    elif transform == 'log':
        data[column] = data[column] + 1e-6
    
    if kde:
        p1 = (
         ggplot(data, aes(x=column, y='..density..')) +
         geom_histogram(bins=hist_bins, color='black', fill='steelblue') + 
         geom_density()
        )
    else:
        p1 = (
         ggplot(data, aes(x=column)) +
         geom_histogram(bins=hist_bins, color='black', fill='steelblue')
        ) 
    
    if transform in ['log', 'log_exclude0']:
        p1 += scale_x_log10()
        
    _ = p1._draw_using_figure(fig, [ax])
    
    return ax

def _eda_boxplot(data, column, fig, ax, flip_axis=True, transform='identity', lower_cutoff=0, 
                 upper_cutoff=1):
    """ 
    Creates a boxplot for a numeric variable.
    
    Parameters
    ----------
    data: pandas.DataFrame
        Dataset to perform EDA on 
    column: str
        A string matching a column in the data 
    flip_axis: bool, optional
        Whether to flip the boxplot orientation to horizontal.
    transform: ['identity', 'log', 'log_exclude0'] 
        Transformation to apply to the data for plotting:
          - 'identity': no transformation
          - 'log': apply a logarithmic transformation with small constant added in case of 0 
          - 'log_exclude0': apply a logarithmic transformation with zero removed
    lower_cutoff: float, optional [0, 1]
        Lower quantile of data to remove before plotting for ignoring outliers
    upper_cutoff: float, optional [0, 1]
        Upper quantile of data to remove before plotting for ignoring outliers
    ax: matplotlib Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes. 
        
    Returns 
    -------
    matplotlib Axes 
        Returns the Axes object with the plot drawn onto it. 
    """
    # cut out upper and lower percentiles in case of outliers
    lq = data[column].quantile(lower_cutoff)
    hq = data[column].quantile(upper_cutoff)
    data = data.query(f"{column} >= {lq} and {column} <= {hq}")
    
    # modify data for transforms 
    if transform == 'log_exclude0':
        data = data[data[column] > 0]
    elif transform == 'log':
        data[column] = data[column] + 1e-6
    
    p1 = (
     ggplot(data, aes(x=[''], y=column)) +
     geom_boxplot(fill='steelblue')
    )
    
    if transform in ['log', 'log_exclude0']:
        if flip_axis:
            p1 += scale_y_log10()
        else:
            p1 += scale_x_log10()
        
    if flip_axis:
        p1 += coord_flip()
        
    _ = p1._draw_using_figure(fig, [ax])
        
    return ax

def categorical_univariate_eda(data, column, fig_height=6, fig_width=12, label_counts=True,
                               flip_axis=False, level_order='Default', top_n=30):
    """ 
    Creates a univariate EDA summary for a provided categorical/low dimensional column in a 
    pandas DataFrame.
        
    For the provided column produces: 
     - a countplot with optional twin axis for percentage 
     - frequency table with count and percentage
    
    The provided column should be a category type or a discrete variable that only takes on a small 
    number of values. 
    
    Parameters
    ----------
    data: pandas.DataFrame
        Dataset to perform EDA on 
    column: str
        A string matching a column in the data 
    fig_height: int, optional
        Height of the plot 
    fig_width: int, optional
        Width of the plot 
    flip_axis: bool, optional
        Whether to flip the countplot so labels are on y axis. Useful for long level names
        or lots of levels.
    level_order: str, optional ('Default', 'Descending', 'Ascending', 'Sorted', or 'Random')
        in which to order the levels for the countplot and table. 
         - 'Default' sorts ordinal variables by provided ordering, nominal variables by 
            descending frequency, and numeric variables in sorted order.
         - 'Descending' sorts in descending frequency.
         - 'Ascending' sorts in ascending frequency.
         - 'Sorted' sorts according to sorted order of the levels themselves.
         - 'Random' produces a random order. Useful if there are too many levels for one plot. 
    top_n: int, optional 
        Maximum number of levels to attempt to plot on a single plot. If exceeded, only the 
        top_n - 1 levels will be plotted individually and the remainder will be grouped into an 
        'Other' category. 

    Returns 
    -------
    None
        No return value. Directly displays the resulting matplotlib figure and table.
    """
    data = data.copy()
    
    # handle missing data
    num_missing = data[column].isnull().sum()
    perc_missing = num_missing / data.shape[0]
    data.dropna(subset=[column], inplace=True)
    
    # determine order to plot levels
    value_counts = data[column].value_counts()
    if level_order == 'Default':
        if data[column].dtype.name == 'category':
            if data[column].cat.ordered:
                order = list(data[column].cat.categories)
            else:
                order = list(value_counts.sort_values(ascending=False).index)
        else:
            order = sorted(list(value_counts.index))
    elif level_order == 'Ascending':
        order = list(value_counts.sort_values(ascending=True).index)
    elif level_order == 'Descending':
        order = list(value_counts.sort_values(ascending=False).index)
    elif level_order == 'Sorted':
        order = sorted(list(value_counts.index))
    elif level_order == 'Random':
        order = list(value_counts.sample(frac=1).index)
    else:
        raise ValueError(f"Unknown level order specification: {level_order}")
        
    # restrict to top_n levels (condense rest into Other)
    num_levels = len(data[column].unique())
    if num_levels > top_n:
        print()
        print(f"{num_levels - top_n} levels condensed into 'Other'")
        other_levels = order[top_n:]
        order = order[:top_n] + ['Other']
        if data[column].dtype.name == 'category':
            data[column].cat.add_categories(['Other'], inplace=True)
        data.loc[data[column].isin(other_levels), column] = 'Other'
    
    # convert to ordered categorical variable
    data[column] = pd.Categorical(data[column], categories=order)
        
    # reverse order when flipping axis so order is as expected
    if flip_axis:
        order = order[::-1]
        data[column] = pd.Categorical(data[column], categories=order)
        
    # make countplot of data (empty figure hack for plotting with subplots)
    # https://github.com/has2k1/plotnine/issues/373
    fig = (ggplot() + geom_blank(data=data) + theme_void()).draw()
    fig.set_size_inches(fig_width, fig_height)
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0])
    
    _eda_countplot(data, column, fig=fig, ax=ax, flip_axis=flip_axis, label_counts=label_counts,
                   percent=True)
    
    ax.set_title((f"{data[column].size} observations over {num_levels} levels\n"
                  f"{num_missing} missing observations ({perc_missing}%)"))
    
    # show the output
    plt.show()

def numeric_univariate_eda(data, column, fig_height=6, fig_width=12, hist_bins=0, 
                           kde=False, transform='identity', lower_cutoff=0, upper_cutoff=1):
    """ 
    Creates a univariate EDA plot and table for a provided numeric variable 
    column in a pandas DataFrame.
        
    For the provided column produces: 
     - a histogram of the data
     - a boxplot of the data
     - a table with summary statistics including measures of center, spread, and quantiles.
     
    The provided column should be a high dimensional numeric type. Missing values are ignored.
    
    Parameters
    ----------
    data: pandas.DataFrame
        Dataset to perform EDA on 
    column: str
        A string matching a column in the data 
    fig_height: int, optional
        Height of the plot 
    fig_width: int, optional
        Width of the plot 
    hist_bins: int, optional (Default is 0 which translates to automatically determined bins)
        Number of bins to use for the histogram 
    kde: bool, optional 
        Whether to overlay a KDE plot 
    transform: str, ['identity', 'log', 'log_exclude0'] 
        Transformation to apply to the data for plotting:
          - 'identity': no transformation
          - 'log': apply a logarithmic transformation with small constant added in case of 0 
          - 'log_exclude0': apply a logarithmic transformation with zero removed
    lower_cutoff: float, optional [0, 1]
        Lower quantile of data to remove before plotting for ignoring outliers
    upper_cutoff: float, optional [0, 1]
        Upper quantile of data to remove before plotting for ignoring outliers

    Returns 
    -------
    None
        No return value. Directly displays the resulting matplotlib figure and table.
    """
    table = data[column].describe()
    display(pd.DataFrame(table).T)
    
    # handle missing data
    num_missing = data[column].isnull().sum()
    perc_missing = num_missing / data.shape[0]
    data.dropna(subset=[column], inplace=True)
    
    if hist_bins == 0:
        hist_bins = None
        
    
        
    # make histogram and boxplot figure (empty figure hack for plotting with subplots)
    # https://github.com/has2k1/plotnine/issues/373
    fig = (ggplot() + geom_blank(data=data) + theme_void()).draw()
    fig.set_size_inches(fig_width, fig_height * 2)
    gs = gridspec.GridSpec(2, 1)
    ax_hist = fig.add_subplot(gs[0])
    ax_box = fig.add_subplot(gs[1])
    
    _eda_histogram(data, column, fig=fig, ax=ax_hist, kde=kde, hist_bins=hist_bins, 
                   transform=transform)
    _eda_boxplot(data, column, fig=fig, ax=ax_box, transform=transform)
    
    ax_hist.set_title(
        (f"{data[column].size} observations ranging from {data[column].min()} to {data[column].max()}\n"
         f"{num_missing} missing observations ({perc_missing}%)"))
    
    plt.show()
    
def datetime_univariate_eda(data, column, fig_height=6, fig_width=12, ts_freq='1M', delta_freq='1D',
                            hist_bins=0, kde=False, transform='identity', lower_cutoff=0, 
                            upper_cutoff=1):
    """ 
    Creates a univariate EDA plot for a provided datetime variable column in a pandas DataFrame.
        
    For the provided column produces:
      - a timeseries plot of counts aggregated at the temporal resolution provided by ts_freq 
      - a histogram of time deltas between successive observations in units defined by delta_freq 
      - a boxplot of time deltas between successive observations in units defined by delta_freq 
      - countplots for the following metadata from the datetime object:
        - day of week
        - day of month
        - month
        - year
        - hour
        - minute
      - title with # of observations and min and max timestamps
    
    Parameters
    ----------
    data: pandas.DataFrame
        Dataset to perform EDA on 
    column: str
        A string matching a column in the data 
    fig_height: int, optional
        Height of the plot 
    fig_width: int, optional
        Width of the plot 
    ts_freq: str, optional (Default is '1M' = 1 month) 
        pandas offset string that denotes the frequency at which to resample for the time
        series plot. See the following link for options:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    delta_freq: str, optional (Default is '1D' = 1 day) 
        pandas offset string that denotes the units at which to compute time deltas. 
        See the following link for options:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    hist_bins: int, optional (Default is 0 which translates to automatically determined bins)
        Number of bins to use for the time delta histogram 
    kde: bool, optional 
        Whether to overlay a KDE plot for the time deltas
    transform: str, ['identity', 'log', 'log_exclude0'] 
        Transformation to apply to the time deltas for plotting:
          - 'identity': no transformation
          - 'log': apply a logarithmic transformation with small constant added in case of 0 
          - 'log_exclude0': apply a logarithmic transformation with zero removed
    lower_cutoff: float, optional [0, 1]
        Lower quantile of data to remove before plotting time deltas for ignoring outliers
    upper_cutoff: float, optional [0, 1]
        Upper quantile of data to remove before plotting time deltas for ignoring outliers

    Returns 
    -------
    None
        No return value. Directly displays the resulting matplotlib figure and table.
    """
    data = data.copy()
    
    # handle missing data
    num_missing = data[column].isnull().sum()
    perc_missing = num_missing / data.shape[0]
    data.dropna(subset=[column], inplace=True)
    
    if hist_bins == 0:
        hist_bins = None
    
    # make histogram and boxplot figure (empty figure hack for plotting with subplots)
    # https://github.com/has2k1/plotnine/issues/373
    fig = (ggplot() + geom_blank(data=data) + theme_void()).draw()
    fig.set_size_inches(fig_width, fig_height * 6)
    grid = fig.add_gridspec(6, 2)
    
    # compute extra columns with datetime attributes
    data['month'] = data[column].dt.month_name()
    data['day of month'] = data[column].dt.day
    data['year'] = data[column].dt.year
    data['hour'] = data[column].dt.hour
    data['minute'] = data[column].dt.minute
    data['second'] = data[column].dt.second
    data['day of week'] = data[column].dt.day_name()
    
    # compute time deltas
    dts = data[column].sort_values(ascending=True)
    data['deltas'] = (dts - dts.shift(1)) / pd.Timedelta(delta_freq) 
    
    # time series count plot
    ax = fig.add_subplot(grid[0, :])
    tmp = data \
        .set_index(column) \
        .resample(ts_freq) \
        .agg('size')
    
    ax = tmp.plot(ax=ax)
    ax_perc = ax.twinx()
    ax_perc.set_yticks(100 * ax.get_yticks() / tmp.sum())
    ax_perc.set_ylim((100.0 * (float(ax.get_ylim()[0]) / tmp.sum())), 
                      100.0 * (float(ax.get_ylim()[1]) / tmp.sum()))
    ax_perc.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_perc.grid(None)
    ax.set_xlabel(f"Time series of observation counts resampled at {ts_freq}")
    ax.set_ylabel('count')
    plt.title((
        f"{data[column].size} observations ranging from {data[column].min()} to {data[column].max()}\n"
        f"{num_missing} missing observations ({perc_missing}%)"))
    
    # histogram and boxplot of time deltas
    ax_hist = _eda_histogram(data, 'deltas', fig=fig, ax=fig.add_subplot(grid[1, 0]), kde=kde, 
                            hist_bins=hist_bins, transform=transform, lower_cutoff=lower_cutoff,
                            upper_cutoff=upper_cutoff)
    ax_hist.set_xlabel(f"Time deltas between observations in units of {delta_freq}")
    ax_box = _eda_boxplot(data, 'deltas', fig=fig, ax=fig.add_subplot(grid[1, 1]), flip_axis=True, 
                         lower_cutoff=lower_cutoff, upper_cutoff=upper_cutoff, transform=transform)
    ax_box.set_xlabel(f"Time deltas between observations in units of {delta_freq}")
    
    # plot countplot by year 
    year_order = np.arange(data['year'].min(), data['year'].max() + 1, 1)
    data['year'] = pd.Categorical(data['year'], categories=year_order)
    ax = _eda_countplot(data, 'year', fig=fig, flip_axis=False, 
                       ax=fig.add_subplot(grid[2, :]))
    for tick in ax.get_xticklabels():
        tick.set_rotation(60)
    
    # plot countplot by month
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                   'September', 'October', 'November', 'December'][::-1]
    data['month'] = pd.Categorical(data['month'], categories=month_order)
    _eda_countplot(data, 'month', fig=fig, flip_axis=True, ax=fig.add_subplot(grid[3, 0]))
    
    # plot countplot by day of month
    day_month_order = np.arange(1, 32)[::-1]
    ax=fig.add_subplot(grid[3, 1])
    data['day of month'] = pd.Categorical(data['day of month'], categories=day_month_order)
    _eda_countplot(data, 'day of month', fig=fig, flip_axis=True, ax=ax) 
    
    # plot countplot by day of week
    day_week_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    data['day of week'] = pd.Categorical(data['day of week'], categories=day_week_order)
    eda_countplot(data, 'day of week', fig=fig, flip_axis=True, ax=fig.add_subplot(grid[4, 0]))
    
    ## plot countplot by minute of hour 
    #minute_order = np.arange(0, 60)
    #ax = eda_countplot(data, 'minute', order=minute_order, flip_axis=True, 
    #                   ax=fig.add_subplot(grid[4, 1]), percent=False)
    #ax.set_yticks(np.arange(0, 65, 5))
    #ax.set_yticklabels(np.arange(0, 65, 5))
    #
    ## plot countplot by hour of day
    #hour_order = np.arange(0, 24)
    #eda_countplot(data, 'hour',  order=hour_order, flip_axis=True, ax=fig.add_subplot(grid[5, 0]))
    #
    ## plot countplot by second of hour 
    #second_order = np.arange(0, 60)
    #ax = eda_countplot(data, 'second', order=second_order, flip_axis=True, 
    #                   ax=fig.add_subplot(grid[5, 1]), percent=False)
    #ax.set_yticks(np.arange(0, 65, 5))
    #ax.set_yticklabels(np.arange(0, 65, 5))
        
    
    plt.show()
    
def text_univariate_eda(data, column, fig_height=6, fig_width=12, top_n=10, remove_punct=True,
                        remove_stop=True, lower_case=True, kde=False, transform='identity',
                        hist_bins=0, lower_cutoff=0, upper_cutoff=1):
    """ 
    Creates a univariate EDA summary for a provided text variable column in a pandas DataFrame.
        
    For the provided column produces:
      - histograms of token and character counts across entries
      - boxplot of document frequencies
      - countplots with top_n unigrams, bigrams, and trigrams
      - title with total # of tokens, vocab size, and corpus size 
    
    Parameters
    ----------
    data: pandas.DataFrame
        Dataset to perform EDA on 
    column: str
        A string matching a column in the data 
    fig_height: int, optional
        Height of the plot 
    fig_width: int, optional
        Width of the plot 
    top_n: int, optional 
        Maximum number to plot for the top most frequent unigrams, bigrams, and trigrams
    remove_stop: bool, optional
        Whether to remove stop words from consideration as tokens
    remove_punct: bool, optional
        Whether to remove punctuation from consideration as tokens
    lower_case: bool, optional
        Whether to lower case text when forming tokens
    hist_bins: int, optional (Default is 0 which translates to automatically determined bins)
        Number of bins to use for the histograms
    kde: bool, optional 
        Whether to overlay a KDE plot for the histograms
    transform: str, ['identity', 'log', 'log_exclude0'] 
        Transformation to apply to the histogram/boxplot variables for plotting:
          - 'identity': no transformation
          - 'log': apply a logarithmic transformation with small constant added in case of 0 
          - 'log_exclude0': apply a logarithmic transformation with zero removed
    lower_cutoff: float, optional [0, 1]
        Lower quantile of data to remove before plotting histogram/boxplots for ignoring outliers
    upper_cutoff: float, optional [0, 1]
        Upper quantile of data to remove before plotting histograms/boxplots for ignoring outliers

    Returns 
    -------
    None
        No return value. Directly displays the resulting matplotlib figure and table.
    """
    from nltk import word_tokenize, ngrams
    from nltk.corpus import stopwords
    
    if hist_bins == 0:
        hist_bins = None
    
    data.dropna(subset=[column], inplace=True)
    data['characters_per_document'] = data[column].apply(lambda x: len(x))
    
    # tokenize and compute number of tokens 
    if lower_case:
        data['tokens'] = data[column].apply(lambda x: [w.lower() for w in word_tokenize(x)])
    else:
        data['tokens'] = data[column].apply(lambda x: [w for w in word_tokenize(x)])
    
    if remove_stop:
        stop_words = set(stopwords.words('english'))
        data['tokens'] = data['tokens'].apply(lambda x: [w for w in x if w.lower() not in stop_words])
        
    if remove_punct:
        data['tokens'] = data['tokens'].apply(lambda x: [w for w in x if w.isalnum()])
        
    data['tokens_per_document'] = data['tokens'].apply(lambda x: len(x))
    
    f, axs = plt.subplots(3, 2, figsize=(fig_width, fig_height * 3))
    
    # plot most frequent unigrams 
    unigrams = pd.DataFrame({'Most Common Tokens': [x for y in data['tokens'] for x in y]})
    order = list(unigrams['Most Common Tokens'].value_counts().sort_values(ascending=False).index)
    eda_countplot(unigrams, 'Most Common Tokens', ax=axs[0, 0], order=order[:top_n], 
                  flip_axis=True)
    
    # plot most frequent bigrams
    bigrams = pd.DataFrame({'Most Common Bigrams': [x for y in data['tokens'] for x in ngrams(y, 2)]})
    order = list(bigrams['Most Common Bigrams'].value_counts().sort_values(ascending=False).index)
    eda_countplot(bigrams, 'Most Common Bigrams', ax=axs[1, 0], order=order[:top_n], 
                  flip_axis=True)
    
    # plot most frequent trigams
    trigrams = pd.DataFrame({'Most Common Trigams': [x for y in data['tokens'] for x in ngrams(y, 3)]})
    order = list(trigrams['Most Common Trigams'].value_counts().sort_values(ascending=False).index)
    eda_countplot(trigrams, 'Most Common Trigams', ax=axs[2, 0], order=order[:top_n], 
                  flip_axis=True)
    
    # histograms of token and character counts per document
    eda_histogram(data, 'tokens_per_document', ax=axs[0, 1], kde=kde, hist_bins=hist_bins, 
                  transform=transform, lower_cutoff=lower_cutoff, upper_cutoff=upper_cutoff)
    eda_histogram(data, 'characters_per_document', ax=axs[1, 1], kde=kde, hist_bins=hist_bins,
                  transform=transform, lower_cutoff=lower_cutoff, upper_cutoff=upper_cutoff)
    tmp = pd.DataFrame({'observations_per_document': list(data[column].value_counts())})
    eda_boxplot(tmp, 'observations_per_document', flip_axis=True, transform=transform,
                lower_cutoff=lower_cutoff, upper_cutoff=upper_cutoff, ax=axs[2, 1])
    
    tokens = list(unigrams['Most Common Tokens'])
    num_tokens = len(tokens) 
    vocab_size = len(set(tokens)) 
    corpus_size = data[column].size
    plt.suptitle(f"{num_tokens} tokens with a vocabulary size of {vocab_size} in a corpus of {corpus_size} documents")
    
    plt.subplots_adjust(top=.95)
    plt.show()
    
def sequence_univariate_eda(data, column, fig_height=6, fig_width=12, top_n=10):
    """ 
    Creates a univariate EDA summary for a provided sequence column in a pandas DataFrame.
        
    For the provided column produces: 
     - a countplot with the number of entries per sequence across observations 
     - countplots with the top_n most frequent single entries, pairs of entries, and triplets of 
       entries across observations
    
    The provided column should be an object type containing lists, tuples, or sets. 
    
    Parameters
    ----------
    data: pandas.DataFrame
        Dataset to perform EDA on 
    column: str
        A string matching a column in the data 
    fig_height: int, optional
        Height of the plot 
    fig_width: int, optional
        Width of the plot 
    top_n: int, optional 
        Maximum number of entries to plot for the top most frequent single entries, pairs, 
        and triplets.

    Returns 
    -------
    None
        No return value. Directly displays the resulting matplotlib figure.
    """
    data = data.copy()
    data.dropna(subset=[column], inplace=True)
    f, axs = plt.subplots(4, 1, figsize=(fig_width, fig_height * 4))
    
    # plot number of elements
    data['# Entries per Observation'] = data[column].apply(lambda x: len(x)) 
    eda_countplot(data, '# Entries per Observation', ax=axs[0], flip_axis=False,
                  order=sorted(data['# Entries per Observation'].unique()))
    
    # compute most common entries
    entries = [i for e in data[column] for i in e]
    singletons = pd.DataFrame({'Most Common Entries': entries})
    order = list(singletons['Most Common Entries'].value_counts().sort_values(ascending=False).index)
    eda_countplot(singletons, 'Most Common Entries', order=order[:top_n], ax=axs[1], flip_axis=True)
    
    # compute most common pairs
    pairs = [comb for coll in data[column] for comb in combinations(coll, 2)]
    pairs = pd.DataFrame({'Most Common Entry Pairs': pairs})
    order = list(pairs['Most Common Entry Pairs'].value_counts().sort_values(ascending=False).index)
    eda_countplot(pairs, 'Most Common Entry Pairs', order=order[:top_n], ax=axs[2], flip_axis=True)
    
    # comput most common triples
    triples = [comb for coll in data[column] for comb in combinations(coll, 3)]
    triples = pd.DataFrame({'Most Common Entry Triples': triples})
    order = list(triples['Most Common Entry Triples'].value_counts().sort_values(ascending=False).index)
    eda_countplot(triples, 'Most Common Entry Triples', order=order[:top_n], ax=axs[3], 
                  flip_axis=True)
    
    axs[0].set_title(f"{len(set(entries))} unique entries with {len(entries)} total entries across {data[column].size} observations")
    plt.show()

def univariate_eda_interact(data):
    sns.set_style('whitegrid')
    theme_set(theme_bw())
    warnings.simplefilter("ignore")
    
    widget = interactive(
        column_univariate_eda_interact, 
        data=fixed(data), 
        column=data.columns,
        cat_cutoff=(10, 100, 1)
    ) 
    widget.layout = Layout(flex_flow='row wrap')
    display(widget)
    
def column_univariate_eda_interact(data, column, cat_cutoff=30):
    from pandas.api.types import is_numeric_dtype
    from pandas.api.types import is_datetime64_any_dtype
    
    # ranges for widgets
    fig_height_range = (1, 30, 1)
    fig_width_range = (1, 30, 1)
    hist_bin_range = (0, 50, 1)
    cutoff_range = (0, 1, .01)
    level_orders = ['Default', 'Ascending', 'Descending', 'Sorted', 'Random']
    kde_default = False
    flip_axis_default = False
    transforms = ['identity', 'log', 'log_exclude0']
    top_n_range = (5, 100, 1)
    
    # datetime variables 
    if is_datetime64_any_dtype(data[column]):
        print(("Datetime variable detected. Depending on your data size and the frequency chosen, "
               "this may take a little bit to load."))
        print("See here for valid frequency strings: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects")
        widget = interactive(
            datetime_univariate_eda, 
            data=fixed(data), 
            column=fixed(column),
            fig_height=fig_height_range,
            fig_width=fig_width_range,
            hist_bins=hist_bin_range,
            lower_cutoff=cutoff_range,
            upper_cutoff=cutoff_range,
            transform=transforms,
            kde=kde_default
        )
    # numeric variables
    elif is_numeric_dtype(data[column]):
        if len(data[column].unique()) > cat_cutoff:
            print("High-dimensional numeric variable detected")
            widget = interactive(
                numeric_univariate_eda, 
                data=fixed(data), 
                column=fixed(column),
                fig_height=fig_height_range,
                fig_width=fig_width_range,
                hist_bins=hist_bin_range,
                lower_cutoff=cutoff_range,
                upper_cutoff=cutoff_range,
                transform=transforms,
                kde=kde_default
             ) 
        else:
            print("Low-dimensional numeric variable detected")
            widget = interactive(
                categorical_univariate_eda, 
                data=fixed(data), 
                column=fixed(column),
                fig_height=fig_height_range,
                fig_width=fig_width_range,
                flip_axis=flip_axis_default,
                level_order=level_orders,
                top_n=top_n_range
             ) 
    # categorical variables
    elif data[column].dtype.name == 'category':
        if data[column].cat.ordered:
            print("Ordered categorical variable detected")
        else:
            print("Unordered categorical variable detected")
        widget = interactive(
            categorical_univariate_eda, 
            data=fixed(data), 
            column=fixed(column),
            fig_height=fig_height_range,
            fig_width=fig_width_range,
            flip_axis=flip_axis_default,
            level_order=level_orders,
            top_n=top_n_range
         ) 
    # text variables
    elif data[column].dtype.name == 'string':
        print(("Text variable detected. Depending on your data size, this may take a little bit "
               "to load."))
        widget = interactive(
            text_univariate_eda,
            data=fixed(data),
            column=fixed(column),
            fig_height=fig_height_range,
            fig_width=fig_width_range,
            hist_bins=hist_bin_range,
            lower_cutoff=cutoff_range,
            upper_cutoff=cutoff_range,
            transform=transforms,
            kde=kde_default,
            top_n=top_n_range
        )
    # object variables 
    elif data[column].dtype.name == 'object':
        test_value = data[column].dropna().iat[0]
        if isinstance(test_value, (list, tuple, set)):
            print("Sequence variable detected")
            widget = interactive(
                sequence_univariate_eda,
                data=fixed(data),
                column=fixed(column),
                fig_height=fig_height_range,
                fig_width=fig_width_range,
                top_n=top_n_range
            )
        elif isinstance(test_value, str):
            avg_words = data[column].dropna().apply(lambda x: len(x.split())).mean()
            if len(data[column].unique()) < cat_cutoff or avg_words <= 3:
                print("Inferring unordered categorical variable from object column")
                widget = interactive(
                    categorical_univariate_eda, 
                    data=fixed(data), 
                    column=fixed(column),
                    fig_height=fig_height_range,
                    fig_width=fig_width_range,
                    flip_axis=flip_axis_default,
                    level_order=level_orders,
                    top_n=top_n_range
                 ) 
            else:
                print(("Inferring text variable from object column. "
                       "Depending on your data size, this may take a little bit to load."))
                widget = interactive(
                    text_univariate_eda,
                    data=fixed(data),
                    column=fixed(column),
                    fig_height=fig_height_range,
                    fig_width=fig_width_range,
                    hist_bins=hist_bin_range,
                    lower_cutoff=cutoff_range,
                    upper_cutoff=cutoff_range,
                    transform=transforms,
                    kde=kde_default,
                    top_n=top_n_range
                )
            
        else:
            print("No EDA support for this variable type")
            return
    else:
        print("No EDA support for this variable type")
        return 
    
    widget.layout = Layout(flex_flow='row wrap')
    display(widget)
    
