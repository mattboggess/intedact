import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from itertools import combinations
from plotnine import *
from matplotlib import gridspec
from .utils import *
from .config import *

FLIP_LEVEL_COUNT = 10


def discrete_univariate_eda(
        data: pd.DataFrame,
        column: str,
        fig_height: int = 5,
        fig_width: int = 10,
        level_order: str = "auto",
        max_levels: int = 30,
        label_counts: bool = True,
        flip_axis: bool = None,
        rotate_labels: bool = False
    ) -> None:
    """ 
    Creates a univariate EDA summary for a provided discrete data column in a pandas DataFrame.

    Summary consists of a single bar plot with twin axes for counts and percentages for each level of the
    variable. Percentages are relative to observed data only (missing observations are ignored). A discrete data
    column refers to any categorical data column or any low cardinality numerical column.

    Args:
        data: pandas DataFrame to perform EDA on
        column: A string matching a column in the data
        fig_height: Height of the plot in inches
        fig_width: Width of the plot in inches
        level_order: Order in which to sort the levels.
         - 'auto' sorts ordinal variables by provided ordering, nominal variables by
            descending frequency, and numeric variables in sorted order.
         - 'descending' sorts in descending frequency.
         - 'ascending' sorts in ascending frequency.
         - 'sorted' sorts according to sorted order of the levels themselves.
         - 'random' produces a random order. Useful if there are too many levels for one plot.
        max_levels: Maximum number of levels to attempt to plot on a single plot. If exceeded, only the
         max_level - 1 levels will be plotted and the remainder will be grouped into an 'Other' category.
        label_counts: Whether to add exact counts and percentages as text annotations on each bar in the plot.
        flip_axis: Whether to flip the plot so labels are on y axis. Useful for long level names or lots of levels.
        rotate_labels: Whether to rotate x axis levels 90 degrees to prevent overlapping labels.

    Returns:
        No return value. Directly displays the results.
    """
    data = data.copy()
    
    # handle missing data
    num_missing = data[column].isnull().sum()
    perc_missing = num_missing / data.shape[0]
    data.dropna(subset=[column], inplace=True)
    num_levels = data[column].nunique()

    # try to intelligently determine how to flip axis if not specified
    if flip_axis is None:
        if rotate_labels:
            flip_axis = False
        else:
            flip_axis = num_levels > FLIP_LEVEL_COUNT

    # reorder column levels
    data[column] = order_levels(
        data, column, None, level_order=level_order, max_levels=max_levels, flip_axis=flip_axis
    )

    # draw the barplot
    count_data = (
        data
        .groupby(column)
        .size()
        .reset_index()
        .rename({0: 'count'}, axis='columns')
    )
    count_data['label'] = [f"{x} ({100 * x / count_data['count'].sum():.1f}%)" for x in count_data['count']]
    gg = (
        ggplot(count_data, aes(x=column, y='count')) +
        geom_col(fill=BAR_COLOR, color='black')
    )

    # flip axis
    value_counts = count_data['count']
    nudge = value_counts.max() / 100
    mid = value_counts.max() / 5 * 4
    if flip_axis:
        gg += coord_flip()
        va = ['center'] * len(value_counts)
        ha = ['right' if x > mid else 'left' for x in value_counts]
        nudge_y = [-nudge if x > mid else nudge for x in value_counts]
    else:
        va = ['top' if x > mid else 'bottom' for x in value_counts]
        ha = ['center'] * len(value_counts)
        nudge_y = [-nudge if x > mid else nudge for x in value_counts]

    # rotate labels
    if rotate_labels:
        gg += theme(axis_text_x=element_text(rotation=90, hjust=1))
    else:
        gg += theme(axis_text_x=element_text(rotation=0))

    # add count/percentage annotations
    if data[column].nunique() > 10:
        size = 8
    else:
        size = 11
    if label_counts:
        gg += geom_text(
            aes(label='label', group=1, ha=ha, va=va),
            nudge_y=nudge_y,
            color='black',
            size=size
        )

    gg.draw()
    fig = plt.gcf()
    ax = fig.axes[0]

    # add a twin axis for percentage
    add_percent_axis(ax, len(data[column]), flip_axis=flip_axis)

    # warn user about 'Other' condensing and add info to title
    if num_levels > max_levels:
        addition = f" ({num_levels - max_levels} levels condensed into 'Other')"
    else:
        addition = ""
    title = (f"{data[column].size} observations over {num_levels} levels{addition}\n"
             f"{num_missing} missing observations ({perc_missing}%)")
    ax.set_title(title)

    fig.set_size_inches(fig_width, fig_height)
    plt.show()


def continuous_univariate_eda(data, column, fig_height=4, fig_width=8, hist_bins=0,
                              transform='identity', lower_quantile=0, upper_quantile=1, kde=False):
    """ 
    Creates a univariate EDA summary for a provided continuous data column in a pandas DataFrame.

    Summary consists of a histogram, boxplot, and small table of summary statistics.

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
    transform: str, ['identity', 'log', 'log_exclude0', 'sqrt']
        Transformation to apply to the data for plotting:
          - 'identity': no transformation
          - 'log': apply a logarithmic transformation with small constant added in case of 0 
          - 'log_exclude0': apply a logarithmic transformation with zero removed
          - 'sqrt': apply a square root transformation
    lower_quantile: float, optional [0, 1]
        Lower quantile of data to remove before plotting for ignoring outliers
    upper_quantile: float, optional [0, 1]
        Upper quantile of data to remove before plotting for ignoring outliers
    kde: bool, optional
        Whether to overlay a KDE plot

    Returns 
    -------
    None
        No return value. Directly displays the resulting matplotlib figure and table.
    """
    # handle missing data
    num_missing = data[column].isnull().sum()
    perc_missing = num_missing / data.shape[0]
    data.dropna(subset=[column], inplace=True)

    # preprocess column for transforms and remove outlier quantiles
    data = preprocess_numeric_variables(data, column, lq1=lower_quantile, hq1=upper_quantile,
                                        transform1=transform)

    if hist_bins == 0:
        hist_bins = freedman_diaconis_bins(data[column], transform)

    # compute and display summary statistics
    table = pd.DataFrame(data[column].describe()).T
    table['iqr'] = data[column].quantile(.75) - data[column].quantile(.25)
    table['missing_count'] = num_missing
    table['missing_percent'] = perc_missing
    display(pd.DataFrame(table))

    # make histogram and boxplot figure (empty figure hack for plotting with subplots)
    # https://github.com/has2k1/plotnine/issues/373
    fig = (ggplot() + geom_blank(data=data) + theme_void()).draw()
    fig.set_size_inches(fig_width, fig_height * 2)
    gs = gridspec.GridSpec(2, 1)
    ax_hist = fig.add_subplot(gs[0])
    ax_box = fig.add_subplot(gs[1])

    # plot histogram
    if kde:
        ylabel = 'density'
        gg_hist = (
            ggplot(data, aes(x=column, y='..density..')) +
            geom_histogram(bins=hist_bins, color='black', fill=BAR_COLOR) +
            geom_density()
        )
    else:
        ylabel = 'count'
        gg_hist = (
            ggplot(data, aes(x=column)) +
            geom_histogram(bins=hist_bins, color='black', fill=BAR_COLOR)
        )

    # plot boxplot
    gg_box = (
        ggplot(data, aes(x=[''], y=column)) +
        geom_boxplot(fill=BAR_COLOR) +
        coord_flip()
    )

    # handle axes transforms
    if transform in ['log', 'log_exclude0']:
        gg_hist += scale_x_log10()
        gg_box += scale_y_log10()
        xlabel = f"{column} (log10 scale)"
    elif transform == 'sqrt':
        gg_hist += scale_x_sqrt()
        gg_box += scale_y_sqrt()
        xlabel = f"{column} (square root scale)"
    else:
        xlabel = column

    # draw plots to axes
    _ = gg_hist._draw_using_figure(fig, [ax_hist])
    _ = gg_box._draw_using_figure(fig, [ax_box])

    ax_hist.set_xlabel(xlabel)
    ax_hist.set_ylabel(ylabel)
    ax_box.set_xlabel(xlabel)

    plt.show()


def datetime_univariate_eda(data, column, fig_height=4, fig_width=8, ts_freq='1M', delta_freq='1D',
                            hist_bins=0, transform='identity', lower_quantile=0, upper_quantile=1, kde=False):
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
    transform: str, ['identity', 'log', 'log_exclude0']
        Transformation to apply to the time deltas for plotting:
          - 'identity': no transformation
          - 'log': apply a logarithmic transformation with small constant added in case of 0 
          - 'log_exclude0': apply a logarithmic transformation with zero removed
    lower_quantile: float, optional [0, 1]
        Lower quantile of data to remove before plotting time deltas for ignoring outliers
    upper_quantile: float, optional [0, 1]
        Upper quantile of data to remove before plotting time deltas for ignoring outliers
    kde: bool, optional
        Whether to overlay a KDE plot for the time deltas

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
    
    # make histogram and boxplot figure (empty figure hack for plotting with subplots)
    # https://github.com/has2k1/plotnine/issues/373
    fig = (ggplot() + geom_blank(data=data) + theme_void()).draw()
    fig.set_size_inches(fig_width, fig_height * 5)
    grid = fig.add_gridspec(5, 2)
    
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
    ax = tmp.plot(ax=ax, color=BAR_COLOR)
    add_percent_axis(ax, data.shape[0])
    ax.set_xlabel(f"Time series of observation counts resampled at {ts_freq}")
    ax.set_ylabel('count')
    plt.title((
        f"{data[column].size} observations ranging from {data[column].min()} to {data[column].max()}\n"
        f"{num_missing} missing observations ({perc_missing}%)"))
    
    # histogram and boxplot of time deltas
    data = preprocess_numeric_variables(data, 'deltas', lq1=lower_quantile, hq1=upper_quantile,
                                        transform1=transform)
    if hist_bins == 0:
        hist_bins = freedman_diaconis_bins(data['deltas'], transform)
    ax_hist = fig.add_subplot(grid[1, 0])
    ax_box = fig.add_subplot(grid[1, 1])

    # plot histogram
    if kde:
        ylabel = 'density'
        gg_hist = (
                ggplot(data, aes(x='deltas', y='..density..')) +
                geom_histogram(bins=hist_bins, color='black', fill=BAR_COLOR) +
                geom_density()
        )
    else:
        ylabel = 'count'
        gg_hist = (
                ggplot(data, aes(x='deltas')) +
                geom_histogram(bins=hist_bins, color='black', fill=BAR_COLOR)
        )

    # plot boxplot
    gg_box = (
            ggplot(data, aes(x=[''], y='deltas')) +
            geom_boxplot(fill=BAR_COLOR) +
            coord_flip()
    )

    # handle axes transforms
    if transform in ['log', 'log_exclude0']:
        gg_hist += scale_x_log10()
        gg_box += scale_y_log10()
    elif transform == 'sqrt':
        gg_hist += scale_x_sqrt()
        gg_box += scale_y_sqrt()

    # draw plots to axes
    _ = gg_hist._draw_using_figure(fig, [ax_hist])
    _ = gg_box._draw_using_figure(fig, [ax_box])
    ax_hist.set_xlabel(f"Time deltas between observations in units of {delta_freq}")
    ax_hist.set_ylabel('count')
    ax_box.set_xlabel(f"Time deltas between observations in units of {delta_freq}")
    
    # plot countplot by year 
    year_order = np.arange(data['year'].min(), data['year'].max() + 1, 1)
    data['year'] = pd.Categorical(data['year'], categories=year_order)
    ax_year = fig.add_subplot(grid[2, :])
    gg_year = (
        ggplot(data, aes(x='year')) +
        geom_bar(fill=BAR_COLOR, color='black') +
        theme(axis_text_x=element_text(rotation=90))
    )
    _ = gg_year._draw_using_figure(fig, [ax_year])
    ax_year.set_ylabel('count')
    add_percent_axis(ax_year, data.shape[0], flip_axis=False)

    # plot countplot by month
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                   'September', 'October', 'November', 'December']
    data['month'] = pd.Categorical(data['month'], categories=month_order)
    ax_month = fig.add_subplot(grid[3, 0])
    gg_month = (
            ggplot(data, aes(x='month')) +
            geom_bar(fill=BAR_COLOR, color='black') +
            coord_flip()
    )
    _ = gg_month._draw_using_figure(fig, [ax_month])
    ax_month.set_xlabel('count')
    add_percent_axis(ax_month, data.shape[0], flip_axis=True)

    # plot countplot by day of month
    ax_day_month = fig.add_subplot(grid[3, 1])
    gg_day_month = (
            ggplot(data, aes(x='day of month')) +
            geom_histogram(bins=31, fill=BAR_COLOR, color='black') +
            coord_flip()
    )
    _ = gg_day_month._draw_using_figure(fig, [ax_day_month])
    ax_day_month.set_xlabel('count')
    ax_day_month.set_ylabel('day of month')
    add_percent_axis(ax_day_month, data.shape[0], flip_axis=True)
    ax_day_month.set_yticks(np.arange(1, 32, 5))
    ax_day_month.set_yticklabels(np.arange(1, 32, 5))

    # plot countplot by day of week
    day_week_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    data['day of week'] = pd.Categorical(data['day of week'], categories=day_week_order)
    ax_day_week = fig.add_subplot(grid[4, 0])
    gg_day_week = (
            ggplot(data, aes(x='day of week')) +
            geom_bar(fill=BAR_COLOR, color='black') +
            coord_flip()
    )
    _ = gg_day_week._draw_using_figure(fig, [ax_day_week])
    ax_day_week.set_xlabel('count')
    add_percent_axis(ax_day_week, data.shape[0], flip_axis=True)

    # plot countplot by hour of day
    ax_hour = fig.add_subplot(grid[4, 1])
    gg_hour = (
            ggplot(data, aes(x='hour')) +
            geom_histogram(bins=24, fill=BAR_COLOR, color='black') +
            coord_flip()
    )
    _ = gg_hour._draw_using_figure(fig, [ax_hour])
    ax_hour.set_xlabel('count')
    ax_hour.set_ylabel('hour')
    ax_hour.set_yticks(np.arange(0, 24))
    ax_hour.set_yticklabels(np.arange(0, 24))
    add_percent_axis(ax_hour, data.shape[0], flip_axis=True)

    plt.tight_layout()
    plt.show()


def text_univariate_eda(data, column, fig_height=4, fig_width=8, top_ngrams=10, transform='identity',
                        hist_bins=0, lower_quantile=0, upper_quantile=1, remove_punct=True, remove_stop=True,
                        lower_case=True):
    """ 
    Creates a univariate EDA summary for a provided text variable column in a pandas DataFrame.
        
    For the provided column produces:
      - histograms of token and character counts across entries
      - boxplot of document frequencies
      - countplots with top_ngrams unigrams, bigrams, and trigrams
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
    top_ngrams: int, optional
        Maximum number of ngrams to plot for the top most frequent unigrams and bigrams
    hist_bins: int, optional (Default is 0 which translates to automatically determined bins)
        Number of bins to use for the histograms
    transform: str, ['identity', 'log', 'log_exclude0']
        Transformation to apply to the histogram/boxplot variables for plotting:
          - 'identity': no transformation
          - 'log': apply a logarithmic transformation with small constant added in case of 0 
          - 'log_exclude0': apply a logarithmic transformation with zero removed
    lower_quantile: float, optional [0, 1]
        Lower quantile of data to remove before plotting histogram/boxplots for ignoring outliers
    upper_quantile: float, optional [0, 1]
        Upper quantile of data to remove before plotting histograms/boxplots for ignoring outliers
    remove_stop: bool, optional
        Whether to remove stop words from consideration for ngrams
    remove_punct: bool, optional
        Whether to remove punctuation from consideration for ngrams
    lower_case: bool, optional
        Whether to lower case text when forming tokens for ngrams

    Returns 
    -------
    None
        No return value. Directly displays the resulting matplotlib figure and table.
    """
    from nltk import word_tokenize, ngrams
    from nltk.corpus import stopwords

    data = data.copy()
    # handle missing data
    num_missing = data[column].isnull().sum()
    perc_missing = num_missing / data.shape[0]
    data.dropna(subset=[column], inplace=True)

    # tokenize and compute number of tokens 
    data['characters_per_document'] = data[column].apply(lambda x: len(x))
    data['tokens'] = data[column].apply(lambda x: [w for w in word_tokenize(x)])
    data['tokens_per_document'] = data['tokens'].apply(lambda x: len(x))
    tokens = [x for y in data['tokens'] for x in y]

    # filters for ngram computations
    if lower_case:
        data['tokens'] = data['tokens'].apply(lambda x: [w.lower() for w in x])
    if remove_stop:
        stop_words = set(stopwords.words('english'))
        data['tokens'] = data['tokens'].apply(lambda x: [w for w in x if w.lower() not in stop_words])
    if remove_punct:
        data['tokens'] = data['tokens'].apply(lambda x: [w for w in x if w.isalnum()])
        

    fig = (ggplot() + geom_blank(data=data) + theme_void()).draw()
    fig.set_size_inches(fig_width, fig_height * 3)
    gs = gridspec.GridSpec(3, 2)

    # histogram of tokens per document
    ax_tok = fig.add_subplot(gs[2, 0])
    data = preprocess_numeric_variables(data, 'tokens_per_document', lq1=lower_quantile,
                                        hq1=upper_quantile, transform1=transform)
    if hist_bins == 0:
        hist_bins = freedman_diaconis_bins(data['tokens_per_document'], transform)
    gg_hist = (
            ggplot(data, aes(x='tokens_per_document')) +
            geom_histogram(bins=hist_bins, color='black', fill=BAR_COLOR)
    )
    if transform in ['log', 'log_exclude0']:
        gg_hist += scale_x_log10()
    elif transform == 'sqrt':
        gg_hist += scale_x_sqrt()
    _ = gg_hist._draw_using_figure(fig, [ax_tok])
    ax_tok.set_xlabel('# Tokens / Document')
    ax_tok.set_ylabel('count')

    # histogram of characters per document
    ax_char = fig.add_subplot(gs[2, 1])
    data = preprocess_numeric_variables(data, 'characters_per_document', lq1=lower_quantile,
                                        hq1=upper_quantile, transform1=transform)
    if hist_bins == 0:
        hist_bins = freedman_diaconis_bins(data['characters_per_document'], transform)
    gg_hist = (
            ggplot(data, aes(x='characters_per_document')) +
            geom_histogram(bins=hist_bins, color='black', fill=BAR_COLOR)
    )
    if transform in ['log', 'log_exclude0']:
        gg_hist += scale_x_log10()
    elif transform == 'sqrt':
        gg_hist += scale_x_sqrt()
    _ = gg_hist._draw_using_figure(fig, [ax_char])
    ax_char.set_xlabel('# Characters / Document')

    # plot most frequent unigrams
    ax_unigram = fig.add_subplot(gs[0, 0])
    unigrams = pd.DataFrame({'unigrams': [x for y in data['tokens'] for x in set(y)]})
    unigrams['unigrams'] = pd.Categorical(
        unigrams['unigrams'],
        list(unigrams['unigrams'].value_counts().sort_values(ascending=False).index)[:top_ngrams][::-1])
    unigrams = unigrams.dropna()
    gg_uni = (
            ggplot(unigrams, aes(x='unigrams')) +
            geom_bar(fill=BAR_COLOR, color='black') +
            coord_flip()
    )
    _ = gg_uni._draw_using_figure(fig, [ax_unigram])
    ax_unigram.set_ylabel('Most Common Unigrams')
    add_percent_axis(ax_unigram, data.shape[0], flip_axis=True)

    # boxplot of observations per document
    ax_obs = fig.add_subplot(gs[0, 1])
    tmp = pd.DataFrame({'observations_per_document': list(data[column].value_counts())})
    tmp = preprocess_numeric_variables(tmp, 'observations_per_document', lq1=lower_quantile,
                                       hq1=upper_quantile, transform1=transform)
    if hist_bins == 0:
        hist_bins = freedman_diaconis_bins(tmp['observations_per_document'], transform)
    gg_box = (
            ggplot(tmp, aes(x=[''], y='observations_per_document')) +
            geom_boxplot(color='black', fill=BAR_COLOR) +
            coord_flip()
    )
    if transform in ['log', 'log_exclude0']:
        gg_box += scale_y_log10()
    elif transform == 'sqrt':
        gg_box += scale_y_sqrt()
    _ = gg_box._draw_using_figure(fig, [ax_obs])
    ax_obs.set_ylabel('# Observations / Document')

    # plot most frequent bigrams
    ax_bigram = fig.add_subplot(gs[1, :])
    bigrams = pd.DataFrame({'bigrams': [x for y in data['tokens'] for x in set(ngrams(y, 2))]})
    bigrams['bigrams'] = pd.Categorical(
        bigrams['bigrams'],
        list(bigrams['bigrams'].value_counts().sort_values(ascending=False).index)[:top_ngrams][::-1])
    bigrams = bigrams.dropna()
    gg_bi = (
            ggplot(bigrams, aes(x='bigrams')) +
            geom_bar(fill=BAR_COLOR, color='black') +
            coord_flip()
    )
    _ = gg_bi._draw_using_figure(fig, [ax_bigram])
    ax_bigram.set_ylabel('Most Common Bigrams')
    add_percent_axis(ax_bigram, data.shape[0], flip_axis=True)

    num_tokens = len(tokens)
    vocab_size = len(set(tokens)) 
    corpus_size = data[column].size
    plt.suptitle((
        f"{num_tokens} tokens with a vocabulary size of {vocab_size} in a corpus of {corpus_size} documents\n"
        f"{num_missing} missing observations ({perc_missing}%)"), fontsize=12)

    plt.subplots_adjust(top=.93)
    plt.show()


def list_univariate_eda(data, column, fig_height=4, fig_width=8, top_entries=10):
    """ 
    Creates a univariate EDA summary for a provided list column in a pandas DataFrame.
        
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
    top_entries: int, optional
        Maximum number of entries to plot for the top most frequent single entries and pairs.

    Returns 
    -------
    None
        No return value. Directly displays the resulting matplotlib figure.
    """
    data = data.copy()
    # handle missing data
    num_missing = data[column].isnull().sum()
    perc_missing = num_missing / data.shape[0]
    data.dropna(subset=[column], inplace=True)

    fig = (ggplot() + geom_blank(data=data) + theme_void()).draw()
    fig.set_size_inches(fig_width, fig_height * 3)
    gs = gridspec.GridSpec(3, 2)

    # plot most common entries
    ax_single = fig.add_subplot(gs[0, :])
    entries = [i for e in data[column] for i in e]
    singletons = pd.DataFrame({'single': entries})
    order = list(singletons['single'].value_counts().sort_values(ascending=False).index)[:top_entries][::-1]
    singletons['single'] = pd.Categorical(singletons['single'], order)
    singletons = singletons.dropna()

    gg_s = (
            ggplot(singletons, aes(x='single')) +
            geom_bar(fill=BAR_COLOR, color='black') +
            coord_flip()
    )
    _ = gg_s._draw_using_figure(fig, [ax_single])
    ax_single.set_ylabel('Most Common Entries')
    add_percent_axis(ax_single, data.shape[0], flip_axis=True)

    # plot most common entries
    ax_double = fig.add_subplot(gs[1, :])
    pairs = [comb for coll in data[column] for comb in combinations(coll, 2)]
    pairs = pd.DataFrame({'pair': pairs})
    order = list(pairs['pair'].value_counts().sort_values(ascending=False).index)[:top_entries][::-1]
    pairs['pair'] = pd.Categorical(pairs['pair'], order)
    pairs = pairs.dropna()

    gg_s = (
            ggplot(pairs, aes(x='pair')) +
            geom_bar(fill=BAR_COLOR, color='black') +
            coord_flip()
    )
    _ = gg_s._draw_using_figure(fig, [ax_double])
    ax_double.set_ylabel('Most Common Entry Pairs')
    ax_double.set_xlabel('# Observations')
    add_percent_axis(ax_double, data.shape[0], flip_axis=True)

    # plot number of elements
    data['num_entries'] = data[column].apply(lambda x: len(x))
    ax_num = fig.add_subplot(gs[2, 0])
    gg_hour = (
            ggplot(data, aes(x='num_entries')) +
            geom_histogram(bins=data['num_entries'].max(), fill=BAR_COLOR, color='black')
    )
    _ = gg_hour._draw_using_figure(fig, [ax_num])
    ax_num.set_ylabel('count')
    ax_num.set_xlabel('# Entries / Observation')
    ax_num.set_xticks(np.arange(0, data['num_entries'].max() + 1))
    ax_num.set_xticklabels(np.arange(0, data['num_entries'].max() + 1))
    add_percent_axis(ax_num, data.shape[0], flip_axis=False)

    ax_obs = fig.add_subplot(gs[2, 1])
    tmp = pd.DataFrame({'obs': list(data[column].value_counts())})
    gg_box = (
            ggplot(tmp, aes(x=[''], y='obs')) +
            geom_boxplot(color='black', fill=BAR_COLOR) +
            coord_flip()
    )
    _ = gg_box._draw_using_figure(fig, [ax_obs])
    ax_obs.set_xlabel('# Observations / Unique List')

    ax_single.set_title(f"{len(set(entries))} unique entries with {len(entries)} total entries across {data[column].size} observations")
    plt.show()


