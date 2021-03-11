import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from itertools import combinations
from matplotlib import gridspec
from IPython.display import display
from typing import Union, List, Tuple
from .utils import *
from .univariate_plots import (
    histogram,
    boxplot,
    countplot,
    continuous_summary_stats,
    time_series_countplot,
)
from .bivariate_plots import time_series_plot
from .config import BAR_COLOR
import warnings
import calendar


def discrete_univariate_summary(
    data: pd.DataFrame,
    column: str,
    fig_height: int = 5,
    fig_width: int = 10,
    order: Union[str, List] = "auto",
    max_levels: int = 30,
    flip_axis: Optional[bool] = None,
    label_rotation: Optional[int] = None,
    percent_axis: bool = True,
    label_counts: bool = True,
    label_fontsize: Optional[float] = None,
    include_missing: bool = False,
    interactive: bool = False,
    **kwargs,
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Creates a univariate EDA summary for a provided discrete data column in a pandas DataFrame.

    Summary consists of a count plot with twin axes for counts and percentages for each level of the
    variable and a small summary table.

    Args:
        data: pandas DataFrame with data to be plotted
        column: column in the dataframe to plot
        fig_width: figure width in inches
        fig_height: figure height in inches
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
        include_missing: Whether to include missing values as an additional level in the data
        interactive: Whether to display plot and table for interactive use in a jupyter notebook
        kwargs: Additional keyword arguments passed through to [sns.barplot](https://seaborn.pydata.org/generated/seaborn.barplot.html)

    Returns:
        Summary table and matplotlib figure with countplot

    Example:
        .. plot::

            import seaborn as sns
            import intedact
            data = sns.load_dataset('tips')
            intedact.discrete_univariate_summary(data, 'day', interactive=True)
    """
    data = data.copy()

    # Get summary table
    count_missing = data[column].isnull().sum()
    perc_missing = 100 * count_missing / data.shape[0]
    count_obs = data.shape[0] - count_missing
    count_levels = data[column].nunique()
    summary_table = pd.DataFrame(
        {
            "count_observed": [count_obs],
            "count_levels": [count_levels],
            "count_missing": [count_missing],
            "percent_missing": [perc_missing],
        }
    )

    # Plot countplot
    fig, axs = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax = countplot(
        data,
        column,
        order=order,
        max_levels=max_levels,
        percent_axis=percent_axis,
        label_counts=label_counts,
        flip_axis=flip_axis,
        label_fontsize=label_fontsize,
        include_missing=include_missing,
        label_rotation=label_rotation,
    )

    if interactive:
        display(summary_table)
        plt.show()

    return summary_table, fig


def continuous_univariate_summary(
    data: pd.DataFrame,
    column: str,
    fig_height: int = 4,
    fig_width: int = 8,
    hist_bins: int = 0,
    transform: str = "identity",
    lower_quantile: float = 0,
    upper_quantile: float = 1,
    kde: bool = False,
    bar_color: str = BAR_COLOR,
    interactive: bool = False,
) -> None:
    """
    Creates a univariate EDA summary for a provided continuous data column in a pandas DataFrame.

    Summary consists of a histogram, boxplot, and small table of summary statistics.

    Args:
        data: pandas DataFrame to perform EDA on
        column: A string matching a column in the data to visualize
        fig_height: Height of the plot in inches
        fig_width: Width of the plot in inches
        hist_bins: Number of bins to use for the histogram. Default is 0 which determines # of bins from the data
        transform: Transformation to apply to the data for plotting:

            - 'identity': no transformation
            - 'log': apply a logarithmic transformation with small constant added in case of zero values
            - 'log_exclude0': apply a logarithmic transformation with zero values removed
            - 'sqrt': apply a square root transformation
        lower_quantile: Lower quantile of data to remove before plotting for ignoring outliers
        upper_quantile: Upper quantile of data to remove before plotting for ignoring outliers
        kde: Whether to overlay a KDE plot on the histogram
        bar_color: Color to use for bars
        interactive: Whether to modify to be used with interactive for ipywidgets

    Returns:
        Tuple containing matplotlib Figure drawn and summary stats DataFrame

    Example:
        .. plot::

            import seaborn as sns
            import intedact
            data = sns.load_dataset('tips')
            intedact.continuous_univariate_summary(data, 'total_bill', interactive=True)[0]
    """
    if interactive:
        data = data.copy()

    # compute summary stats
    table = continuous_summary_stats(data, column, lower_quantile, upper_quantile)
    if interactive:
        display(table)

    # make histogram and boxplot figure (empty figure hack for plotnine plotting with subplots)
    # https://github.com/has2k1/plotnine/issues/373
    fig = (ggplot() + geom_blank(data=data) + theme_void()).draw()
    fig.set_size_inches(fig_width, fig_height * 2)
    gs = gridspec.GridSpec(2, 1)
    ax_hist = fig.add_subplot(gs[0])
    ax_box = fig.add_subplot(gs[1])

    histogram(
        data,
        column,
        fig,
        ax=ax_hist,
        hist_bins=hist_bins,
        transform=transform,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        kde=kde,
    )

    boxplot(
        data,
        column,
        fig,
        ax=ax_box,
        transform=transform,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )

    return fig, table


def datetime_univariate_summary(
    data: pd.DataFrame,
    column: str,
    fig_height: int = 4,
    fig_width: int = 8,
    ts_freq: str = "auto",
    delta_freq: str = "auto",
    ts_type: str = "line",
    trend_line: str = "auto",
    label_counts: bool = True,
    theme: str = None,
    interactive: bool = False,
) -> plt.Figure:
    """
    Creates a univariate EDA summary for a provided datetime data column in a pandas DataFrame.

    Produces the following summary plots:

      - a time series plot of counts aggregated at the temporal resolution provided by ts_freq
      - a time series plot of time deltas between successive observations in units defined by delta_freq
      - countplots for the following metadata from the datetime object:

        - day of week
        - day of month
        - month
        - year
        - hour
        - minute

    Args:
        data: pandas DataFrame to perform EDA on
        column: A string matching a column in the data
        fig_height: Height of the plot in inches
        fig_width: Width of the plot in inches
        ts_freq: String describing the frequency at which to aggregate data in one of two formats:

            - A `pandas offset string <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_.
            - A human readable string in the same format passed to date breaks (e.g. "4 months")
            Default is to attempt to intelligently determine a good aggregation frequency.
        delta_freq: String describing the units in which to compute time deltas between successive observations in one of two formats:

            - A `pandas offset string <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_.
            - A human readable string in the same format passed to date breaks (e.g. "4 months")
            Default is to attempt to intelligently determine a good frequency unit.
        ts_type: 'line' plots a line graph while 'point' plots points for observations
        trend_line: Trend line to plot over data. "None" produces no trend line. Other options are passed
            to `geom_smooth <https://plotnine.readthedocs.io/en/stable/generated/plotnine.geoms.geom_smooth.html>`_.
        label_counts: Whether to add count/percentage on bars in countplots where feasible
        theme: plotnine theme to use for the plot, str must match available theme listed `here <https://plotnine.readthedocs.io/en/stable/api.html#themes>`_.
        interactive: Whether to display figures and tables in jupyter notebook for interactive use

    Returns:
        matplotlib Figure plot is drawn to

    Examples:
        .. plot::

            import pandas as pd
            import intedact
            data = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/tidytuesday_tweets/data.csv")
            data['created_at'] = pd.to_datetime(data.created_at)
            intedact.datetime_univariate_summary(data, 'created_at', ts_freq='1 week', delta_freq='1 hour')
    """
    if interactive:
        data = data.copy()

    # handle missing data
    num_missing = data[column].isnull().sum()
    perc_missing = num_missing / data.shape[0]

    # make histogram and boxplot figure (empty figure hack for plotting with subplots)
    # https://github.com/has2k1/plotnine/issues/373
    fig = (ggplot() + geom_blank(data=data) + theme_void()).draw()
    fig.set_size_inches(fig_width, fig_height * 5)
    grid = fig.add_gridspec(5, 2)

    # compute extra columns with datetime attributes
    data["Month"] = data[column].dt.month_name()
    data["Day of Month"] = data[column].dt.day
    data["Year"] = data[column].dt.year
    data["Hour"] = data[column].dt.hour
    data["Day of Week"] = data[column].dt.day_name()

    # compute time deltas
    # TODO: Handle this more intelligently
    if delta_freq == "auto":
        delta_freq = "1D"
    delta_freq_original = delta_freq
    delta_freq = convert_to_freq_string(delta_freq)
    dts = data[column].sort_values(ascending=True)
    data["deltas"] = (dts - dts.shift(1)) / pd.Timedelta(delta_freq)

    # time series count plot
    ax = fig.add_subplot(grid[0, :])
    _, ax, _ = time_series_countplot(
        data, column, fig, ax, ts_freq=ts_freq, ts_type=ts_type, trend_line=trend_line
    )
    ax.set_ylabel(f"Count (aggregated at {ts_freq})")
    plt.title(
        (
            f"{data[column].size} observations ranging from {data[column].min()} to {data[column].max()}\n"
            f"{num_missing} missing observations ({perc_missing}%)"
        )
    )

    # time series of time deltas
    ax = fig.add_subplot(grid[1, :])
    _, ax, _ = time_series_plot(
        data,
        column,
        "deltas",
        fig,
        ax,
        ts_type=ts_type,
        trend_line=trend_line,
    )
    ax.set_ylabel(f"Time deltas between observations\nUnits of {delta_freq_original}")

    # countplot by year
    year_order = np.arange(data["Year"].min(), data["Year"].max() + 1, 1)
    num_years = len(year_order)
    if num_years <= 10:
        label_rotation = 0
        year_label_counts = label_counts
    else:
        label_rotation = 90
        year_label_counts = False
    data["Year"] = pd.Categorical(data["Year"], categories=year_order, ordered=True)
    ax = fig.add_subplot(grid[2, :])
    _, ax, _ = countplot(
        data,
        "Year",
        fig,
        ax,
        theme=theme,
        label_counts=year_label_counts,
        label_rotation=label_rotation,
    )

    # countplot by month
    data["Month"] = pd.Categorical(
        data["Month"], categories=list(calendar.month_name)[1:], ordered=True
    )
    ax = fig.add_subplot(grid[3, 0])
    _, ax, _ = countplot(
        data, "Month", fig, ax, theme=theme, label_counts=label_counts, flip_axis=True
    )

    # countplot by day of month
    data["Day of Month"] = pd.Categorical(
        data["Day of Month"], categories=np.arange(1, 32, 1), ordered=True
    )
    ax = fig.add_subplot(grid[3, 1])
    _, ax, _ = countplot(
        data,
        "Day of Month",
        fig,
        ax,
        theme=theme,
        label_counts=False,
        flip_axis=True,
        max_levels=35,
    )

    # countplot by day of week
    data["Day of Week"] = pd.Categorical(
        data["Day of Week"], categories=list(calendar.day_name), ordered=True
    )
    ax = fig.add_subplot(grid[4, 0])
    _, ax, _ = countplot(
        data,
        "Day of Week",
        fig,
        ax,
        theme=theme,
        label_counts=label_counts,
        flip_axis=True,
    )

    # countplot by hour of day
    data["Hour"] = pd.Categorical(
        data["Hour"], categories=np.arange(0, 24, 1), ordered=True
    )
    ax = fig.add_subplot(grid[4, 1])
    _, ax, _ = countplot(
        data, "Hour", fig, ax, theme=theme, label_counts=False, flip_axis=True
    )

    plt.tight_layout()
    plt.show()


def text_univariate_eda(
    data,
    column,
    fig_height=4,
    fig_width=8,
    top_ngrams=10,
    transform="identity",
    hist_bins=0,
    lower_quantile=0,
    upper_quantile=1,
    remove_punct=True,
    remove_stop=True,
    lower_case=True,
):
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
    data["characters_per_document"] = data[column].apply(lambda x: len(x))
    data["tokens"] = data[column].apply(lambda x: [w for w in word_tokenize(x)])
    data["tokens_per_document"] = data["tokens"].apply(lambda x: len(x))
    tokens = [x for y in data["tokens"] for x in y]

    # filters for ngram computations
    if lower_case:
        data["tokens"] = data["tokens"].apply(lambda x: [w.lower() for w in x])
    if remove_stop:
        stop_words = set(stopwords.words("english"))
        data["tokens"] = data["tokens"].apply(
            lambda x: [w for w in x if w.lower() not in stop_words]
        )
    if remove_punct:
        data["tokens"] = data["tokens"].apply(lambda x: [w for w in x if w.isalnum()])

    fig = (ggplot() + geom_blank(data=data) + theme_void()).draw()
    fig.set_size_inches(fig_width, fig_height * 3)
    gs = gridspec.GridSpec(3, 2)

    # histogram of tokens per document
    ax_tok = fig.add_subplot(gs[2, 0])
    data = preprocess_numeric_variables(
        data,
        "tokens_per_document",
        lq1=lower_quantile,
        hq1=upper_quantile,
        transform1=transform,
    )
    if hist_bins == 0:
        hist_bins = freedman_diaconis_bins(data["tokens_per_document"], transform)
    gg_hist = ggplot(data, aes(x="tokens_per_document")) + geom_histogram(
        bins=hist_bins, color="black", fill=BAR_COLOR
    )
    if transform in ["log", "log_exclude0"]:
        gg_hist += scale_x_log10()
    elif transform == "sqrt":
        gg_hist += scale_x_sqrt()
    _ = gg_hist._draw_using_figure(fig, [ax_tok])
    ax_tok.set_xlabel("# Tokens / Document")
    ax_tok.set_ylabel("count")

    # histogram of characters per document
    ax_char = fig.add_subplot(gs[2, 1])
    data = preprocess_numeric_variables(
        data,
        "characters_per_document",
        lq1=lower_quantile,
        hq1=upper_quantile,
        transform1=transform,
    )
    if hist_bins == 0:
        hist_bins = freedman_diaconis_bins(data["characters_per_document"], transform)
    gg_hist = ggplot(data, aes(x="characters_per_document")) + geom_histogram(
        bins=hist_bins, color="black", fill=BAR_COLOR
    )
    if transform in ["log", "log_exclude0"]:
        gg_hist += scale_x_log10()
    elif transform == "sqrt":
        gg_hist += scale_x_sqrt()
    _ = gg_hist._draw_using_figure(fig, [ax_char])
    ax_char.set_xlabel("# Characters / Document")

    # plot most frequent unigrams
    ax_unigram = fig.add_subplot(gs[0, 0])
    unigrams = pd.DataFrame({"unigrams": [x for y in data["tokens"] for x in set(y)]})
    unigrams["unigrams"] = pd.Categorical(
        unigrams["unigrams"],
        list(unigrams["unigrams"].value_counts().sort_values(ascending=False).index)[
            :top_ngrams
        ][::-1],
    )
    unigrams = unigrams.dropna()
    gg_uni = (
        ggplot(unigrams, aes(x="unigrams"))
        + geom_bar(fill=BAR_COLOR, color="black")
        + coord_flip()
    )
    _ = gg_uni._draw_using_figure(fig, [ax_unigram])
    ax_unigram.set_ylabel("Most Common Unigrams")
    add_percent_axis(ax_unigram, data.shape[0], flip_axis=True)

    # boxplot of observations per document
    ax_obs = fig.add_subplot(gs[0, 1])
    tmp = pd.DataFrame({"observations_per_document": list(data[column].value_counts())})
    tmp = preprocess_numeric_variables(
        tmp,
        "observations_per_document",
        lq1=lower_quantile,
        hq1=upper_quantile,
        transform1=transform,
    )
    if hist_bins == 0:
        hist_bins = freedman_diaconis_bins(tmp["observations_per_document"], transform)
    gg_box = (
        ggplot(tmp, aes(x=[""], y="observations_per_document"))
        + geom_boxplot(color="black", fill=BAR_COLOR)
        + coord_flip()
    )
    if transform in ["log", "log_exclude0"]:
        gg_box += scale_y_log10()
    elif transform == "sqrt":
        gg_box += scale_y_sqrt()
    _ = gg_box._draw_using_figure(fig, [ax_obs])
    ax_obs.set_ylabel("# Observations / Document")

    # plot most frequent bigrams
    ax_bigram = fig.add_subplot(gs[1, :])
    bigrams = pd.DataFrame(
        {"bigrams": [x for y in data["tokens"] for x in set(ngrams(y, 2))]}
    )
    bigrams["bigrams"] = pd.Categorical(
        bigrams["bigrams"],
        list(bigrams["bigrams"].value_counts().sort_values(ascending=False).index)[
            :top_ngrams
        ][::-1],
    )
    bigrams = bigrams.dropna()
    gg_bi = (
        ggplot(bigrams, aes(x="bigrams"))
        + geom_bar(fill=BAR_COLOR, color="black")
        + coord_flip()
    )
    _ = gg_bi._draw_using_figure(fig, [ax_bigram])
    ax_bigram.set_ylabel("Most Common Bigrams")
    add_percent_axis(ax_bigram, data.shape[0], flip_axis=True)

    num_tokens = len(tokens)
    vocab_size = len(set(tokens))
    corpus_size = data[column].size
    plt.suptitle(
        (
            f"{num_tokens} tokens with a vocabulary size of {vocab_size} in a corpus of {corpus_size} documents\n"
            f"{num_missing} missing observations ({perc_missing}%)"
        ),
        fontsize=12,
    )

    plt.subplots_adjust(top=0.93)
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
    singletons = pd.DataFrame({"single": entries})
    order = list(
        singletons["single"].value_counts().sort_values(ascending=False).index
    )[:top_entries][::-1]
    singletons["single"] = pd.Categorical(singletons["single"], order)
    singletons = singletons.dropna()

    gg_s = (
        ggplot(singletons, aes(x="single"))
        + geom_bar(fill=BAR_COLOR, color="black")
        + coord_flip()
    )
    _ = gg_s._draw_using_figure(fig, [ax_single])
    ax_single.set_ylabel("Most Common Entries")
    add_percent_axis(ax_single, data.shape[0], flip_axis=True)

    # plot most common entries
    ax_double = fig.add_subplot(gs[1, :])
    pairs = [comb for coll in data[column] for comb in combinations(coll, 2)]
    pairs = pd.DataFrame({"pair": pairs})
    order = list(pairs["pair"].value_counts().sort_values(ascending=False).index)[
        :top_entries
    ][::-1]
    pairs["pair"] = pd.Categorical(pairs["pair"], order)
    pairs = pairs.dropna()

    gg_s = (
        ggplot(pairs, aes(x="pair"))
        + geom_bar(fill=BAR_COLOR, color="black")
        + coord_flip()
    )
    _ = gg_s._draw_using_figure(fig, [ax_double])
    ax_double.set_ylabel("Most Common Entry Pairs")
    ax_double.set_xlabel("# Observations")
    add_percent_axis(ax_double, data.shape[0], flip_axis=True)

    # plot number of elements
    data["num_entries"] = data[column].apply(lambda x: len(x))
    ax_num = fig.add_subplot(gs[2, 0])
    gg_hour = ggplot(data, aes(x="num_entries")) + geom_histogram(
        bins=data["num_entries"].max(), fill=BAR_COLOR, color="black"
    )
    _ = gg_hour._draw_using_figure(fig, [ax_num])
    ax_num.set_ylabel("count")
    ax_num.set_xlabel("# Entries / Observation")
    ax_num.set_xticks(np.arange(0, data["num_entries"].max() + 1))
    ax_num.set_xticklabels(np.arange(0, data["num_entries"].max() + 1))
    add_percent_axis(ax_num, data.shape[0], flip_axis=False)

    ax_obs = fig.add_subplot(gs[2, 1])
    tmp = pd.DataFrame({"obs": list(data[column].value_counts())})
    gg_box = (
        ggplot(tmp, aes(x=[""], y="obs"))
        + geom_boxplot(color="black", fill=BAR_COLOR)
        + coord_flip()
    )
    _ = gg_box._draw_using_figure(fig, [ax_obs])
    ax_obs.set_xlabel("# Observations / Unique List")

    ax_single.set_title(
        f"{len(set(entries))} unique entries with {len(entries)} total entries across {data[column].size} observations"
    )
    plt.show()
