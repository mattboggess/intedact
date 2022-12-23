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
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import tldextract
from IPython.display import display
from matplotlib import gridspec
from plotly.subplots import make_subplots

from .bivariate_plots import time_series_plot
from .config import TIME_UNITS
from .data_utils import compute_time_deltas
from .data_utils import convert_to_freq_string
from .data_utils import trim_values
from .helper_plots import boxplot
from .helper_plots import countplot
from .helper_plots import plot_ngrams
from .helper_plots import timeseries_countplot
from .plot_utils import *
from .univariate_plots import histogram
from .univariate_plots import time_series_countplot

FLIP_LEVEL_MINIMUM = 5


def categorical_summary(
    data: pd.DataFrame,
    column: str,
    fig_height: int = 600,
    fig_width: int = 1200,
    order: Union[str, List] = "auto",
    max_levels: int = 20,
    flip_axis: Optional[bool] = None,
    include_missing: bool = False,
    interactive: bool = False,
) -> go.Figure:
    """
    Creates a univariate EDA summary for a provided categorical data column in a pandas DataFrame.

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
         size and number of levels.
        flip_axis: Whether to flip the plot so labels are on y axis. Useful for long level names or lots of levels.
         Default tries to infer based on number of levels and label_rotation value.
        include_missing: Whether to include missing values as an additional level in the data
        interactive: Whether to display plot for interactive use in a jupyter notebook
    """
    data = data.copy()
    if flip_axis is None:
        flip_axis = data[column].nunique() > FLIP_LEVEL_MINIMUM

    fig = make_subplots(
        rows=1,
        cols=1,
    )
    fig.update_layout(height=fig_height, width=fig_width)
    fig = countplot(
        data,
        column,
        fig=fig,
        fig_row=1,
        fig_col=1,
        order=order,
        max_levels=max_levels,
        flip_axis=flip_axis,
        include_missing=include_missing,
    )

    if interactive:
        fig.show()

    return fig


def numeric_summary(
    data: pd.DataFrame,
    column: str,
    fig_height: int = 600,
    fig_width: int = 1200,
    bins: int = 0,
    transform: str = "identity",
    lower_quantile: float = 0,
    upper_quantile: float = 1,
    interactive: bool = False,
) -> go.Figure:
    """
    Creates a univariate EDA summary for a high cardinality numeric data column in a pandas DataFrame.

    Args:
        data: pandas DataFrame to perform EDA on
        column: A string matching a column in the data to visualize
        fig_height: Height of the plot in pixels
        fig_width: Width of the plot in pixels
        bins: Number of bins to use for the histogram. Default (0) is to determines # of bins from the data
        transform: Transformation to apply to the data for plotting:

            - 'identity': no transformation
            - 'log': apply a logarithmic transformation (zero and negative values will be filtered out)
            - 'sqrt': apply a square root transformation
        lower_quantile: Lower quantile to filter data above
        upper_quantile: Upper quantile to filter data below
        interactive: Whether to modify to be used with interactive for ipywidgets
    """
    if bins == 0:
        bins = None
    data = data.copy()
    data = trim_values(data, column, lower_quantile, upper_quantile)
    if transform == "log":
        label = f"log({column})"
        data[label] = np.log(data[column])
    elif transform == "sqrt":
        label = f"sqrt({column})"
        data[label] = np.sqrt(data[column])
    else:
        label = column

    fig = px.histogram(data, x=label, marginal="box", nbins=bins)
    fig.update_layout(height=fig_height, width=fig_width)

    if interactive:
        fig.show()

    return fig


def datetime_summary(
    data: pd.DataFrame,
    column: str,
    fig_height: int = 1000,
    fig_width: int = 1200,
    ts_freq: str = "auto",
    ts_type: str = "line",
    trend_line: str = "auto",
    lower_quantile: float = 0,
    upper_quantile: float = 1,
    interactive: bool = False,
) -> go.Figure:
    """
    Creates a univariate EDA summary for a datetime data column in a pandas DataFrame.

    Args:
        data: pandas DataFrame to perform EDA on
        column: A string matching a column in the data
        fig_height: Height of the plot in inches
        fig_width: Width of the plot in inches
        ts_freq: String describing the frequency at which to aggregate data in one of two formats:

            - A `pandas offset string <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_.
            - A human readable string in the same format passed to date breaks (e.g. "4 months")
            Default is to attempt to intelligently determine a good aggregation frequency.
        ts_type: 'lines', 'markers', or 'lines+markers' to plot a line, points, or line + points
        trend_line: Trend line to plot over data. "None" produces no trend line. Other options are passed
            to `geom_smooth <https://plotnine.readthedocs.io/en/stable/generated/plotnine.geoms.geom_smooth.html>`_.
          a time period ranging from seconds to years. (e.g. '1 year', '3 minutes')
        lower_quantile: Lower quantile to filter data above
        upper_quantile: Upper quantile to filter data below
        interactive: Whether to display figures and tables in jupyter notebook for interactive use
    """
    data = data.copy()
    data = trim_values(data, column, lower_quantile, upper_quantile)

    if trend_line == "none":
        trend_line = None

    data["Month"] = data[column].dt.month_name()
    data["Day of Month"] = data[column].dt.day
    data["Year"] = data[column].dt.year
    data["Hour"] = data[column].dt.hour
    data["Day of Week"] = data[column].dt.day_name()

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"colspan": 1, "rowspan": 1}, {"colspan": 1, "rowspan": 1}],
            [{"colspan": 1, "rowspan": 1}, {"colspan": 1, "rowspan": 1}],
        ],
    )
    fig.update_layout(width=fig_width, height=fig_height)
    fig = timeseries_countplot(
        data,
        column,
        fig,
        fig_row=1,
        fig_col=1,
        ts_freq=ts_freq,
        ts_type=ts_type,
        trend_line=trend_line,
    )

    data["Month"] = pd.Categorical(
        data["Month"], categories=list(calendar.month_name)[1:], ordered=True
    )
    fig = countplot(
        data,
        "Month",
        fig,
        fig_row=2,
        fig_col=1,
        flip_axis=True,
    )

    data["Day of Month"] = pd.Categorical(
        data["Day of Month"], categories=np.arange(1, 32, 1), ordered=True
    )
    fig = countplot(
        data,
        "Day of Month",
        fig,
        fig_row=2,
        fig_col=2,
        flip_axis=True,
        max_levels=35,
    )

    data["Day of Week"] = pd.Categorical(
        data["Day of Week"], categories=list(calendar.day_name), ordered=True
    )
    fig = countplot(
        data,
        "Day of Week",
        fig,
        fig_row=3,
        fig_col=1,
        flip_axis=True,
    )

    data["Hour"] = pd.Categorical(
        data["Hour"], categories=np.arange(0, 24, 1), ordered=True
    )
    fig = countplot(
        data, "Hour", fig, fig_row=3, fig_col=2, flip_axis=True, max_levels=25
    )

    fig.update(layout_showlegend=False)

    if interactive:
        fig.show()

    return fig


def text_summary(
    data: pd.DataFrame,
    column: str,
    fig_height: int = 1000,
    fig_width: int = 1200,
    top_ngrams: int = 10,
    remove_punct: bool = True,
    remove_stop: bool = True,
    lower_case: bool = True,
    interactive: bool = False,
) -> go.Figure:
    """
    Creates a univariate EDA summary for a text variable column in a pandas DataFrame. Currently only
    supports English.

    Args:
        data: Dataset to perform EDA on
        column: A string matching a column in the data
        fig_height: Height of the plot in pixels
        fig_width: Width of the plot in pixels
        top_ngrams: Maximum number of ngrams to plot for the top most frequent unigrams to trigrams
        remove_punct: Whether to remove punctuation during tokenization
        remove_stop: Whether to remove stop words during tokenization
        lower_case: Whether to lower case text for tokenization
        interactive: Whether to display figures and tables in jupyter notebook for interactive use

    Returns:
        Tuple containing matplotlib Figure drawn and summary stats DataFrame
    """
    from nltk import word_tokenize
    from nltk.corpus import stopwords

    data = data.copy()
    data = data.dropna(subset=[column])

    # Compute number of characters per document
    data["# Characters / Document"] = data[column].apply(lambda x: len(x))

    # Tokenize the text
    data["tokens"] = data[column].apply(lambda x: [w for w in word_tokenize(x)])
    if lower_case:
        data["tokens"] = data["tokens"].apply(lambda x: [w.lower() for w in x])
    if remove_stop:
        stop_words = set(stopwords.words("english"))
        data["tokens"] = data["tokens"].apply(
            lambda x: [w for w in x if w.lower() not in stop_words]
        )
    if remove_punct:
        data["tokens"] = data["tokens"].apply(lambda x: [w for w in x if w.isalnum()])
    data["# Tokens / Document"] = data["tokens"].apply(lambda x: len(x))

    # Compute summary table
    vocab_size = len(set([x for y in data["tokens"] for x in y]))

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"colspan": 1, "rowspan": 1}, {"colspan": 1, "rowspan": 1}],
            [{"colspan": 1, "rowspan": 1}, {"colspan": 1, "rowspan": 1}],
            [{"colspan": 1, "rowspan": 1}, {"colspan": 1, "rowspan": 1}],
        ],
    )
    fig.update_layout(width=fig_width, height=fig_height)

    fig = plot_ngrams(
        data["tokens"],
        fig=fig,
        fig_col=1,
        fig_row=1,
        ngram_type="tokens",
        lim_ngrams=top_ngrams,
    )

    fig = plot_ngrams(
        data["tokens"],
        fig=fig,
        fig_col=1,
        fig_row=2,
        ngram_type="bigrams",
        lim_ngrams=top_ngrams,
    )

    fig = plot_ngrams(
        data["tokens"],
        fig=fig,
        fig_col=1,
        fig_row=3,
        ngram_type="trigrams",
        lim_ngrams=top_ngrams,
    )

    fig = boxplot(data, "# Tokens / Document", fig, fig_row=1, fig_col=2)
    fig = boxplot(data, "# Characters / Document", fig, fig_row=2, fig_col=2)
    tmp = pd.DataFrame({"# Repeats / Document": list(data[column].value_counts())})
    fig = boxplot(tmp, "# Repeats / Document", fig, fig_row=3, fig_col=2)
    fig.update(layout_showlegend=False)
    fig.update_layout(title_text=f"{column} (vocab size: {vocab_size})", title_x=0.5)

    if interactive:
        fig.show()
    return fig


def collection_univariate_summary(
    data: pd.DataFrame,
    column: str,
    fig_height: int = 6,
    fig_width: int = 12,
    fontsize: int = 15,
    color_palette: str = None,
    top_entries: int = 10,
    sort_collections: bool = False,
    remove_duplicates: bool = False,
    interactive: bool = False,
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Creates a univariate EDA summary for a provided collections column in a pandas DataFrame.

    The provided column should be an object type containing lists, tuples, or sets.

    Args:
        data: Dataset to perform EDA on
        column: A string matching a column in the data
        fig_height: Height of the plot in inches
        fig_width: Width of the plot in inches
        fontsize: Font size of axis and tick labels
        color_palette: Seaborn color palette to use
        top_entries: Max number of entries to show for countplots
        sort_collections: Whether to sort collections and ignore original order
        remove_duplicates: Whether to remove duplicate entries from collections
        interactive: Whether to display figures and tables in jupyter notebook for interactive use

    Returns:
        Tuple containing matplotlib Figure drawn and summary stats DataFrame
    """
    data = data.copy()
    if color_palette != "":
        sns.set_palette(color_palette)
    else:
        sns.set_palette("tab10")

    # Compute derived transforms
    data[column] = data[column].apply(lambda x: tuple(x))
    data["# Entries / Collection"] = data[column].apply(lambda x: len(x))
    tmp = data.explode(column)

    # Compute Summary Table
    table = compute_univariate_summary_table(data, column, "categorical")
    table["count_unique_entries"] = tmp[~tmp[column].isnull()][column].nunique()
    num_table = compute_univariate_summary_table(
        data, "# Entries / Collection", "numeric"
    )
    table = pd.concat([table, num_table])

    fig = plt.figure(figsize=(fig_width, fig_height * 2))
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

    # Remove duplicates and sort collections
    if remove_duplicates:
        data[column] = data[column].apply(lambda x: tuple(set(x)))
    if sort_collections:
        data[column] = data[column].apply(lambda x: tuple(sorted(x)))

    # Plot most common collections
    ax = fig.add_subplot(spec[0, :])
    ax = countplot(
        data,
        column,
        ax=ax,
        flip_axis=True,
        max_levels=top_entries,
        add_other=False,
        label_fontsize=10,
        fontsize=fontsize,
    )

    # Plot most common individual entries
    ax = fig.add_subplot(spec[1, 0])
    ax = countplot(
        tmp,
        column,
        ax=ax,
        flip_axis=True,
        max_levels=top_entries,
        add_other=False,
        label_fontsize=10,
        percent_denominator=data.shape[0],
        fontsize=fontsize,
    )
    ax.set_ylabel("Most Common Entries")
    set_fontsize(ax, fontsize)

    # Plot most common individual entries
    ax = fig.add_subplot(spec[1, 1])
    if data["# Entries / Collection"].nunique() <= 20:
        ax = countplot(
            data,
            "# Entries / Collection",
            ax=ax,
            flip_axis=True,
            max_levels=20,
            add_other=False,
            label_fontsize=10,
            fontsize=fontsize,
        )
    else:
        ax = histogram(
            data,
            "# Entries / Collection",
            ax=ax,
        )
    set_fontsize(ax, fontsize)

    plt.tight_layout()
    if interactive:
        display(table)
        plt.show()

    return table, fig


def url_summary(
    data: pd.DataFrame,
    column: str,
    fig_height: int = 1000,
    fig_width: int = 1200,
    top_entries: int = 10,
    interactive: bool = False,
):
    """
    Creates a univariate EDA summary for a provided url column in a pandas DataFrame. The provided column should be
    a string/object column containing urls.

    Args:
        data: Dataset to perform EDA on
        column: A string matching a column in the data
        fig_height: Height of the plot in inches
        fig_width: Width of the plot in inches
        fontsize: Font size of axis and tick labels
        color_palette: Seaborn color palette to use
        top_entries: Max number of entries to show for countplots
        interactive: Whether to display figures and tables in jupyter notebook for interactive use

    Returns:
        Tuple containing matplotlib Figure drawn and summary stats DataFrame
    """
    data = data.copy()

    # Compute Derived Information
    data["is_https"] = data[column].str.startswith("https")
    data["parse"] = data[column].apply(
        lambda x: tldextract.extract(x) if not pd.isna(x) else None
    )
    data["Domain"] = data["parse"].apply(
        lambda x: x.domain if not pd.isna(x) else "Domain Unknown"
    )
    data["Domain Suffix"] = data["parse"].apply(
        lambda x: x.suffix if not pd.isna(x) else "Domain Suffix Unknown"
    )
    data["File Type"] = data[column].str.extract("\.([a-z]{3})$")
    data["File Type"] = data.apply(
        lambda x: x["File Type"] if x["File Type"] != x["Domain Suffix"] else None,
        axis=1,
    ).fillna("No File Detected")

    # Compute Summary Table
    # table["percent_https"] = data["is_https"].mean() * 100
    # table["count_unique_domains"] = data["Domain"].nunique()
    # table["count_unique_domain_suffixes"] = data["Domain Suffix"].nunique()
    # table["count_unique_file_types"] = data[data["File Type"] != "No File Detected"][
    #    "File Type"
    # ].nunique()

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"colspan": 2, "rowspan": 1}, None],
            [{"colspan": 2, "rowspan": 1}, None],
            [{"colspan": 1, "rowspan": 1}, {"colspan": 1, "rowspan": 1}],
        ],
    )
    fig.update_layout(width=fig_width, height=fig_height)

    # Plot most common urls
    fig = countplot(
        data,
        column,
        fig=fig,
        fig_col=1,
        fig_row=1,
        flip_axis=True,
        max_levels=top_entries,
    )

    # Plot most common domains
    fig = countplot(
        data,
        "Domain",
        fig=fig,
        fig_col=1,
        fig_row=2,
        flip_axis=True,
        max_levels=top_entries,
    )

    # Plot most common domain suffixes
    fig = countplot(
        data,
        "Domain Suffix",
        fig=fig,
        fig_col=1,
        fig_row=3,
        flip_axis=True,
        max_levels=top_entries,
    )

    # Plot most common file types
    fig = countplot(
        data,
        "File Type",
        fig=fig,
        fig_col=2,
        fig_row=3,
        flip_axis=False,
        max_levels=top_entries,
    )

    fig.update_layout(title_text=f"{column}", title_x=0.5)
    fig.update(layout_showlegend=False)

    if interactive:
        fig.show()

    return fig
