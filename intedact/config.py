# Plot controls (fill color for histograms, bar, and boxplots that don't have fill mapped to column)
BAR_COLOR = 'steelblue'

# Column type groupings
DISCRETE_TYPES = [
    'discrete_numeric',
    'unordered_categorical',
    'ordered_categorical',
    'unordered_categorical (inferred from object)'
]

# Widget control values
WIDGET_VALUES = {
    'manual_update': {
        'description': "Update plots manually",
        'width': '0%',
        'widget_options': 'N/A'
    },
    'Run Interact': {
        'description': "Update Plot",
        'width': '0%',
        'widget_options': 'N/A'
    },
    'column': {
        'description': "column: Column to be plotted",
        'width': '22%',
        'widget_options': 'N/A'
    },
    'column1': {
        'description': "column1: Column to be plotted as independent variable",
        'width': '23%',
        'widget_options': 'N/A'
    },
    'column2': {
        'description': "column2: Column to be plotted as dependent variable",
        'width': '24%',
        'widget_options': 'N/A'
    },
    'col_type': {
        'description': ("column type: Column type that determines which univariate summary will be displayed."
                        " Type is automatically inferred by default, but you can manually change the type to"
                        " try to produce a different summary if the column data is compatible."
                        " It is recommended you preprocess the columns to have the desired type prior to"
                        " running this function for greatest accuracy of automatic type inference.\n\n"
                        " The following types are available:\n"
                        "  - 'discrete': categorical or low dimensional numeric variables\n"
                        "  - 'continuous': high dimensional numeric variables \n"
                        "  - 'datetime': datetime columns (should be coerceable to datetime type) \n"
                        "  - 'text': column with strings that represent longer freeform text \n"
                        "  - 'list': column with variable length lists of items, should be pre-coerced such that elements are python lists/tuples/sets\n"),
        'width': '33%',
        'widget_options': ['discrete', 'continuous', 'datetime', 'text', 'list']
    },
    'col1_type': {
        'description': ("column1 type: Column type for column1 that determines which bivariate summary will be displayed with col_type2."
                        " Type is automatically inferred by default, but you can manually change the type to"
                        " try to produce a different summary if the column data is compatible."
                        " It is recommended you preprocess the columns to have the desired type prior to"
                        " running this function for greatest accuracy of automatic type inference.\n\n"
                        " The following types are available:\n"
                        "  - 'discrete': categorical or low dimensional numeric variables\n"
                        "  - 'continuous': high dimensional numeric variables \n"
                        "  - 'datetime': datetime columns (should be coerceable to datetime type) \n"
                        "  - 'text': columns with strings that represent longer freeform text \n"
                        "  - 'list': columns with variable length lists of items, should be preprocessed such that elements are python lists/tuples/set objects\n"),
        'width': '35%',
        'widget_options': ['discrete', 'continuous', 'datetime', 'text', 'list']
    },
    'col2_type': {
        'description': ("column2 type: Column type for column2 that determines which bivariate summary will be displayed with col_type1."
                        " Type is automatically inferred by default, but you can manually change the type to"
                        " try to produce a different summary if the column data is compatible." 
                        " It is recommended you preprocess the columns to have the desired type prior to"
                        " running this function for greatest accuracy of automatic type inference.\n\n"
                        " The following types are available:\n"
                        "  - 'discrete': categorical or low dimensional numeric variables\n"
                        "  - 'continuous': high dimensional numeric variables \n"
                        "  - 'datetime': datetime columns (should be coerceable to datetime type) \n"
                        "  - 'text': column with strings that represent longer freeform text, not categorical data \n"
                        "  - 'list': column with variable length lists of items, should be pre-coerced such that elements are python lists/tuples/sets\n"),
        'width': '35%',
        'widget_options': ['discrete', 'continuous', 'datetime', 'text', 'list']
    },
    'discrete_limit': {
        'description': ("discrete limit: # of unique values a variable must have before it is considered "
                        "continuous rather than discrete"),
        'width': '33%',
        'widget_options': (1, 100, 1)
    },
    'fig_width': {
        'description': "fig width: width of figure in inches",
        'width': '25%',
        'widget_options': (1, 30, 1)
    },
    'fig_height': {
        'description': "fig height: height of figure in inches (multiplied if multiple subplots)",
        'width': '26%',
        'widget_options': (1, 30, 1)
    },
    'level_order': {
        'description':
            ("level order: Order for arranging column levels on plot\n" 
             "  - descending: Arrange levels from most frequent to least frequent\n"
             "  - ascending: Arrange levels from least frequent to most frequent\n"
             "  - sorted: Arrange levels in sorted order of the level values themselves\n"
             "  - random: Randomly arrange levels\n"
             "  - auto: order based on variable type (sorted for numeric variables, provided level order for"
             " ordinal variables, descending for unordered categorical variables"),
        'width': '30%',
        'widget_options': ['auto', 'descending', 'ascending', 'sorted', 'random']
    },
    'level_order1': {
        'description':
            ("level order1: Order for arranging column1 levels on plot\n"
             "  - descending: Arrange levels from most frequent to least frequent\n"
             "  - ascending: Arrange levels from least frequent to most frequent\n"
             "  - sorted: Arrange levels in sorted order of the level values themselves\n"
             "  - random: Randomly arrange levels\n"
             "  - auto: order based on variable type (sorted for numeric variables, provided level order for"
             " ordinal variables, descending for unordered categorical variables"),
        'width': '30%',
        'widget_options': ['auto', 'descending', 'ascending', 'sorted', 'random']
    },
    'level_order2': {
        'description':
            ("level order2: Order for arranging column1 levels on plot\n"
             "  - descending: Arrange levels from most frequent to least frequent\n"
             "  - ascending: Arrange levels from least frequent to most frequent\n"
             "  - sorted: Arrange levels in sorted order of the level values themselves\n"
             "  - random: Randomly arrange levels\n"
             "  - auto: order based on variable type (sorted for numeric variables, provided level order for"
             " ordinal variables, descending for unordered categorical variables"),
        'width': '30%',
        'widget_options': ['auto', 'descending', 'ascending', 'sorted', 'random']
    },
    'top_n': {
        'description': "top n: Maximum number of levels to display before condensing remaining into '__OTHER__'",
        'width': '20%',
        'widget_options': (1, 100, 1)
    },
    'top_entries': {
        'description': "top entries: Maximum number of most common entries/entry pairs to display in plots.",
        'width': '30%',
        'widget_options': (1, 100, 1)
    },
    'flip_axis': {
        'description': "Flip plot orientation",
        'width': '0%',
        'widget_options': 'N/A'
    },
    'label_counts': {
        'description': "Add counts and percentages",
        'width': '0%',
        'widget_options': 'N/A'
    },
    'rotate_labels': {
        'description': "Rotate x axis labels",
        'width': '0%',
        'widget_options': 'N/A'
    },
    'hist_bins': {
        'description': "hist bins: Number of bins to use for the histogram (0 uses geom_histogram default bins)",
        'width': '25%',
        'widget_options': (0, 100, 1)
    },
    'kde': {
        'description': "Overlay density plot on histogram",
        'width': '0%',
        'widget_options': 'N/A'
    },
    'transform': {
        'description':
            ("transform: Transformation to apply to column for plotting \n"
             "  - identity: No transformation\n"
             "  - log: Log transform (add a small constant in case of 0's)\n"
             "  - log_exclude0: Log transform with 0's filtered out\n" 
             "  - sqrt: Square root transform"),
        'width': '28%',
        'widget_options': ['identity', 'log', 'log_exclude0', 'sqrt']
    },
    'transform1': {
        'description':
            ("transform1: Transformation to apply to column1 for plotting \n"
             "  - identity: No transformation\n"
             "  - log: Log transform (add a small constant in case of 0's)\n"
             "  - log_exclude0: Log transform with 0's filtered out\n"
             "  - sqrt: Square root transform"),
        'width': '28%',
        'widget_options': ['identity', 'log', 'log_exclude0', 'sqrt']
    },
    'transform2': {
        'description':
            ("transform2: Transformation to apply to column2 for plotting \n"
             "  - identity: No transformation\n"
             "  - log: Log transform (add a small constant in case of 0's)\n"
             "  - log_exclude0: Log transform with 0's filtered out\n"
             "  - sqrt: Square root transform"),
        'width': '29%',
        'widget_options': ['identity', 'log', 'log_exclude0', 'sqrt']
    },
    'lower_quantile': {
        'description': "lower quantile: Lower quantile of data to remove for plot (not removed for statistics).",
        'width': '35%',
        'widget_options': (0, 1, .01)
    },
    'upper_quantile': {
        'description': "upper quantile: Upper quantile of data to remove for plot (not removed for statistics).",
        'width': '37%',
        'widget_options': (0, 1, .01)
    },
    'lower_quantile1': {
        'description': "lower quantile1: Lower quantile of column1 data to remove for plot.",
        'width': '37%',
        'widget_options': (0, 1, .01)
    },
    'upper_quantile1': {
        'description': "upper quantile1: Upper quantile of column1 data to remove for plot.",
        'width': '39%',
        'widget_options': (0, 1, .01)
    },
    'lower_quantile2': {
        'description': "lower quantile2: Lower quantile of column2 data to remove for plot.",
        'width': '37%',
        'widget_options': (0, 1, .01)
    },
    'upper_quantile2': {
        'description': "upper quantile2: Upper quantile of column2 data to remove for plot.",
        'width': '39%',
        'widget_options': (0, 1, .01)
    },
    'ts_freq': {
        'description': ("time series sampling frequency: pandas frequency string at which to resample and aggregate "
                        "counts for time series plot (i.e. 1M means aggregate by month)"),
        'width': '70%',
        'widget_options': 'N/A'
    },
    'delta_freq': {
        'description': ("time delta units: pandas frequency string to define time delta units. "
                        "(i.e. 1D means compute time deltas in units of days)"),
        'width': '38%',
        'widget_options': 'N/A'
    },
    'top_ngrams': {
        'description': "top ngrams: Maximum number of bars to plot for ngrams",
        'width': '29%',
        'widget_options': (1, 100, 1)
    },
    'remove_punct': {
        'description': "Ignore punctuation for ngrams",
        'width': '0%',
        'widget_options': 'N/A'
    },
    'remove_stop': {
        'description': "Ignore stop words for ngrams",
        'width': '0%',
        'widget_options': 'N/A'
    },
    'lower_case': {
        'description': "Lower case text for ngrams",
        'width': '0%',
        'widget_options': 'N/A'
    },
    'plot_type_cc': {
        'description': ("plot type: Type of plot to show\n"
            "  - 'auto': Defaults to scatter plot\n"
            "  - 'scatter': Draw a scatter plot using geom_scatter\n"
            "  - 'bin2d': Draw a 2d histogram using geom_bin2d\n"
            "  - 'count': Draw a 2d count plot using geom_count"),
        'width': '25%',
        'widget_options': ['auto', 'scatter', 'bin2d', 'count']
    },
    'plot_type_dc1': {
        'description': ("plot type: Type of plot to show\n"
                        "  - 'auto': Defaults to bar plot\n"
                        "  - 'bar': Draw a bar plot\n"
                        "  - 'point': Draw a point plot\n"),
        'width': '25%',
        'widget_options': ['auto', 'bar', 'point']
    },
    'plot_type_dc2': {
        'description': ("plot type: Type of plot to show\n"
                        "  - 'auto': Defaults to overlapping histograms \n"
                        "  - 'histogram': Draw overlapping histograms \n"
                        "  - 'density': Draw overlapping KDEs \n"),
        'width': '25%',
        'widget_options': ['auto', 'histogram', 'density']
    },
    'plot_type_dcn': {
        'description': ("plot type: Type of plot to show\n"
                        "  - 'auto': Defaults to boxplot \n"
                        "  - 'freqpoly': Draw overlapping densities/histograms as colored lines with geom_freqpoly \n"
                        "  - 'boxplot': Draw boxplot per level of discrete variable \n"
                        "  - 'violinplot': Draw violin per level of discrete variable \n"
                        "  - 'facted_histogram': Draw faceted histograms stacked vertically in facets by level \n"
                        "  - 'facted_density': Draw faceted densities stacked vertically in facets by level \n"),
        'width': '25%',
        'widget_options': ['auto', 'freqpoly', 'boxplot', 'violin', 'faceted_histogram', 'faceted_density']
    },
    'trend_line': {
        'description': ("trend line: Trend line to plot over data. 'none' will plot no trend line." 
                        "Other options are passed to plotnine's geom_smooth."),
        'width': '27%',
        'widget_options': ['auto', 'none', 'loess', 'lm']
    },
    'equalize_axes': {
        'description': "Match x and y axes",
        'width': '30%',
        'widget_options': 'N/A'
    },
    'reference_line': {
        'description': "Plot a y = x reference line",
        'width': '0%',
        'widget_options': 'N/A'
    },
    'plot_density': {
        'description': "Overlay a bivariate KDE",
        'width': '0%',
        'widget_options': 'N/A'
    },
    'alpha': {
        'description': "alpha: Amount of transparency to use for points/histograms ranging from 0 (fully transparent) to 1 (opaque)",
        'width': '18%',
        'widget_options': (0, 1, .05)
    },
    'normalize': {
        'description': "Normalize counts to percentages",
        'width': '0%',
        'widget_options': 'N/A'
    },
    'ref_lines': {
        'description': "Add mean & median reference lines",
        'width': '0%',
        'widget_options': 'N/A'
    },
    'varwidth': {
        'description': "Scale boxplot by sample size",
        'width': '0%',
        'widget_options': 'N/A'
    },
    'normalize_dist': {
        'description': "Normalize distributions to densities",
        'width': '0%',
        'widget_options': 'N/A'
    }
}

