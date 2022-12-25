FLIP_LEVEL_COUNT = 5

WIDGET_PARAMS = {
    "manual_update": {
        "description": "Update plots manually",
        "width": "0%",
        "widget": "N/A",
    },
    "Update Summary": {
        "description": "Update Plot",
        "width": "0%",
        "widget": "N/A",
    },
    "column": dict(
        description="Column: Column to be plotted",
        style={"description_width": "22%"},
    ),
    "column1": dict(
        description="Column1: Column to be plotted on x-axis",
        style={"description_width": "22%"},
    ),
    "column2": dict(
        description="Column2: Column to be plotted on y-axis",
        style={"description_width": "22%"},
    ),
    "univariate_summary_type": dict(
        description=(
            "Summary Type: Type of univariate summary to display."
            " Type is automatically inferred by default, but you can manually change the type to"
            " try to produce a different summary if the column data is compatible."
            " It is recommended you preprocess the columns to have the desired type prior to"
            " running this function for greatest accuracy of automatic type inference.\n\n"
            " The following types are available:\n"
            "  - 'categorical': categorical or low cardinality numeric variables\n"
            "  - 'numeric': high cardinality numeric variables \n"
            "  - 'datetime': datetime columns (should be coerceable to datetime type) \n"
            "  - 'text': column with strings that represent longer freeform text \n"
            "  - 'collection': column with variable length collections of items, should be pre-coerced such that elements are python lists/tuples/sets\n"
            "  - 'url': column with strings that represent urls\n"
        ),
        style={"description_width": "36%"},
        options=["categorical", "numeric", "datetime", "text", "collection", "url"],
    ),
    "bivariate_summary_type": dict(
        description=(
            "Summary Type: Type of bivariate summary to display."
            " Type is automatically inferred by default, but you can manually change the type to"
            " try to produce a different summary if the column data is compatible."
            " It is recommended you preprocess the columns to have the desired type prior to"
            " running this function for greatest accuracy of automatic type inference.\n\n"
            " The following types are available:\n"
            "  - 'numeric-numeric': For pairs of numeric variables \n"
            "  - 'numeric-categorical': numeric as independent, categorical as dependent \n"
            "  - 'categorical-numeric': categorical as independent, numeric as dependent \n"
            "  - 'categorical-categorical': For pairs of categorical variables \n"
        ),
        style={"description_width": "36%"},
        options=[
            "numeric-numeric",
            "categorical-categorical",
            "categorical-numeric",
            "numeric-categorical",
            "unsupported",
        ],
    ),
    "auto_update": dict(description="Auto Update Summaries", value=True),
    "fig_width": dict(
        description="Figure Width: Width of figure in pixels",
        min=100,
        max=5000,
        step=1,
        value=1200,
        style={"description_width": "32%"},
    ),
    "fig_height": dict(
        description="Figure Height: Height of figure in pixels",
        min=100,
        max=5000,
        step=1,
        value=1000,
        style={"description_width": "34%"},
    ),
    "order": dict(
        description=(
            "Level Order: Order in which to sort the levels of the categorical variable for plotting:\n"
            "  - ** 'auto' **: sorts ordinal variables by provided ordering, nominal variables by descending frequency, and numeric variables in sorted order.\n"
            "  - ** 'descending' **: sorts in descending frequency.\n"
            "  - ** 'ascending' **: sorts in ascending frequency.\n"
            "  - ** 'sorted' **: sorts according to sorted order of the levels themselves.\n"
            "  - ** 'random' **: produces a random order.Useful if there are too many levels for one plot.\n"
            "Or you can pass a list of level names in directly for your own custom order.\n"
        ),
        options=["auto", "descending", "ascending", "sorted", "random"],
        value="auto",
        style={"description_width": "30%"},
    ),
    "dist_type": dict(
        description=(
            "Distribution Type: Type of distribution to plot \n"
            "  - ** 'norm_hist+kde' **: Histogram and KDE normalized as probability density \n"
            "  - ** 'norm_hist_only' **: Histogram normalized as probability density \n"
            "  - ** 'kde_only' **: KDE normalized as probability density \n"
            "  - ** 'unnorm_hist_only' **: Unnormalized histogram with counts\n"
        ),
        options=["norm_hist+kde", "norm_hist_only", "kde_only", "unnorm_hist_only"],
        value="norm_hist+kde",
        style={"description_width": "42%"},
    ),
    "max_levels": dict(
        description="Max Levels: Maximum number of levels to display before condensing remaining into 'Other'",
        min=1,
        max=100,
        step=1,
        value=20,
        style={"description_width": "28%"},
    ),
    "flip_axis": dict(description="Flip Plot Orientation", value=True),
    "include_missing": dict(description="Add Missing Values Level", value=False),
    "bins": dict(
        description="# Hist Bins: Number of bins to use for the histogram (0 will use default bin count used by plotly)",
        min=0,
        max=5000,
        step=1,
        value=0,
        style={"description_width": "30%"},
    ),
    "num_intervals": dict(
        description="# Intervals: Number of intervals to discretize the numeric variable into",
        min=1,
        max=1000,
        step=1,
        value=4,
        style={"description_width": "30%"},
    ),
    "transform": dict(
        description=(
            "Transform: Transformation to apply to numeric column for plotting \n"
            "  - identity: No transformation\n"
            "  - log: Log transform (0 and negative values will be filtered)\n"
            "  - sqrt: Square root transform\n"
        ),
        value="identity",
        options=["identity", "log", "sqrt"],
        style={"description_width": "27%"},
    ),
    "transform1": dict(
        description=(
            "Transform1: Transformation to apply to column1 for plotting \n"
            "  - identity: No transformation\n"
            "  - log: Log transform (see clip variable for handling 0's)\n"
        ),
        value="identity",
        options=["identity", "log"],
        style={"description_width": "30%"},
    ),
    "transform2": dict(
        description=(
            "Transform2: Transformation to apply to column2 for plotting \n"
            "  - identity: No transformation\n"
            "  - log: Log transform (see clip variable for handling 0's)\n"
        ),
        value="identity",
        options=["identity", "log"],
        style={"description_width": "30%"},
    ),
    "lower_quantile": dict(
        description=(
            "Lower Quantile: Remove values below the provided quantile for a numeric column. Use to remove outliers in data."
        ),
        value=0,
        min=0,
        max=1,
        step=0.0001,
        style={"description_width": "40%"},
    ),
    "upper_quantile": dict(
        description=(
            "Upper Quantile: Remove values above the provided quantile for a numeric column. Use to remove outliers in data."
        ),
        value=1,
        min=0,
        max=1,
        step=0.0001,
        style={"description_width": "40%"},
    ),
    "lower_quantile1": dict(
        description=(
            "Lower Quantile1: Remove values below the provided quantile for column1. Use to remove outliers in data."
        ),
        value=0,
        min=0,
        max=1,
        step=0.0001,
        style={"description_width": "40%"},
    ),
    "upper_quantile1": dict(
        description=(
            "Upper Quantile1: Remove values above the provided quantile for column1. Use to remove outliers in data."
        ),
        value=1,
        min=0,
        max=1,
        step=0.0001,
        style={"description_width": "40%"},
    ),
    "lower_quantile2": dict(
        description=(
            "Lower Quantile2: Remove values below the provided quantile for column2. Use to remove outliers in data."
        ),
        value=0,
        min=0,
        max=1,
        step=0.0001,
        style={"description_width": "40%"},
    ),
    "upper_quantile2": dict(
        description=(
            "Upper Quantile2: Remove values above the provided quantile for column2. Use to remove outliers in data."
        ),
        value=1,
        min=0,
        max=1,
        step=0.0001,
        style={"description_width": "40%"},
    ),
    "ts_freq": dict(
        description=(
            "Aggregation Frequency: Frequency at which to aggregate counts. Can either be a quantity followed by"
            " a time unit (i.e. 6 months) or a pandas frequency string."
        ),
        value="auto",
        style={"description_width": "52%"},
    ),
    "trend_line": dict(
        description=(
            "Trend Line: Trend line to plot over data. 'none' will plot no trend line."
            "Other options are passed to plotnine's geom_smooth."
        ),
        value="auto",
        options=["auto", "none", "loess", "lm"],
        style={"description_width": "28%"},
    ),
    "ts_type": dict(
        description="Time Series Type: 'lines', 'markers', or 'lines+markers' to plot a line, points, or line + points",
        options=["lines", "markers", "lines+markers"],
        value="lines+markers",
        style={"description_width": "40%"},
    ),
    "top_ngrams": dict(
        description="Top Ngrams: Maximum number of bars to plot for ngrams",
        style={"description_width": "29%"},
        min=1,
        max=100,
        step=1,
        value=10,
    ),
    "remove_punct": dict(description="Remove punctuation tokens", value=True),
    "remove_stop": dict(description="Remove stop word tokens", value=True),
    "lower_case": dict(description="Lower case tokens", value=True),
    "compute_ngrams": dict(description="Plot most common ngrams", value=True),
    "top_entries": dict(
        description="Max Entries: Maximum number of bars to show for barplots",
        style={"description_width": "25%"},
        min=1,
        max=100,
        step=1,
        value=10,
    ),
    "opacity": dict(
        description="Opacity: Amount of transparency to use ranging from 0 (fully transparent) to 1 (opaque)",
        style={"description_width": "20%"},
        min=0,
        max=1,
        step=0.05,
        value=1,
    ),
    "sort_collections": dict(description="Sort Collections", value=True),
    "remove_duplicates": dict(description="Remove Duplicate Entries", value=True),
    "barmode": dict(
        description=(
            "Bar Mode: Manner in which to display levels of color variable in plotly Bar chart"
        ),
        options=["stack", "group", "overlay", "relative"],
        value="stack",
        style={"description_width": "28%"},
    ),
    "interval_type": dict(
        description=(
            "Interval Type: Type of interval to segment numeric variable into. Either quantile or equal width."
        ),
        options=["quantile", "equal_width"],
        value="quantile",
        style={"description_width": "33%"},
    ),
}
