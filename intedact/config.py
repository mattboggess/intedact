import ipywidgets as widgets

FLIP_LEVEL_COUNT = 5

TIME_UNITS = [
    "nanoseconds",
    "microseconds",
    "milliseconds",
    "seconds",
    "months",
    "hours",
    "days",
    "weeks",
    "months",
    "years",
]

# Column type groupings
DISCRETE_TYPES = [
    "discrete_numeric",
    "unordered_categorical",
    "ordered_categorical",
    "unordered_categorical (inferred from object)",
]

# Widget control values
WIDGET_PARAMS = {
    "manual_update": {
        "description": "Update plots manually",
        "width": "0%",
        "widget": "N/A",
    },
    "Run Interact": {
        "description": "Update Plot",
        "width": "0%",
        "widget": "N/A",
    },
    "column": dict(
        description="Column: Column to be plotted",
        style={"description_width": "22%"},
    ),
    "summary_type": dict(
        description=(
            "Summary Type: Type of univariate summary to display."
            " Type is automatically inferred by default, but you can manually change the type to"
            " try to produce a different summary if the column data is compatible."
            " It is recommended you preprocess the columns to have the desired type prior to"
            " running this function for greatest accuracy of automatic type inference.\n\n"
            " The following types are available:\n"
            "  - 'discrete': categorical or low dimensional numeric variables\n"
            "  - 'continuous': high dimensional numeric variables \n"
            "  - 'datetime': datetime columns (should be coerceable to datetime type) \n"
            "  - 'text': column with strings that represent longer freeform text \n"
            "  - 'list': column with variable length lists of items, should be pre-coerced such that elements are python lists/tuples/sets\n"
        ),
        style={"description_width": "36%"},
        options=["discrete", "continuous", "datetime", "text", "list"],
    ),
    "fig_width": dict(
        description="Figure Width: Width of figure in inches",
        min=1,
        max=50,
        step=1,
        value=12,
        style={"description_width": "31%"},
    ),
    "fig_height": dict(
        description="Figure Height: Height of figure in inches (multiplied if multiple subplots)",
        min=1,
        max=50,
        step=1,
        value=6,
        style={"description_width": "33%"},
    ),
    "fontsize": dict(
        description="Font Size: fontsize for axis and tick labels",
        min=1,
        max=40,
        step=0.5,
        value=12,
        style={"description_width": "25%"},
    ),
    "color_palette": dict(
        description="Color Palette:",
        placeholder="Passed to sns.set_palette()",
        value=None,
    ),
    "order": dict(
        description=(
            "Level Order: Order in which to sort the levels of the variable for plotting:\n"
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
    "max_levels": dict(
        description="Max Levels: Maximum number of levels to display before condensing remaining into 'Other'",
        min=1,
        max=100,
        step=1,
        value=30,
        style={"description_width": "28%"},
    ),
    "flip_axis": dict(description="Flip Plot Orientation", value=True),
    "label_counts": dict(description="Add Counts and Percentages", value=True),
    "percent_axis": dict(description="Add a Percentage Axis", value=True),
    "include_missing": dict(description="Plot Missing Values", value=False),
    "label_rotation": dict(
        description="Label Rotation: Degree to rotate axis labels (Ignored if axis is flipped)",
        min=0,
        max=90,
        step=1,
        value=0,
        style={"description_width": "35%"},
    ),
    "label_fontsize": dict(
        description="Label Font Size: Font size for the count and percentage annotations",
        min=1,
        max=30,
        step=0.5,
        value=12,
        style={"description_width": "38%"},
    ),
    "bins": dict(
        description="# Bins: Number of bins to use for the histogram",
        min=1,
        max=1000,
        step=1,
        value=100,
        style={"description_width": "20%"},
    ),
    "kde": dict(description="Overlay Density on Histogram", value=False),
    "transform": dict(
        description=(
            "Transform: Transformation to apply to column for plotting \n"
            "  - identity: No transformation\n"
            "  - log: Log transform (see clip variable for handling 0's)\n"
        ),
        value="identity",
        options=["identity", "log"],
        style={"description_width": "27%"},
    ),
    "clip": dict(
        description=(
            "Positive Clip: Clip values below this value to this value (filter > 0 if 0)."
            " Used for log transform."
        ),
        value=0,
        min=0,
        max=10,
        step=1e-6,
        style={"description_width": "31%"},
    ),
    "lower_trim": dict(
        description=(
            "Lower Trim: Remove X values from lower end of distribution. Use to remove outliers in data."
        ),
        value=0,
        min=0,
        max=10000,
        step=1,
        style={"description_width": "30%"},
    ),
    "upper_trim": dict(
        description=(
            "Upper Trim: Remove X values from upper end of distribution. Use to remove outliers in data."
        ),
        value=0,
        min=0,
        max=10000,
        step=1,
        style={"description_width": "30%"},
    ),
    "ts_freq": dict(
        description=(
            "Aggregation Frequency: Frequency at which to aggregate counts. Can either be a quantity followed by"
            " a time unit (i.e. 6 months) or a pandas frequency string."
        ),
        value="auto",
        style={"description_width": "52%"},
    ),
    "delta_units": dict(
        description=(
            "Time Delta Units: Units in which to report the time differences between successive observations."
        ),
        value="auto",
        options=["auto"] + TIME_UNITS,
        style={"description_width": "40%"},
    ),
    "trend_line": dict(
        description=(
            "Trend Line: Trend line to plot over data. 'none' will plot no trend line."
            "Other options are passed to plotnine's geom_smooth."
        ),
        value="auto",
        options=["auto", "none", "loess", "lm"],
        style={"description_width": "27%"},
    ),
    "span": dict(
        description=(
            "Loess Span: Span parameter to control loess trend line smoothing."
        ),
        min=0,
        max=1,
        value=0.75,
        style={"description_width": "32%"},
    ),
    "date_breaks": dict(
        description=(
            "Date Breaks: Frequency at which to add ticks to time series x axis. Format is"
            " a quantity followed time unit (i.e. 6 months)."
        ),
        value="auto",
        style={"description_width": "32%"},
    ),
    "date_labels": dict(
        description=(
            "Date Labels: Format for the date x axis labels. Format string passed to strftime"
        ),
        value="auto",
        style={"description_width": "30%"},
    ),
    "ts_type": dict(
        description="Time Series Type: 'point' plots a time series scatter plot and 'line' plots a line graph.",
        options=["point", "line"],
        value="line",
        style={"description_width": "40%"},
    ),
    "top_ngrams": {
        "description": "top ngrams: Maximum number of bars to plot for ngrams",
        "width": "29%",
        "widget": (1, 100, 1),
    },
    "remove_punct": {
        "description": "Ignore punctuation for ngrams",
        "width": "0%",
        "widget": "N/A",
    },
    "remove_stop": {
        "description": "Ignore stop words for ngrams",
        "width": "0%",
        "widget": "N/A",
    },
    "lower_case": {
        "description": "Lower case text for ngrams",
        "width": "0%",
        "widget": "N/A",
    },
    "plot_type_cc": {
        "description": (
            "plot type: Type of plot to show\n"
            "  - 'auto': Defaults to scatter plot\n"
            "  - 'scatter': Draw a scatter plot using geom_scatter\n"
            "  - 'bin2d': Draw a 2d histogram using geom_bin2d\n"
            "  - 'count': Draw a 2d count plot using geom_count"
        ),
        "width": "25%",
        "widget": ["auto", "scatter", "bin2d", "count"],
    },
    "plot_type_dc1": {
        "description": (
            "plot type: Type of plot to show\n"
            "  - 'auto': Defaults to bar plot\n"
            "  - 'bar': Draw a bar plot\n"
            "  - 'point': Draw a point plot\n"
        ),
        "width": "25%",
        "widget": ["auto", "bar", "point"],
    },
    "plot_type_dc2": {
        "description": (
            "plot type: Type of plot to show\n"
            "  - 'auto': Defaults to overlapping histograms \n"
            "  - 'histogram': Draw overlapping histograms \n"
            "  - 'density': Draw overlapping KDEs \n"
        ),
        "width": "25%",
        "widget": ["auto", "histogram", "density"],
    },
    "plot_type_dcn": {
        "description": (
            "plot type: Type of plot to show\n"
            "  - 'auto': Defaults to boxplot \n"
            "  - 'freqpoly': Draw overlapping densities/histograms as colored lines with geom_freqpoly \n"
            "  - 'boxplot': Draw boxplot per level of discrete variable \n"
            "  - 'violinplot': Draw violin per level of discrete variable \n"
            "  - 'facted_histogram': Draw faceted histograms stacked vertically in facets by level \n"
            "  - 'facted_density': Draw faceted densities stacked vertically in facets by level \n"
        ),
        "width": "25%",
        "widget": [
            "auto",
            "freqpoly",
            "boxplot",
            "violin",
            "faceted_histogram",
            "faceted_density",
        ],
    },
    "equalize_axes": {
        "description": "Match x and y axes",
        "width": "30%",
        "widget": "N/A",
    },
    "reference_line": {
        "description": "Plot a y = x reference line",
        "width": "0%",
        "widget": "N/A",
    },
    "plot_density": {
        "description": "Overlay a bivariate KDE",
        "width": "0%",
        "widget": "N/A",
    },
    "alpha": {
        "description": "alpha: Amount of transparency to use for points/histograms ranging from 0 (fully transparent) to 1 (opaque)",
        "width": "18%",
        "widget": (0, 1, 0.05),
    },
    "normalize": {
        "description": "Normalize counts to percentages",
        "width": "0%",
        "widget": "N/A",
    },
    "ref_lines": {
        "description": "Add mean & median reference lines",
        "width": "0%",
        "widget": "N/A",
    },
    "varwidth": {
        "description": "Scale boxplot by sample size",
        "width": "0%",
        "widget": "N/A",
    },
    "normalize_dist": {
        "description": "Normalize distributions to densities",
        "width": "0%",
        "widget": "N/A",
    },
}
