import pandas as pd
import seaborn as sns
from plotnine import *
import ipywidgets as widgets
import warnings

from .config import WIDGET_VALUES, FLIP_LEVEL_COUNT, BAR_COLOR
from .data_utils import detect_column_type, coerce_column_type, freedman_diaconis_bins
from .univariate_summaries import *
from .bivariate_eda import *


def univariate_eda_interact(data):
    pd.set_option("precision", 2)
    sns.set(style="whitegrid")
    theme_set(theme_bw())
    warnings.simplefilter("ignore")

    widget = widgets.interactive(
        column_univariate_eda_interact,
        data=widgets.fixed(data),
        column=data.columns,
        col_type=WIDGET_VALUES["col_type"]["widget_options"],
    )
    widget.layout = widgets.Layout(flex_flow="row wrap")
    for ch in widget.children:
        if hasattr(ch, "description") and ch.description in WIDGET_VALUES:
            ch.style = {"description_width": WIDGET_VALUES[ch.description]["width"]}
            ch.description = WIDGET_VALUES[ch.description]["description"]

    def match_type(*args):
        type_widget.value = detect_column_type(data[col_widget.value])

    col_widget = widget.children[0]
    type_widget = widget.children[1]
    col_widget.observe(match_type, "value")
    type_widget.value = detect_column_type(data[data.columns[0]])

    display(widget)


def column_univariate_eda_interact(
    data, column, col_type="discrete", manual_update=False
):
    data = data.copy()

    data[column] = coerce_column_type(data[column], col_type)
    print("Plot Controls:")

    if col_type == "discrete":
        widget = widgets.interactive(
            discrete_univariate_summary,
            {"manual": manual_update},
            data=widgets.fixed(data),
            column=widgets.fixed(column),
            fig_height=widgets.IntSlider(
                min=1,
                max=50,
                step=1,
                value=max(6, min(data[column].nunique(), 24) // 2),
            ),
            fig_width=widgets.IntSlider(min=1, max=50, step=1, value=12),
            order=WIDGET_VALUES["level_order"]["widget_options"],
            max_levels=widgets.IntSlider(min=1, max=100, step=1, value=30),
            label_counts=widgets.Checkbox(True),
            percent_axis=widgets.Checkbox(True),
            label_fontsize=widgets.FloatSlider(min=1, max=30, step=0.5, value=12),
            include_missing=widgets.Checkbox(False),
            flip_axis=widgets.Checkbox(data[column].nunique() > 5),
            label_rotation=widgets.IntSlider(min=0, max=90, step=1, value=0),
            interactive=fixed(True),
        )
    elif col_type == "continuous":
        widget = interactive(
            continuous_univariate_summary,
            {"manual": manual_update},
            data=fixed(data),
            column=fixed(column),
            fig_height=widgets.IntSlider(
                min=1,
                max=50,
                step=1,
                value=6,
            ),
            fig_width=widgets.IntSlider(min=1, max=50, step=1, value=12),
            bins=widgets.IntSlider(
                min=1,
                max=1000,
                step=1,
                value=freedman_diaconis_bins(
                    data[~data[column].isna()][column], log=False
                ),
            ),
            clip=widgets.FloatText(value=0),
            transform=WIDGET_VALUES["transform"]["widget_options"],
            lower_quantile=widgets.FloatSlider(min=0, max=1, step=0.01, value=0),
            upper_quantile=widgets.FloatSlider(min=0, max=1, step=0.01, value=1),
            kde=widgets.Checkbox(False),
            interactive=fixed(True),
        )
    # datetime variables
    elif col_type == "datetime":
        print(
            "See here for valid frequency strings: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects"
        )
        widget = interactive(
            datetime_univariate_summary,
            {"manual": manual_update},
            data=fixed(data),
            column=fixed(column),
            fig_height=widgets.IntSlider(
                min=1,
                max=50,
                step=1,
                value=6,
            ),
            label_counts=widgets.Checkbox(True),
            fig_width=widgets.IntSlider(min=1, max=50, step=1, value=12),
            ts_type=widgets.Select(
                options=["point", "line"],
                value="point",
                description="Plot Type:",
                disabled=False,
            ),
            ts_freq="auto",
            delta_freq="auto",
            trend_line="auto",
            interactive=fixed(True),
        )
    elif col_type == "text":
        widget = interactive(
            text_univariate_eda,
            {"manual": manual_update},
            data=fixed(data),
            column=fixed(column),
            fig_height=WIDGET_VALUES["fig_height"]["widget_options"],
            fig_width=WIDGET_VALUES["fig_width"]["widget_options"],
            hist_bins=WIDGET_VALUES["hist_bins"]["widget_options"],
            lower_quantile=WIDGET_VALUES["lower_quantile"]["widget_options"],
            upper_quantile=WIDGET_VALUES["upper_quantile"]["widget_options"],
            transform=WIDGET_VALUES["transform"]["widget_options"],
            top_n=WIDGET_VALUES["top_n"]["widget_options"],
        )
    elif col_type == "list":
        widget = interactive(
            list_univariate_eda,
            {"manual": manual_update},
            data=fixed(data),
            column=fixed(column),
            fig_height=WIDGET_VALUES["fig_height"]["widget_options"],
            fig_width=WIDGET_VALUES["fig_width"]["widget_options"],
            top_entries=WIDGET_VALUES["top_entries"]["widget_options"],
        )
    else:
        print("No EDA support for this variable type")
        return

    for ch in widget.children[:-1]:
        if hasattr(ch, "description") and ch.description in WIDGET_VALUES:
            ch.style = {"description_width": WIDGET_VALUES[ch.description]["width"]}
            ch.description = WIDGET_VALUES[ch.description]["description"]
    widget.update()
    controls = widgets.HBox(widget.children[:-1], layout=Layout(flex_flow="row wrap"))
    output = widget.children[-1]
    display(widgets.VBox([controls, output]))
