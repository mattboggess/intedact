import pandas as pd
import seaborn as sns
from plotnine import *
import ipywidgets as widgets
import warnings

from .config import WIDGET_VALUES, FLIP_LEVEL_COUNT, BAR_COLOR
from .utils import detect_column_type, coerce_column_type
from .univariate_summaries import *
from .bivariate_eda import *

def univariate_eda_interact(data):
    pd.set_option('precision', 2)
    sns.set(style='whitegrid')
    theme_set(theme_bw())
    warnings.simplefilter("ignore")

    widget = widgets.interactive(
        column_univariate_eda_interact,
        data=widgets.fixed(data),
        column=data.columns,
        col_type=WIDGET_VALUES['col_type']['widget_options']
    )
    widget.layout = widgets.Layout(flex_flow='row wrap')
    for ch in widget.children:
        if hasattr(ch, 'description') and ch.description in WIDGET_VALUES:
            ch.style = {'description_width': WIDGET_VALUES[ch.description]['width']}
            ch.description = WIDGET_VALUES[ch.description]['description']

    def match_type(*args):
        type_widget.value = detect_column_type(data[col_widget.value])

    col_widget = widget.children[0]
    type_widget = widget.children[1]
    col_widget.observe(match_type, 'value')
    type_widget.value = detect_column_type(data[data.columns[0]])

    display(widget)


def column_univariate_eda_interact(data, column, col_type='discrete', manual_update=False):
    data = data.copy()

    data[column] = coerce_column_type(data[column], col_type)
    print('Plot Controls:')

    if col_type == 'discrete':
        if data[column].nunique() > FLIP_LEVEL_COUNT:
            flip_axis_default = True
        else:
            flip_axis_default = False
        widget = widgets.interactive(
            discrete_univariate_summary,
            {'manual': manual_update},
            data=widgets.fixed(data),
            column=widgets.fixed(column),
            fig_height=WIDGET_VALUES['fig_height']['widget_options'],
            fig_width=WIDGET_VALUES['fig_width']['widget_options'],
            level_order=WIDGET_VALUES['level_order']['widget_options'],
            max_levels=WIDGET_VALUES['max_levels']['widget_options'],
            label_counts=widgets.Checkbox(True),
            flip_axis=widgets.Checkbox(flip_axis_default),
            bar_color=fixed(BAR_COLOR),
            label_rotation=WIDGET_VALUES['label_rotation']['widget_options'],
            interactive=fixed(True)
        )
    elif col_type == 'continuous':
        widget = interactive(
            continuous_univariate_summary,
            {'manual': manual_update},
            data=fixed(data),
            column=fixed(column),
            fig_height=WIDGET_VALUES['fig_height']['widget_options'],
            fig_width=WIDGET_VALUES['fig_width']['widget_options'],
            hist_bins=WIDGET_VALUES['hist_bins']['widget_options'],
            transform=WIDGET_VALUES['transform']['widget_options'],
            lower_quantile=WIDGET_VALUES['lower_quantile']['widget_options'],
            upper_quantile=WIDGET_VALUES['upper_quantile']['widget_options'],
            kde=widgets.Checkbox(False),
            bar_color=fixed(BAR_COLOR),
            interactive=fixed(True)
        )
    # datetime variables
    elif col_type == 'datetime':
        print(
            "See here for valid frequency strings: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects")
        widget = interactive(
            datetime_univariate_eda,
            {'manual': manual_update},
            data=fixed(data),
            column=fixed(column),
            fig_height=WIDGET_VALUES['fig_height']['widget_options'],
            fig_width=WIDGET_VALUES['fig_width']['widget_options'],
            hist_bins=WIDGET_VALUES['hist_bins']['widget_options'],
            lower_quantile=WIDGET_VALUES['lower_quantile']['widget_options'],
            upper_quantile=WIDGET_VALUES['upper_quantile']['widget_options'],
            transform=WIDGET_VALUES['transform']['widget_options']
        )
    elif col_type == 'text':
        widget = interactive(
            text_univariate_eda,
            {'manual': manual_update},
            data=fixed(data),
            column=fixed(column),
            fig_height=WIDGET_VALUES['fig_height']['widget_options'],
            fig_width=WIDGET_VALUES['fig_width']['widget_options'],
            hist_bins=WIDGET_VALUES['hist_bins']['widget_options'],
            lower_quantile=WIDGET_VALUES['lower_quantile']['widget_options'],
            upper_quantile=WIDGET_VALUES['upper_quantile']['widget_options'],
            transform=WIDGET_VALUES['transform']['widget_options'],
            top_n=WIDGET_VALUES['top_n']['widget_options']
        )
    elif col_type == 'list':
        widget = interactive(
            list_univariate_eda,
            {'manual': manual_update},
            data=fixed(data),
            column=fixed(column),
            fig_height=WIDGET_VALUES['fig_height']['widget_options'],
            fig_width=WIDGET_VALUES['fig_width']['widget_options'],
            top_entries=WIDGET_VALUES['top_entries']['widget_options']
        )
    else:
        print("No EDA support for this variable type")
        return

    for ch in widget.children[:-1]:
        if hasattr(ch, 'description') and ch.description in WIDGET_VALUES:
            ch.style = {'description_width': WIDGET_VALUES[ch.description]['width']}
            ch.description = WIDGET_VALUES[ch.description]['description']
    widget.update()
    controls = widgets.HBox(widget.children[:-1], layout=Layout(flex_flow='row wrap'))
    output = widget.children[-1]
    display(widgets.VBox([controls, output]))

