import json
import os
import warnings

import ipywidgets as widgets
import pandas as pd
from IPython.display import display

from intedact import univariate_summaries
from intedact.config import FLIP_LEVEL_COUNT, WIDGET_PARAMS
from intedact.utils import coerce_column_type, detect_column_type


def univariate_eda_interact(
    data,
    notes_file: str = None,
):
    warnings.simplefilter("ignore")

    col_widget = widgets.Dropdown(**WIDGET_PARAMS["column"])
    col_widget.options = data.columns
    col_widget.value = data.columns[0]

    widget = widgets.widgets.interactive(
        column_univariate_eda_interact,
        data=widgets.widgets.fixed(data),
        column=col_widget,
        summary_type=widgets.Dropdown(**WIDGET_PARAMS["univariate_summary_type"]),
        auto_update=widgets.Checkbox(**WIDGET_PARAMS["auto_update"]),
        notes_file=widgets.fixed(notes_file),
    )

    widget.layout = widgets.widgets.Layout(flex_flow="row wrap")

    def match_type(*args):
        summary_type = detect_column_type(data[col_widget.value])
        type_widget.value = summary_type
        auto_widget.value = True  # summary_type not in ["text"]

    col_widget = widget.children[0]
    type_widget = widget.children[1]
    auto_widget = widget.children[2]
    col_widget.observe(match_type, "value")
    type_widget.value = detect_column_type(data[data.columns[0]])

    display(widget)


def column_univariate_eda_interact(
    data: pd.DataFrame,
    column: str,
    summary_type: str = "categorical",
    auto_update: bool = True,
    notes_file: str = None,
) -> None:
    data = data.copy()

    data[column] = coerce_column_type(data[column], summary_type)

    if summary_type == "categorical":
        fig_height_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_height"])
        fig_height_widget.value = max(600, 100 * min(data[column].nunique(), 24) // 2)
        flip_axis_widget = widgets.Checkbox(**WIDGET_PARAMS["flip_axis"])
        flip_axis_widget.value = data[column].nunique() > FLIP_LEVEL_COUNT
        widget = widgets.interactive(
            univariate_summaries.categorical_summary,
            {"manual": not auto_update},
            data=widgets.widgets.fixed(data),
            column=widgets.widgets.fixed(column),
            fig_height=fig_height_widget,
            fig_width=widgets.IntSlider(**WIDGET_PARAMS["fig_width"]),
            order=widgets.Dropdown(**WIDGET_PARAMS["order"]),
            max_levels=widgets.IntSlider(**WIDGET_PARAMS["max_levels"]),
            include_missing=widgets.Checkbox(**WIDGET_PARAMS["include_missing"]),
            flip_axis=flip_axis_widget,
            display_figure=widgets.fixed(True),
        )
    elif summary_type == "numeric":
        bins_widget = widgets.IntSlider(**WIDGET_PARAMS["bins"])
        height_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_height"])
        height_widget.value = 600
        lower_quantile_widget = widgets.BoundedFloatText(
            **WIDGET_PARAMS["lower_quantile"]
        )
        upper_quantile_widget = widgets.BoundedFloatText(
            **WIDGET_PARAMS["upper_quantile"]
        )
        widget = widgets.interactive(
            univariate_summaries.numeric_summary,
            {"manual": not auto_update},
            data=widgets.fixed(data),
            column=widgets.fixed(column),
            fig_height=height_widget,
            fig_width=widgets.IntSlider(**WIDGET_PARAMS["fig_width"]),
            bins=bins_widget,
            lower_quantile=lower_quantile_widget,
            upper_quantile=upper_quantile_widget,
            transform=widgets.Dropdown(**WIDGET_PARAMS["transform"]),
            display_figure=widgets.fixed(True),
        )
    elif summary_type == "datetime":
        print(
            "See here for valid frequency strings: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects"
        )
        lower_quantile_widget = widgets.BoundedFloatText(
            **WIDGET_PARAMS["lower_quantile"]
        )
        upper_quantile_widget = widgets.BoundedFloatText(
            **WIDGET_PARAMS["upper_quantile"]
        )
        widget = widgets.interactive(
            univariate_summaries.datetime_summary,
            {"manual": not auto_update},
            data=widgets.fixed(data),
            column=widgets.fixed(column),
            fig_height=widgets.IntSlider(**WIDGET_PARAMS["fig_height"]),
            fig_width=widgets.IntSlider(**WIDGET_PARAMS["fig_width"]),
            ts_freq=widgets.Text(**WIDGET_PARAMS["ts_freq"]),
            ts_type=widgets.Dropdown(**WIDGET_PARAMS["ts_type"]),
            trend_line=widgets.Dropdown(**WIDGET_PARAMS["trend_line"]),
            lower_quantile=lower_quantile_widget,
            upper_quantile=upper_quantile_widget,
            display_figure=widgets.fixed(True),
        )
    elif summary_type == "text":
        fig_width_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_width"])
        widget = widgets.interactive(
            univariate_summaries.text_summary,
            {"manual": not auto_update},
            data=widgets.fixed(data),
            column=widgets.fixed(column),
            fig_height=widgets.IntSlider(**WIDGET_PARAMS["fig_height"]),
            fig_width=fig_width_widget,
            top_ngrams=widgets.IntSlider(**WIDGET_PARAMS["top_ngrams"]),
            lower_case=widgets.Checkbox(**WIDGET_PARAMS["lower_case"]),
            remove_punct=widgets.Checkbox(**WIDGET_PARAMS["remove_punct"]),
            remove_stop=widgets.Checkbox(**WIDGET_PARAMS["remove_stop"]),
            display_figure=widgets.fixed(True),
        )
    elif summary_type == "collection":
        fig_width_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_width"])
        widget = widgets.interactive(
            univariate_summaries.collection_summary,
            {"manual": not auto_update},
            data=widgets.fixed(data),
            column=widgets.fixed(column),
            fig_height=widgets.IntSlider(**WIDGET_PARAMS["fig_height"]),
            fig_width=fig_width_widget,
            top_entries=widgets.IntSlider(**WIDGET_PARAMS["top_entries"]),
            remove_duplicates=widgets.Checkbox(**WIDGET_PARAMS["remove_duplicates"]),
            sort_collections=widgets.Checkbox(**WIDGET_PARAMS["sort_collections"]),
            display_figure=widgets.fixed(True),
        )
    elif summary_type == "url":
        fig_height_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_height"])
        fig_width_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_width"])
        widget = widgets.interactive(
            univariate_summaries.url_summary,
            {"manual": not auto_update},
            data=widgets.fixed(data),
            column=widgets.fixed(column),
            fig_height=fig_height_widget,
            fig_width=fig_width_widget,
            top_entries=widgets.IntSlider(**WIDGET_PARAMS["top_entries"]),
            display_figure=widgets.fixed(True),
        )
    else:
        print("No EDA support for this variable type")
        return

    print("=============")
    print("Plot Controls")
    print("==============")
    plot_controls = widgets.HBox(
        widget.children[:-1], layout=widgets.Layout(flex_flow="row wrap")
    )
    display(plot_controls)
    widget.update()

    output = widget.children[-1]
    display(output)

    # Handle EDA notes and summary configuration
    if notes_file is not None:
        info_button = widgets.Button(
            description="Save Notes",
            disabled=False,
            button_style="info",
            icon="save",
            layout=widgets.widgets.Layout(width="15%", height="50px"),
            tooltip=f"Save EDA notes to {notes_file}",
        )
        if not os.path.exists(notes_file):
            notes = {column: ""}
        else:
            with open(notes_file, "r") as fid:
                notes = json.load(fid)
            if column not in notes:
                notes[column] = ""

        notes_entry = widgets.Textarea(
            value=notes[column],
            placeholder="Take EDA Notes Here. Make sure to click Save Notes before navigating away.",
            description="EDA Notes:",
            layout=dict(width="80%", height="auto"),
            disabled=False,
        )
        info_button.description = "Save Notes"

        def savenotes_on_click(x):
            if notes_file is not None:
                if not os.path.exists(notes_file):
                    notes = {}
                else:
                    with open(notes_file, "r") as fid:
                        notes = json.load(fid)
                notes[column] = notes_entry.value
                with open(notes_file, "w") as fid:
                    json.dump(notes, fid)

        info_button.on_click(savenotes_on_click)

        display(
            widgets.HBox(
                [notes_entry, info_button],
                layout=widgets.widgets.Layout(height="200px"),
            ),
            layout=widgets.Layout(flex_flow="row wrap"),
        )
