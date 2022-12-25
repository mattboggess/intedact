import json
import os
import warnings

import ipywidgets as widgets
import pandas as pd
from IPython.display import display

from intedact import bivariate_summaries
from intedact.config import WIDGET_PARAMS
from intedact.utils import coerce_column_type, detect_column_type


def bivariate_eda_interact(
    data,
    notes_file: str = None,
):
    warnings.simplefilter("ignore")

    col1_widget = widgets.Dropdown(**WIDGET_PARAMS["column1"])
    col1_widget.options = data.columns
    col2_widget = widgets.Dropdown(**WIDGET_PARAMS["column2"])
    col2_widget.options = data.columns

    widget = widgets.widgets.interactive(
        column_bivariate_eda_interact,
        data=widgets.widgets.fixed(data),
        column1=col1_widget,
        column2=col2_widget,
        summary_type=widgets.Dropdown(**WIDGET_PARAMS["bivariate_summary_type"]),
        auto_update=widgets.Checkbox(**WIDGET_PARAMS["auto_update"]),
        notes_file=widgets.fixed(notes_file),
    )

    widget.layout = widgets.widgets.Layout(flex_flow="row wrap")

    def match_type(*args):
        col1_type = detect_column_type(data[col1_widget.value], discrete_limit=10)
        col2_type = detect_column_type(data[col2_widget.value], discrete_limit=10)
        type_widget.value = f"{col1_type}-{col2_type}"

    col1_widget = widget.children[0]
    col1_widget.value = data.columns[0]
    col2_widget = widget.children[1]
    col2_widget.value = data.columns[1]
    type_widget = widget.children[2]
    col1_widget.observe(match_type, "value")
    col2_widget.observe(match_type, "value")
    col1_type = detect_column_type(data[data.columns[0]], discrete_limit=10)
    col2_type = detect_column_type(data[data.columns[1]], discrete_limit=10)
    if f"{col1_type}-{col2_type}" in WIDGET_PARAMS["bivariate_summary_type"]["options"]:
        type_widget.value = f"{col1_type}-{col2_type}"
    else:
        type_widget.value = "unsupported"

    display(widget)


def column_bivariate_eda_interact(
    data: pd.DataFrame,
    column1: str,
    column2: str,
    summary_type: str = None,
    auto_update: bool = True,
    notes_file: str = None,
) -> None:
    data = data.copy()

    data[column1] = coerce_column_type(data[column1], summary_type)
    data[column2] = coerce_column_type(data[column2], summary_type)

    if column1 == column2:
        print("X and Y columns must be different")
        return

    if summary_type == "numeric-numeric":
        fig_width_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_width"])
        fig_height_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_height"])
        fig_height_widget.value = fig_width_widget.value
        widget = widgets.interactive(
            bivariate_summaries.numeric_numeric_summary,
            {"manual": not auto_update},
            data=widgets.fixed(data),
            column1=widgets.fixed(column1),
            column2=widgets.fixed(column2),
            fig_height=fig_height_widget,
            fig_width=fig_width_widget,
            opacity=widgets.FloatSlider(**WIDGET_PARAMS["opacity"]),
            trend_line=widgets.Dropdown(**WIDGET_PARAMS["trend_line"]),
            hist_bins=widgets.IntSlider(**WIDGET_PARAMS["bins"]),
            lower_quantile1=widgets.BoundedFloatText(
                **WIDGET_PARAMS["lower_quantile1"]
            ),
            upper_quantile1=widgets.BoundedFloatText(
                **WIDGET_PARAMS["upper_quantile1"]
            ),
            lower_quantile2=widgets.BoundedFloatText(
                **WIDGET_PARAMS["lower_quantile2"]
            ),
            upper_quantile2=widgets.BoundedFloatText(
                **WIDGET_PARAMS["upper_quantile2"]
            ),
            transform1=widgets.Dropdown(**WIDGET_PARAMS["transform1"]),
            transform2=widgets.Dropdown(**WIDGET_PARAMS["transform2"]),
            num_intervals=widgets.IntSlider(**WIDGET_PARAMS["num_intervals"]),
            interval_type=widgets.Dropdown(**WIDGET_PARAMS["interval_type"]),
            display_figure=widgets.fixed(True),
        )
    elif summary_type == "categorical-categorical":
        widget = widgets.interactive(
            bivariate_summaries.categorical_categorical_summary,
            {"manual": not auto_update},
            data=widgets.fixed(data),
            column1=widgets.fixed(column1),
            column2=widgets.fixed(column2),
            fig_height=widgets.IntSlider(**WIDGET_PARAMS["fig_height"]),
            fig_width=widgets.IntSlider(**WIDGET_PARAMS["fig_width"]),
            order1=widgets.Dropdown(**WIDGET_PARAMS["order"]),
            order2=widgets.Dropdown(**WIDGET_PARAMS["order"]),
            barmode=widgets.Dropdown(**WIDGET_PARAMS["barmode"]),
            max_levels=widgets.IntSlider(**WIDGET_PARAMS["max_levels"]),
            include_missing=widgets.Checkbox(**WIDGET_PARAMS["include_missing"]),
            display_figure=widgets.fixed(True),
        )
    elif summary_type == "numeric-categorical":
        fig_height_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_height"])
        fig_width_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_width"])
        fig_height_widget.value = int(fig_width_widget.value * 0.5)
        widget = widgets.interactive(
            bivariate_summaries.numeric_categorical_summary,
            {"manual": not auto_update},
            data=widgets.fixed(data),
            column1=widgets.fixed(column1),
            column2=widgets.fixed(column2),
            fig_height=fig_height_widget,
            fig_width=fig_width_widget,
            order=widgets.Dropdown(**WIDGET_PARAMS["order"]),
            num_intervals=widgets.IntSlider(**WIDGET_PARAMS["num_intervals"]),
            interval_type=widgets.Dropdown(**WIDGET_PARAMS["interval_type"]),
            max_levels=widgets.IntSlider(**WIDGET_PARAMS["max_levels"]),
            include_missing=widgets.Checkbox(**WIDGET_PARAMS["include_missing"]),
            display_figure=widgets.fixed(True),
        )
    elif summary_type == "categorical-numeric":
        fig_height_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_height"])
        fig_height_widget.value = 800
        dist_type_widget = widgets.Dropdown(**WIDGET_PARAMS["dist_type"])
        if data[column1].nunique() > 3:
            dist_type_widget.value = "kde_only"
        widget = widgets.interactive(
            bivariate_summaries.categorical_numeric_summary,
            {"manual": not auto_update},
            data=widgets.fixed(data),
            column1=widgets.fixed(column1),
            column2=widgets.fixed(column2),
            fig_height=fig_height_widget,
            fig_width=widgets.IntSlider(**WIDGET_PARAMS["fig_width"]),
            order=widgets.Dropdown(**WIDGET_PARAMS["order"]),
            max_levels=widgets.IntSlider(**WIDGET_PARAMS["max_levels"]),
            include_missing=widgets.Checkbox(**WIDGET_PARAMS["include_missing"]),
            dist_type=dist_type_widget,
            lower_quantile=widgets.BoundedFloatText(**WIDGET_PARAMS["lower_quantile"]),
            upper_quantile=widgets.BoundedFloatText(**WIDGET_PARAMS["upper_quantile"]),
            transform=widgets.Dropdown(**WIDGET_PARAMS["transform"]),
            hist_bins=widgets.IntSlider(**WIDGET_PARAMS["bins"]),
            display_figure=widgets.fixed(True),
        )
    else:
        col1_type = detect_column_type(data[column1], discrete_limit=10)
        col2_type = detect_column_type(data[column2], discrete_limit=10)
        print(f"No EDA support for bivariate summary type: {col1_type}-{col2_type}")
        return

    print("==============")
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
            notes = {f"{column1}-{column2}": ""}
        else:
            with open(notes_file, "r") as fid:
                notes = json.load(fid)
            if f"{column1}-{column2}" not in notes:
                notes[f"{column1}-{column2}"] = ""

        notes_entry = widgets.Textarea(
            value=notes[f"{column1}-{column2}"],
            placeholder="Take EDA Notes Here",
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
                notes[f"{column1}-{column2}"] = notes_entry.value
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
