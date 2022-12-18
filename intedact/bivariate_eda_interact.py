import json
import os
import warnings

import ipywidgets as widgets

from .bivariate_summaries import *
from .config import WIDGET_PARAMS
from .data_utils import coerce_column_type
from .data_utils import detect_column_type
from .data_utils import freedman_diaconis_bins


def bivariate_eda_interact(
    data,
    notes_file: str = None,
):
    pd.set_option("display.precision", 2)
    sns.set(style="whitegrid")
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
    col2_widget = widget.children[1]
    col2_widget.value = data.columns[1]
    type_widget = widget.children[2]
    col1_widget.observe(match_type, "value")
    col2_widget.observe(match_type, "value")
    col1_type = detect_column_type(data[data.columns[0]], discrete_limit=10)
    col2_type = detect_column_type(data[data.columns[1]], discrete_limit=10)
    type_widget.value = f"{col1_type}-{col2_type}"

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

    color_palette_widget = widgets.Text(**WIDGET_PARAMS["color_palette"])
    if column1 == column2:
        print("X and Y columns must be different")
        return

    if summary_type == "numeric-numeric":
        fig_height_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_height"])
        fig_height_widget.value = 8
        fig_width_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_width"])
        fig_width_widget.value = fig_height_widget.value
        bins_widget = widgets.IntSlider(**WIDGET_PARAMS["bins"])
        bins_widget.value = max(
            freedman_diaconis_bins(data[column1]), freedman_diaconis_bins(data[column2])
        )
        widget = widgets.interactive(
            numeric_numeric_bivariate_summary,
            {"manual": not auto_update},
            data=widgets.fixed(data),
            column1=widgets.fixed(column1),
            column2=widgets.fixed(column2),
            fig_height=fig_height_widget,
            fig_width=fig_width_widget,
            fontsize=widgets.FloatSlider(**WIDGET_PARAMS["fontsize"]),
            color_palette=color_palette_widget,
            plot_type=widgets.Dropdown(**WIDGET_PARAMS["numeric2d_plot_type"]),
            trend_line=widgets.Dropdown(**WIDGET_PARAMS["trend_line"]),
            bins=bins_widget,
            alpha=widgets.FloatSlider(**WIDGET_PARAMS["alpha"]),
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
            clip=widgets.BoundedFloatText(**WIDGET_PARAMS["clip"]),
            reference_line=widgets.Checkbox(**WIDGET_PARAMS["reference_line"]),
            plot_density=widgets.Checkbox(**WIDGET_PARAMS["plot_kde"]),
            match_axes=widgets.Checkbox(**WIDGET_PARAMS["match_axes"]),
            interactive=widgets.fixed(True),
        )
    elif summary_type == "categorical-categorical":
        fig_height_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_height"])
        fig_height_widget.value = 1000
        fig_width_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_width"])
        fig_width_widget.value = fig_height_widget.value
        widget = widgets.interactive(
            categorical_categorical_summary,
            {"manual": not auto_update},
            data=widgets.fixed(data),
            column1=widgets.fixed(column1),
            column2=widgets.fixed(column2),
            fig_height=fig_height_widget,
            fig_width=fig_width_widget,
            fontsize=widgets.FloatSlider(**WIDGET_PARAMS["fontsize"]),
            order1=widgets.Dropdown(**WIDGET_PARAMS["order"]),
            order2=widgets.Dropdown(**WIDGET_PARAMS["order"]),
            barmode=widgets.Dropdown(**WIDGET_PARAMS["barmode"]),
            max_levels=widgets.IntSlider(**WIDGET_PARAMS["max_levels"]),
            include_missing=widgets.Checkbox(**WIDGET_PARAMS["include_missing"]),
        )
    elif summary_type == "numeric-categorical":
        fig_height_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_height"])
        fig_height_widget.value = 500
        fig_width_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_width"])
        fig_width_widget.value = fig_height_widget.value * 2
        widget = widgets.interactive(
            numeric_categorical_summary,
            {"manual": not auto_update},
            data=widgets.fixed(data),
            column1=widgets.fixed(column1),
            column2=widgets.fixed(column2),
            fig_height=fig_height_widget,
            fig_width=fig_width_widget,
            order=widgets.Dropdown(**WIDGET_PARAMS["order"]),
            bins=widgets.IntSlider(**WIDGET_PARAMS["quantile_bins"]),
            bin_type=widgets.Dropdown(**WIDGET_PARAMS["bin_type"]),
            max_levels=widgets.IntSlider(**WIDGET_PARAMS["max_levels"]),
            include_missing=widgets.Checkbox(**WIDGET_PARAMS["include_missing"]),
        )
    elif summary_type == "categorical-numeric":
        fig_height_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_height"])
        fig_height_widget.value = 500
        fig_width_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_width"])
        fig_width_widget.value = fig_height_widget.value * 2
        widget = widgets.interactive(
            categorical_numeric_summary,
            {"manual": not auto_update},
            data=widgets.fixed(data),
            column1=widgets.fixed(column1),
            column2=widgets.fixed(column2),
            fig_height=fig_height_widget,
            fig_width=fig_width_widget,
            order=widgets.Dropdown(**WIDGET_PARAMS["order"]),
            max_levels=widgets.IntSlider(**WIDGET_PARAMS["max_levels"]),
            include_missing=widgets.Checkbox(**WIDGET_PARAMS["include_missing"]),
            lower_quantile=widgets.BoundedFloatText(**WIDGET_PARAMS["lower_quantile"]),
            upper_quantile=widgets.BoundedFloatText(**WIDGET_PARAMS["upper_quantile"]),
            transform=widgets.Dropdown(**WIDGET_PARAMS["transform"]),
        )
    else:
        print("No EDA support for this variable type")
        return

    print("=====================")
    print("General Plot Controls")
    print("=====================")
    general_controls = widgets.HBox(
        widget.children[:4], layout=widgets.Layout(flex_flow="row wrap")
    )
    display(general_controls)

    print("=========================")
    print("Summary Specific Controls")
    print("=========================")
    widget.update()

    controls = widgets.HBox(
        widget.children[4:-1], layout=widgets.Layout(flex_flow="row wrap")
    )
    display(controls)

    print("==============")
    print("Summary Output")
    print("==============")
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
