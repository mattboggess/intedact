# TODO: WIP
import json
import os
import warnings

import ipywidgets as widgets
import pandas as pd
import seaborn as sns

from .bivariate_summaries import *
from .config import FLIP_LEVEL_COUNT
from .config import WIDGET_PARAMS
from .data_utils import coerce_column_type
from .data_utils import detect_column_type
from .data_utils import freedman_diaconis_bins


def bivariate_eda_interact(
    data,
    figure_dir: str = None,
    notes_file: str = None,
    data_dict_file: str = None,
):
    pd.set_option("precision", 2)
    sns.set(style="whitegrid")
    warnings.simplefilter("ignore")

    if data_dict_file:
        with open(data_dict_file, "r") as fid:
            data_dict = json.load(fid)
    else:
        data_dict = None

    if figure_dir is not None:
        save_button = widgets.Button(
            description="Save Figure",
            disabled=False,
            button_style="info",
            icon="save",
            layout=widgets.widgets.Layout(width="20%", height="30px"),
            tooltip="Save summary figure to figure_dir/column.png if figure_dir is specified",
        )
    else:
        save_button = None

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
        data_dict=widgets.fixed(data_dict),
        notes_file=widgets.fixed(notes_file),
        figure_dir=widgets.fixed(figure_dir),
        savefig_button=widgets.fixed(save_button),
    )

    if figure_dir is not None:
        widget.children = list(widget.children[:-1]) + [
            save_button,
            widget.children[-1],
        ]

    widget.layout = widgets.widgets.Layout(flex_flow="row wrap")

    def match_type(*args):
        col1_type = detect_column_type(data[col1_widget.value], discrete_limit=10)
        col2_type = detect_column_type(data[col2_widget.value], discrete_limit=10)
        type_widget.value = f"{col1_type}-{col2_type}"
        # auto_widget.value = f"{col1_type}-{col2_type}" not in ["text"]

    col1_widget = widget.children[0]
    col2_widget = widget.children[1]
    col2_widget.value = data.columns[1]
    type_widget = widget.children[2]
    auto_widget = widget.children[3]
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
    data_dict: str = None,
    notes_file: str = None,
    figure_dir: str = None,
    savefig_button: Optional[widgets.Button] = None,
) -> None:
    data = data.copy()

    data[column1] = coerce_column_type(data[column1], summary_type)
    data[column2] = coerce_column_type(data[column2], summary_type)

    color_palette_widget = widgets.Text(**WIDGET_PARAMS["color_palette"])
    if summary_type == "numeric-numeric":
        lower_trim1_widget = widgets.BoundedIntText(**WIDGET_PARAMS["lower_trim1"])
        lower_trim1_widget.max = data.shape[0] - 1
        upper_trim1_widget = widgets.BoundedIntText(**WIDGET_PARAMS["upper_trim1"])
        upper_trim1_widget.max = data.shape[0] - 1
        lower_trim2_widget = widgets.BoundedIntText(**WIDGET_PARAMS["lower_trim2"])
        lower_trim2_widget.max = data.shape[0] - 1
        upper_trim2_widget = widgets.BoundedIntText(**WIDGET_PARAMS["upper_trim2"])
        upper_trim2_widget.max = data.shape[0] - 1
        fig_height_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_height"])
        fig_width_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_width"])
        fig_width_widget.value = fig_height_widget.value
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
            trend_line=widgets.Dropdown(**WIDGET_PARAMS["trend_line"]),
            alpha=widgets.FloatSlider(**WIDGET_PARAMS["alpha"]),
            lower_trim1=lower_trim1_widget,
            upper_trim1=upper_trim1_widget,
            lower_trim2=lower_trim2_widget,
            upper_trim2=upper_trim2_widget,
            transform1=widgets.Dropdown(**WIDGET_PARAMS["transform1"]),
            transform2=widgets.Dropdown(**WIDGET_PARAMS["transform2"]),
            clip=widgets.BoundedFloatText(**WIDGET_PARAMS["clip"]),
            reference_line=widgets.Checkbox(**WIDGET_PARAMS["reference_line"]),
            plot_density=widgets.Checkbox(**WIDGET_PARAMS["plot_kde"]),
            interactive=widgets.fixed(True),
        )
    else:
        print("No EDA support for this variable type")
        return

    if data_dict is not None:
        print(
            f"Column1 Description: {data_dict[column1] if column1 in data_dict else 'N/A'}\n"
            f"Column2 Description: {data_dict[column2] if column2 in data_dict else 'N/A'}\n"
        )

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

    # Add callback to save current figure if figure_dir is specified
    if figure_dir is not None:

        def savefig_on_click(x):
            widget.result[1].savefig(f"{figure_dir}/{column1}-{column2}.png")

        savefig_button.on_click(savefig_on_click)
