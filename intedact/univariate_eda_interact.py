import json
import os
import warnings

import ipywidgets as widgets
import pandas as pd
import seaborn as sns

from .config import FLIP_LEVEL_COUNT
from .config import WIDGET_PARAMS
from .data_utils import coerce_column_type
from .data_utils import detect_column_type
from .data_utils import freedman_diaconis_bins
from .univariate_summaries import *


def univariate_eda_interact(
    data,
    figure_dir: str = None,
    notes_file: str = None,
    data_dict_file: str = None,
):
    pd.set_option("display.precision", 2)
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

    col_widget = widgets.Dropdown(**WIDGET_PARAMS["column"])
    col_widget.options = data.columns

    widget = widgets.widgets.interactive(
        column_univariate_eda_interact,
        data=widgets.widgets.fixed(data),
        column=col_widget,
        summary_type=widgets.Dropdown(**WIDGET_PARAMS["univariate_summary_type"]),
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
        summary_type = detect_column_type(data[col_widget.value])
        type_widget.value = summary_type
        auto_widget.value = summary_type not in ["text"]

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
    data_dict: str = None,
    notes_file: str = None,
    figure_dir: str = None,
    savefig_button: Optional[widgets.Button] = None,
) -> None:
    data = data.copy()

    data[column] = coerce_column_type(data[column], summary_type)

    color_palette_widget = widgets.Text(**WIDGET_PARAMS["color_palette"])
    if summary_type == "categorical":
        fig_height_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_height"])
        fig_height_widget.value = max(6, min(data[column].nunique(), 24) // 2)
        flip_axis_widget = widgets.Checkbox(**WIDGET_PARAMS["flip_axis"])
        flip_axis_widget.value = data[column].nunique() > FLIP_LEVEL_COUNT
        widget = widgets.interactive(
            categorical_univariate_summary,
            {"manual": not auto_update},
            data=widgets.widgets.fixed(data),
            column=widgets.widgets.fixed(column),
            fig_height=fig_height_widget,
            fig_width=widgets.IntSlider(**WIDGET_PARAMS["fig_width"]),
            fontsize=widgets.FloatSlider(**WIDGET_PARAMS["fontsize"]),
            color_palette=color_palette_widget,
            order=widgets.Dropdown(**WIDGET_PARAMS["order"]),
            max_levels=widgets.IntSlider(**WIDGET_PARAMS["max_levels"]),
            label_counts=widgets.Checkbox(**WIDGET_PARAMS["label_counts"]),
            percent_axis=widgets.Checkbox(**WIDGET_PARAMS["percent_axis"]),
            label_fontsize=widgets.FloatSlider(**WIDGET_PARAMS["label_fontsize"]),
            include_missing=widgets.Checkbox(**WIDGET_PARAMS["include_missing"]),
            flip_axis=flip_axis_widget,
            label_rotation=widgets.IntSlider(**WIDGET_PARAMS["label_rotation"]),
            interactive=widgets.fixed(True),
        )
    elif summary_type == "numeric":
        bins_widget = widgets.IntSlider(**WIDGET_PARAMS["bins"])
        bins_widget.value = freedman_diaconis_bins(
            data[~data[column].isna()][column], log=False
        )
        lower_quantile_widget = widgets.BoundedFloatText(
            **WIDGET_PARAMS["lower_quantile"]
        )
        upper_quantile_widget = widgets.BoundedFloatText(
            **WIDGET_PARAMS["upper_quantile"]
        )
        widget = widgets.interactive(
            numeric_univariate_summary,
            {"manual": not auto_update},
            data=widgets.fixed(data),
            column=widgets.fixed(column),
            fig_height=widgets.IntSlider(**WIDGET_PARAMS["fig_height"]),
            fig_width=widgets.IntSlider(**WIDGET_PARAMS["fig_width"]),
            fontsize=widgets.FloatSlider(**WIDGET_PARAMS["fontsize"]),
            color_palette=color_palette_widget,
            bins=bins_widget,
            clip=widgets.BoundedFloatText(**WIDGET_PARAMS["clip"]),
            lower_quantile=lower_quantile_widget,
            upper_quantile=upper_quantile_widget,
            transform=widgets.Dropdown(**WIDGET_PARAMS["transform"]),
            kde=widgets.Checkbox(**WIDGET_PARAMS["kde"]),
            interactive=widgets.fixed(True),
        )
    # datetime variables
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
            datetime_univariate_summary,
            {"manual": not auto_update},
            data=widgets.fixed(data),
            column=widgets.fixed(column),
            fig_height=widgets.IntSlider(**WIDGET_PARAMS["fig_height"]),
            fig_width=widgets.IntSlider(**WIDGET_PARAMS["fig_width"]),
            fontsize=widgets.FloatSlider(**WIDGET_PARAMS["fontsize"]),
            color_palette=color_palette_widget,
            ts_freq=widgets.Text(**WIDGET_PARAMS["ts_freq"]),
            delta_units=widgets.Dropdown(**WIDGET_PARAMS["delta_units"]),
            ts_type=widgets.Dropdown(**WIDGET_PARAMS["ts_type"]),
            trend_line=widgets.Dropdown(**WIDGET_PARAMS["trend_line"]),
            date_breaks=widgets.Text(**WIDGET_PARAMS["date_breaks"]),
            date_labels=widgets.Text(**WIDGET_PARAMS["date_labels"]),
            lower_quantile=lower_quantile_widget,
            upper_quantile=upper_quantile_widget,
            interactive=widgets.fixed(True),
        )
    elif summary_type == "text":
        fig_width_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_width"])
        fig_width_widget.value = 16
        widget = widgets.interactive(
            text_univariate_summary,
            {"manual": not auto_update},
            data=widgets.fixed(data),
            column=widgets.fixed(column),
            fig_height=widgets.IntSlider(**WIDGET_PARAMS["fig_height"]),
            fig_width=fig_width_widget,
            fontsize=widgets.FloatSlider(**WIDGET_PARAMS["fontsize"]),
            color_palette=color_palette_widget,
            top_ngrams=widgets.IntSlider(**WIDGET_PARAMS["top_ngrams"]),
            compute_ngrams=widgets.Checkbox(**WIDGET_PARAMS["compute_ngrams"]),
            lower_case=widgets.Checkbox(**WIDGET_PARAMS["lower_case"]),
            remove_punct=widgets.Checkbox(**WIDGET_PARAMS["remove_punct"]),
            remove_stop=widgets.Checkbox(**WIDGET_PARAMS["remove_stop"]),
            interactive=widgets.fixed(True),
        )
    elif summary_type == "collection":
        fig_width_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_width"])
        fig_width_widget.value = 16
        widget = widgets.interactive(
            collection_univariate_summary,
            {"manual": not auto_update},
            data=widgets.fixed(data),
            column=widgets.fixed(column),
            fontsize=widgets.FloatSlider(**WIDGET_PARAMS["fontsize"]),
            color_palette=color_palette_widget,
            fig_height=widgets.IntSlider(**WIDGET_PARAMS["fig_height"]),
            fig_width=fig_width_widget,
            top_entries=widgets.IntSlider(**WIDGET_PARAMS["top_entries"]),
            remove_duplicates=widgets.Checkbox(**WIDGET_PARAMS["remove_duplicates"]),
            sort_collections=widgets.Checkbox(**WIDGET_PARAMS["sort_collections"]),
            interactive=widgets.fixed(True),
        )
    elif summary_type == "url":
        fig_height_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_height"])
        fig_height_widget.value = 8
        fig_width_widget = widgets.IntSlider(**WIDGET_PARAMS["fig_width"])
        fig_width_widget.value = 16
        widget = widgets.interactive(
            url_univariate_summary,
            {"manual": not auto_update},
            data=widgets.fixed(data),
            column=widgets.fixed(column),
            fontsize=widgets.FloatSlider(**WIDGET_PARAMS["fontsize"]),
            color_palette=color_palette_widget,
            fig_height=fig_height_widget,
            fig_width=fig_width_widget,
            top_entries=widgets.IntSlider(**WIDGET_PARAMS["top_entries"]),
            interactive=widgets.fixed(True),
        )
    else:
        print("No EDA support for this variable type")
        return

    if data_dict is not None:
        print(
            f"Column Description: {data_dict[column] if column in data_dict else 'N/A'}"
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

    # Add callback to save current figure if figure_dir is specified
    if figure_dir is not None:

        def savefig_on_click(x):
            widget.result[1].savefig(f"{figure_dir}/{column}.png")

        savefig_button.on_click(savefig_on_click)
