from typing import List
from typing import Union

import pandas as pd
import plotly.graph_objects as go

from .data_utils import order_levels


def countplot(
    data: pd.DataFrame,
    column: str,
    fig: go.Figure,
    fig_row: int = 1,
    fig_col: int = 1,
    order: Union[str, List] = "auto",
    max_levels: int = 20,
    flip_axis: bool = False,
    include_missing: bool = False,
    add_other: bool = True,
) -> go.Figure:

    # Handle axis flip default
    num_levels = data[column].nunique()
    num_plot_levels = min(max_levels, num_levels)
    if flip_axis is None:
        flip_axis = num_plot_levels > 5

    # Reorder column levels
    order = order_levels(
        data,
        column,
        None,
        order=order,
        max_levels=max_levels,
        include_missing=include_missing,
        add_other=add_other,
    )
    data[column] = data[column].astype(str)
    if num_levels > max_levels and add_other:
        label = (
            f"{column}<br>({num_levels - max_levels + 1} levels condensed into 'Other')"
        )
    else:
        label = column

    count_data = (
        data.groupby(column).size().reset_index().rename({0: "Count"}, axis="columns")
    )
    count_data["Percent"] = (100 * count_data["Count"] / count_data.Count.sum()).apply(
        lambda x: f"{x:.2f}%"
    )
    if flip_axis:
        fig.add_trace(
            go.Bar(
                y=count_data[column],
                x=count_data["Count"],
                text=count_data["Percent"],
                texttemplate="%{x} (%{text})",
                hovertemplate=(
                    f"{column}"
                    + ": %{y}<br>"
                    + "Count: %{x}<br>"
                    + "Percent: %{text}"
                    + "<extra></extra>"
                ),
                orientation="h",
            ),
            row=fig_row,
            col=fig_col,
        )
        fig.update_layout(
            yaxis={"categoryorder": "array", "categoryarray": order[::-1]}
        )
        fig.update_yaxes(title_text=label, row=1, col=1)
        fig.update_xaxes(title_text="Count", row=1, col=1)
    else:
        fig.add_trace(
            go.Bar(
                x=count_data[column],
                y=count_data["Count"],
                text=count_data["Percent"],
                texttemplate="%{y}<br>%{text}",
                hovertemplate=(
                    f"{column}"
                    + ": %{x}<br>"
                    + "Count: %{y}<br>"
                    + "Percent: %{text}"
                    + "<extra></extra>"
                ),
                orientation="v",
            ),
            row=fig_row,
            col=fig_col,
        )
        fig.update_layout(xaxis={"categoryorder": "array", "categoryarray": order})
        fig.update_xaxes(title_text=label, row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)

    return fig
