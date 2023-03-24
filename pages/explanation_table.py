import pandas as pd
import plotly.graph_objects as go

import streamlit as st

from helpers import (
    pivot_melted_df,
    extract_colors,
)


def plot_table(table_df, pivot_df, page_number=0, items_per_page=50):
    index = page_number * items_per_page
    color_df = pd.DataFrame()
    for i in pivot_df.columns:
        import pdb

        # pdb.set_trace()
        color_df[i] = extract_colors(pivot_df, i, cmap="seismic")

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[f"<b>{i}</b>" for i in table_df.columns],
                    line_color="white",
                    fill_color="white",
                    align="center",
                    font=dict(color="black", size=12),
                ),
                cells=dict(
                    values=[
                        table_df.loc[index : index + 50, i] for i in table_df.columns
                    ],
                    line_color=[
                        color_df.loc[index : index + 50, i] for i in table_df.columns
                    ],
                    fill_color=[
                        color_df.loc[index : index + 50, i] for i in table_df.columns
                    ],
                    align="center",
                    font=dict(color="black", size=11),
                ),
            ),
        ],
    )
    fig.update_layout(width=3000, height=1000)
    return fig


binned = st.session_state["binned"]
COLUMNS = st.session_state["settings"]["columns"]
col = st.session_state["col"]

st.write("# Explanation Table")

histogram_df = (
    binned.drop_duplicates(["orig_row_num"])
    .loc[:, lambda x: x.columns.isin(COLUMNS + [col, "orig_row_num"])]
    .copy()
)

table_data = (
    histogram_df.sort_values(by="orig_row_num")
    .reset_index(drop=True)
    .drop(columns=["orig_row_num"])
)
bin_pivot = (
    pivot_melted_df(binned)
    .reset_index()
    .sort_values(by="orig_row_num")
    .reset_index(drop=True)
    .drop(columns=["orig_row_num"])
)


for col in table_data.columns:
    if col not in bin_pivot.columns:
        bin_pivot[col] = 0
page_input, _, _ = st.columns(3)

page_number = page_input.number_input("Page number", 0, len(table_data) // 50, 0)

fig = plot_table(table_data, bin_pivot, page_number=page_number, items_per_page=50)
st.plotly_chart(fig)
