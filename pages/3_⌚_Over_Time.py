# Import packages
import datarobot as dr
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from helpers import (
    load_data,
    get_base_explanations,
    plot_pe_over_time,
    prep_pe_data,
    plot_values_over_time,
)

# Insert title
st.subheader("Stacked Prediction Explanations Over Time")

# Load the data, model, and project from DataRobot
melted = st.session_state["melted_explanations"]
settings = st.session_state.get("settings")
SERIES_ID = settings["grouping_feature"]
DATE_COL = settings["date"]
TARGET = dr.Project.get(settings["project_id"]).target
COLUMNS = st.session_state["columns"]

mapping = {"H": "H", "W": "W", "MS": "M", "QS": "Q", "YS": "Y"}

if DATE_COL:
    bar_plot_inputs = st.container()
    with bar_plot_inputs:
        time_selection, segment_selection, feature_selection = st.columns([2, 2, 2])
        all_series = f"ALL {SERIES_ID}S"
        time_options = list(mapping.keys())
        time_unit = time_selection.selectbox(
            label="Time Frequency",
            options=time_options,
            index=2,
        )
        segment = segment_selection.selectbox(
            label=f"{SERIES_ID.title()}:",
            options=[all_series] + list(melted[SERIES_ID].unique()),
            index=0,
        )
        feature = feature_selection.selectbox(
            label="Feature",
            options=COLUMNS,
            index=0,
        )

    if segment != all_series:
        melted_filter = melted.loc[melted[SERIES_ID] == segment, :].copy()
    else:
        melted_filter = melted.copy()

    barplot_data, scatterplot_data = prep_pe_data(
        melted_filter.loc[melted_filter["feature_name"].isin(COLUMNS), :].reset_index(),
        date_col="PRED_POINT",
        freq=time_unit,
    )

    fig = plot_pe_over_time(
        barplot_data,
        scatterplot_data,
        date_col="PRED_POINT",
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = plot_values_over_time(
        melted_filter.reset_index(),
        target=TARGET,
        freq=mapping[time_unit],
        date_col=DATE_COL,
        feature=feature,
    )

    st.plotly_chart(fig, use_container_width=True)
