# Import packages
import os
import datarobot as dr
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import yaml

from helpers import load_data, get_base_explanations, prep_data_table
from lift_chart import LiftChart
from prediction_explanations import PredictionExplanations


dr.Client(
    token=st.secrets["DATAROBOT_API_TOKEN"], endpoint=st.secrets["DATAROBOT_ENDPOINT"]
)


# Load yaml file
FILEPATH = os.getcwd()


def run():
    with open(f"config_streamlit_app/config_dr_churn.yaml") as file:
        print("READING YAML FILE FROM")
        settings = yaml.load(file, Loader=yaml.FullLoader)

    if "settings" not in st.session_state:
        st.session_state["settings"] = settings

    # Initial confif
    FILENAME = f"{st.session_state['settings']['filename']}"
    DATASET_ID = st.session_state["settings"]["dataset_id"]
    PROJECT_ID = st.session_state["settings"]["project_id"]
    MODEL_ID = st.session_state["settings"]["model_id"]
    N = st.session_state["settings"]["n"]
    DEPLOYMENT_ID = st.session_state["settings"]["deployment_id"]
    SERIES_ID = st.session_state["settings"]["grouping_feature"]
    SAMPLE = st.session_state["settings"]["sample"]
    COLUMNS = st.session_state["settings"]["columns"]
    DATE_COL = st.session_state["settings"]["date"]

    # Set page layout
    st.set_page_config(layout="wide")

    # Insert title
    st.title("Interactive DataRobot Model Plots")

    # Add sub header
    st.caption("Feature Impact; Lift Chart; Prediction Explanations; and More")

    # Load the data, model, and project from DataRobot
    project = dr.Project.get(PROJECT_ID)
    model = dr.Model.get(PROJECT_ID, MODEL_ID)
    df_sample = load_data(project, filename=FILENAME, sample=SAMPLE)

    if COLUMNS is None:
        COLUMNS = model.get_features_used()
    st.session_state["columns"] = COLUMNS

    pe = PredictionExplanations(
        project_id=PROJECT_ID,
        model_id=MODEL_ID,
        deployment_id=DEPLOYMENT_ID,
        segment_id=SERIES_ID,
    )

    try:
        melted = st.session_state["melted_explanations"]
    except KeyError:
        melted = get_base_explanations(pe, df_sample, filepath=FILENAME)
        st.session_state["melted_explanations"] = melted

    # Set up web page structure
    fi_plot = st.container()

    # Insert filters
    left, center, right = st.columns([1, 1, 1])

    all_series = f"ALL {SERIES_ID}S"

    with left:
        n_bins = st.number_input(
            label="Bins: ",
            min_value=2,
            max_value=100,
            value=10,
        )

    with center:
        n_features = st.number_input(
            label="Number of features: ",
            min_value=1,
            max_value=100,
            value=N,
        )
        st.session_state["n_impact_features"] = n_features

    with right:
        segment = st.selectbox(
            label="Segment:",
            options=[all_series] + list(melted[SERIES_ID].unique()),
            index=0,
        )
        st.session_state["segment"] = segment

    # Instantiate a lift chart class object
    lc = LiftChart(
        project_id=PROJECT_ID,
        model_id=MODEL_ID,
        deployment_id=DEPLOYMENT_ID,
    )

    fi_inputs = st.container()
    l, neg, _, pos, r = st.columns([1, 10, 1, 10, 1])

    # Filter data
    if segment != all_series:
        melted_filter = melted.loc[melted[SERIES_ID] == segment, :].copy()
    else:
        melted_filter = melted.copy()

    # Plot Feature Impact
    fig = pe.plot_feature_impact(
        # melted_filter,
        melted_filter.loc[melted_filter["feature_name"].isin(COLUMNS), :],
        n=N,
        height=500,
    )

    with fi_plot:
        st.plotly_chart(fig, use_container_width=True)

    # Plot feature impact by predicted class
    with fi_inputs:
        cutoff = st.number_input(
            label="Input a threshold: ",
            min_value=0.0,
            max_value=1.0,
            value=model.prediction_threshold,
        )

    # Positive class
    col = list(pe._get_column_name_mappings().values())[0]
    st.session_state["col"] = col
    positive_class = melted_filter.loc[
        (melted_filter[col] >= cutoff) & (melted_filter["feature_name"].isin(COLUMNS)),
        :,
    ]
    fig = pe.plot_signed_feature_impact(positive_class, n=N, title="Positive Class")
    with pos:
        st.plotly_chart(
            fig, use_container_width=True, height=500, config={"displayModeBar": False}
        )

    # Negative class
    negative_class = melted_filter.loc[
        (melted_filter[col] < cutoff) & (melted_filter["feature_name"].isin(COLUMNS)), :
    ]
    fig = pe.plot_signed_feature_impact(negative_class, n=N, title="Negative Class")
    with neg:
        st.plotly_chart(
            fig, use_container_width=True, height=500, config={"displayModeBar": False}
        )

    # Insert lift chart
    binned = lc.add_bins_to_data(melted_filter, bins=n_bins)
    grouped = lc.group_data_by_bin(binned)

    # Insert binned prediction explanations
    d_subset = melted_filter.loc[melted_filter["feature_name"].isin(COLUMNS), :]
    binned_explanations = pe.get_prediction_explanations_per_bin(
        d_subset,
        max_features=n_features,
        bins=n_bins,
    )

    fig = pe.plot_prediction_explanations_and_lift_chart(
        df=binned_explanations,
        grouped_df=grouped,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Grouped Histogram
    feature_selector, hist_type_selector, _ = st.columns([1, 1, 1])
    with feature_selector:
        histogram_feature = feature_selector.selectbox(
            label="Select a feature",
            options=COLUMNS,
            index=0,
        )
        hist_type = hist_type_selector.selectbox(
            label="Select a plotting type",
            options=[None, "probability", "percent", "probability density"],
            index=0,
        )

    negative_class = [
        x for x in melted[project.target].unique() if x != project.positive_class
    ][0]
    histogram_df = binned.drop_duplicates(["orig_row_num"]).copy()
    st.session_state["binned"] = binned

    histogram_df["group"] = np.where(
        histogram_df[col] >= cutoff, project.positive_class, negative_class
    )

    fig = px.histogram(
        histogram_df,
        x=histogram_feature,
        color="group",
        barmode="group",
        histnorm=hist_type,  # "probability", #'percent', 'probability denisty',
        color_discrete_map={
            project.positive_class: "blue",
            negative_class: "lightblue",
        },
    )
    y_axis_title = "Record Count" if hist_type is None else hist_type
    fig.update_layout(yaxis_title=f"{y_axis_title}", legend_title="Predicted Class")
    st.plotly_chart(fig, use_container_width=True)

    table_data, gmap = prep_data_table(
        melted_filter, col, columns=COLUMNS, sample=min(melted_filter.shape[0], 1000)
    )

    st.dataframe(
        table_data.style.background_gradient(
            cmap="seismic", axis=None, vmin=-1, vmax=1, gmap=gmap
        )
    )


run()
