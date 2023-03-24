import numpy as np
import streamlit as st

from prediction_explanations import PredictionExplanations


melted = st.session_state["melted_explanations"]
settings = st.session_state.get("settings")
SERIES_ID = settings["grouping_feature"]
columns = st.session_state["columns"]

pe = PredictionExplanations(
    project_id=settings["project_id"],
    model_id=settings["model_id"],
    deployment_id=settings["deployment_id"],
    segment_id=SERIES_ID,
)


# Add separation
st.write("#")
all_series = f"All {SERIES_ID}s"

cluster_details = st.container()
with cluster_details:
    minimum_neighbors, minimum_sample, segment_selection, cluster_selector = st.columns(
        [4, 4, 4, 4]
    )
    neighbor_param = minimum_neighbors.number_input(
        label="Minimum Cluster Size",
        value=round(len(melted["orig_row_num"].unique()) / 20),
        min_value=3,
    )
    sample_param = minimum_sample.number_input(
        label="Minimum Samples Per Cluster",
        value=20,
        min_value=3,
    )
    segment = segment_selection.selectbox(
        label=f"{SERIES_ID.title()}:",
        options=[all_series] + list(melted[SERIES_ID].unique()),
        index=0,
    )

n_features = st.session_state["n_impact_features"]
# Filter data
if segment != all_series:
    d = melted.loc[melted[settings["grouping_feature"]] == segment, :].copy()
else:
    d = melted.copy()

# Plot Feature Impact
pe.prep_feature_impact(d, n=n_features)

# Add separation
st.write("#")

st.subheader("Clustered Prediction Explanations")

# Get clusters and 2-D embeddings
df_cluster, embedding, colorscales = PredictionExplanations.build_clusters(
    df=melted.loc[melted["feature_name"].isin(st.session_state["columns"]), :],
    color_pallette=pe.colors,
    min_cluster_size=neighbor_param,
    min_samples=sample_param,
)
pe.colorscales = colorscales
print(f"Colors: {sorted(colorscales)}")

# Add cluster selector
cluster_selection = cluster_selector.selectbox(
    label="Select a cluster",
    options=["All Clusters"] + list(df_cluster["labels"].unique()),
    index=0,
)
print(f"cluster_selection: {df_cluster}")

# Filter melted data
if cluster_selection != "All Clusters":
    c_subset = df_cluster.loc[df_cluster["labels"] == cluster_selection, :].copy()
else:
    c_subset = df_cluster.copy()

fig = pe.visualize_clusters(
    melted.loc[melted["feature_name"].isin(st.session_state["columns"]), :],
    df_cluster,
    embedding,
    segment,
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Prediction Explanations Per Feature")

fig = pe.plot_prediction_explanations_per_feature(
    df=melted,
    record_selection=segment,
    df_cluster=c_subset,
    columns=columns,
    series_id=SERIES_ID,
)
st.plotly_chart(fig, use_container_width=True)
