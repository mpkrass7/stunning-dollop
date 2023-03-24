import os
import six
import re
from scipy import stats
import datarobot as dr
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


DEFAULT_HOVER_LABEL = dict(
    bgcolor="white", font_size=16, font_family="Rockwell", namelength=-1
)


def _find_dataset(project):
    for d in dr.Dataset.list():
        if project.id in [p.id for p in d.get_projects()]:
            return d


@st.cache_data
def get_dataset(project):
    dataset = _find_dataset(project)

    buff = six.BytesIO()
    dataset.get_file(filelike=buff)

    buff.seek(0)
    training = pd.read_csv(buff)
    buff.close()

    training["unique_ID"] = training.index

    return training


##Instantiate a prediction explanation class object and get melted prediction explanations
# @st.cache_data
def get_base_explanations(pe, df, filepath=None):
    pattern_filepath = re.compile(r"^(.*/)(.*)(\.csv)$")
    file = pattern_filepath.match(filepath).group(2)
    try:
        melted_explanations = pd.read_csv(f"data/Explanations {file}.csv")
        melted_explanations = melted_explanations.sample(frac=1.0)
    except FileNotFoundError:
        melted_explanations = pe.get_melted_prediction_explanations(
            df, max_explanations=25
        )
        melted_explanations.to_csv(f"data/Explanations {file}.csv", index=False)
    return melted_explanations


# @st.cache_data
def load_data(project=None, filename=False, sample=1.0):
    print("LOADING DATA")
    np.random.seed(42)
    if filename:
        return pd.read_csv(f"{filename}", engine="c", low_memory=False).sample(
            frac=sample
        )
    else:
        return get_dataset(project).sample(frac=sample)


@st.cache_data
def get_or_request_training_predictions_from_model(
    df: pd.DataFrame,
    model,
    data_subset=None,
) -> pd.DataFrame:

    project = dr.Project.get(model.project_id)

    if data_subset == dr.enums.DATA_SUBSET.HOLDOUT:
        project.unlock_holdout()

    try:
        predict_job = model.request_training_predictions(data_subset)
        training_predictions = predict_job.get_result_when_complete(max_wait=10000)

    except dr.errors.ClientError as e:
        prediction_id = [
            p.prediction_id
            for p in dr.TrainingPredictions.list(project.id)
            if p.model_id == model.id and p.data_subset == data_subset
        ][0]
        training_predictions = dr.TrainingPredictions.get(project.id, prediction_id)

    preds = training_predictions.get_all_as_dataframe(serializer="csv")

    df = df.merge(
        preds,
        how="inner",
        left_index=True,
        right_on=["row_id"],
        validate="one_to_one",
    )

    return df


def pivot_melted_df(melted_df: pd.DataFrame):
    return (
        melted_df[["orig_row_num", "feature_name", "feature_strength"]]
        .pivot(index="orig_row_num", columns="feature_name", values="feature_strength")
        .fillna(0)
    )


def extract_colors(df, column: str, cmap="seismic"):

    rng = df[column].max() - df[column].min()
    if rng == 0:
        return ["#FFFFFF" for _ in df[column].values]
    else:
        norm = colors.Normalize(-1 - (rng * 0), 1 + (rng * 0))
        normed = norm(df[column].values)
        return [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]


@st.cache_data
def _cut_series_by_rank(df, series, n=1, top=True):
    series = series[:-2]
    non_numerics = [
        "constantMaturityYieldMovement2_10IT",
        "constantMaturityYieldMovement10_30FR",
        "constantMaturityYieldMovement2_10FR",
        "constantMaturityYieldMovement10_30DE",
        "constantMaturityYieldMovement2_10DE",
        "constantMaturityYieldMovement10_30IT",
    ]
    averages = {col: df[col].mean() for col in series if col not in non_numerics}

    averages_sorted = []
    for key, value in sorted(averages.items(), key=lambda x: x[1], reverse=top):
        averages_sorted.append(key)

    return averages_sorted[0:n]


@st.cache_data
def plot_series_over_time(
    df: pd.DataFrame,
    date_col: str,
    series: list,
    n=5,
    top=True,
):
    df[date_col] = pd.to_datetime(df[date_col])

    series_to_keep = _cut_series_by_rank(df, series, n=n, top=top)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["client", date_col], ascending=True)
    client = df["client"].iloc[0]

    fig = go.Figure()
    for feature in series_to_keep:
        df_subset = df.loc[df["client"] == client, ["date", feature]].reset_index(
            drop=True
        )
        df_subset[feature] = np.where(
            np.abs(df_subset[feature]) > 10, 0, df_subset[feature]
        )
        fig.add_trace(
            go.Scatter(x=df_subset["date"], y=df_subset[feature], name=feature)
        )
    fig.update_yaxes(title="ZScore")
    fig.update_xaxes(title="Date")
    fig.update_layout(
        plot_bgcolor="white",
        hovermode="x unified",
        hoverlabel=DEFAULT_HOVER_LABEL,
    )
    fig["layout"]["yaxis"]["fixedrange"] = True

    return fig


def plot_pe_over_time(
    bar,
    scatter,
    date_col,
):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=scatter[date_col],
            y=scatter["Average Prediction"],
            name="Average Prediction",
            mode="lines+markers",
            marker=dict(
                color="Black",
                size=6,
            ),
        ),
        secondary_y=True,
    )

    fig.update_layout(barmode="relative")

    features = np.sort(bar["feature_name"].unique())
    colors = px.colors.qualitative.Plotly[0 : len(features)]
    marker_color = {column: color for column, color in zip(features, colors)}
    for i, trace in enumerate(bar["feature_name"].unique()):
        dft = bar[bar["feature_name"] == trace]
        median_val = (
            "<br>Most Frequent Value</b>: %{customdata[0]}"
            if isinstance(dft["feature_value"].iloc[0], str)
            else "<br>Median Value</b>: %{customdata[0]: .3}"
        )
        fig.add_traces(
            go.Bar(
                x=dft[date_col],
                y=dft["strength_normalized"],
                name=trace,
                opacity=0.5,
                marker_color=marker_color[trace],
                customdata=dft[["feature_value", "feature_strength", "feature_name"]],
                hovertemplate="<br>Period</b>: %{x}"
                + "<br>Feature</b>: %{customdata[2]}"
                + "<br>Strength</b>: %{y: .2}"
                + median_val
                + "<extra></extra>",
                hoverlabel=DEFAULT_HOVER_LABEL,
            )
        )
    fig.update_yaxes(title="Normalized Feature Strength")
    fig.update_xaxes(title="Date")
    fig.update_layout(
        yaxis2={"title": "Average Prediction", "tickformat": ",.0%"},
        height=600,
    )
    return fig


def prep_pe_data(
    df,
    date_col,
    freq="QS",
):
    df[date_col] = pd.to_datetime(df[date_col])
    grouped = df.copy()
    group = ["feature_name"]

    def try_mean_else_mode(x):
        try:
            return x.astype(float).median()
        except:
            return x.value_counts().index[0]

    counts = (
        grouped.resample(freq, on=date_col)
        .agg(
            {
                "orig_row_num": "nunique",
                "feature_strength": "count",
                "Class_1_Prediction": "mean",
            }
        )
        .reset_index()
        .rename(
            {
                "orig_row_num": "row_count",
                "feature_strength": "count",
                "Class_1_Prediction": "average_prediction",
            },
            axis=1,
        )
    )

    resampled = (
        df.groupby(group)
        .resample(freq, on=date_col)
        .agg(
            {
                "feature_strength": "sum",
                "feature_value": try_mean_else_mode,
            }
        )
        .reset_index()
    )

    normalized = resampled.merge(
        counts[[date_col, "row_count"]],
        how="left",
        on=[date_col],
    )

    normalized["strength_normalized"] = (
        normalized["feature_strength"] / normalized["row_count"]
    )

    barplot_data = normalized

    scatterplot_data = (
        df[[date_col, "Class_1_Prediction"]]
        .resample(freq, on=date_col)
        .mean()
        .reset_index()
        .rename({"Class_1_Prediction": "Average Prediction"}, axis=1)
    )

    return barplot_data, scatterplot_data


def plot_values_over_time(
    df,
    target,
    date_col,
    freq,
    feature,
):
    df[date_col] = pd.to_datetime(df[date_col])
    df["index"] = (
        pd.Series(df[date_col].dt.to_period(freq).sort_values())
        .astype(str)
        .reset_index(drop=True)
    )
    positive_class, negative_class = (
        df[target].value_counts().index[0],
        df[target].value_counts().index[1],
    )

    if np.issubdtype(df[feature].dtype, np.number):
        df[date_col] = pd.to_datetime(df[date_col])
        resampled = (
            df[[feature, date_col]]
            .resample(freq, on=date_col)
            .agg(
                {
                    feature: [
                        lambda x: np.percentile(x, 25),
                        lambda x: np.percentile(x, 50),
                        lambda x: np.mean(x),
                        lambda x: np.percentile(x, 75),
                    ]
                }
            )
            .reset_index()
        )
        resampled.columns = [
            date_col,
            "25th Percentile",
            "50th Percentile",
            "Average",
            "75th Percentile",
        ]

        traces = ["25th Percentile", "50th Percentile", "Average", "75th Percentile"]
        colors = ["lightblue", "blue", "blue", "lightblue"]
        line_type = [None, None, "dash", None]

        fig = go.Figure()
        for trace, color, line_type in zip(traces, colors, line_type):
            fig.add_trace(
                go.Scatter(
                    x=resampled[date_col],
                    y=resampled[trace],
                    mode="lines+markers",
                    name=trace,
                    line=dict(dash=line_type, width=3),
                    line_color=color,
                )
            )
            fig.update_yaxes(title=f"{feature}")
            fig.update_xaxes(title="Date")
            fig.update_layout(
                height=500, title=f"Distribution of {feature.lower()} over time"
            )
        return fig
    else:
        df = (
            df.groupby("index")[feature]
            .apply(lambda x: x.value_counts())
            .reset_index()
            .rename({"level_1": feature, feature: "count"}, axis=1)
        )
        df["total"] = df.groupby("index")["count"].transform("sum")
        df["percentage"] = df["count"] / df["total"]
        fig = px.bar(
            df,
            x="index",
            y="percentage",
            color=feature,
            barmode="relative",
            opacity=0.7,
        )
        fig.update_yaxes(title=f"{feature} (Percentage)")
        fig.update_xaxes(title="Date")
        fig.update_layout(
            height=500, title=f"Distribution of {feature.lower()} over time"
        )
        return fig


@st.cache_data
def prep_data_table(bin_df, col, columns, sample=1000):
    bin_df = bin_df.copy()
    table_data = (
        bin_df.drop_duplicates(["orig_row_num"])
        .loc[:, lambda x: x.columns.isin(columns + [col, "orig_row_num"])]
        .sort_values(by="orig_row_num")
        .reset_index(drop=True)
        .set_index("orig_row_num")
    )
    bin_pivot = (
        pivot_melted_df(bin_df)
        .reset_index()
        .sort_values(by="orig_row_num")
        .reset_index(drop=True)
        .set_index("orig_row_num")
    )

    for col in table_data.columns:
        if col not in bin_pivot.columns:
            bin_pivot[col] = 0
    bin_pivot = bin_pivot[table_data.columns]

    gmap = np.array([bin_pivot.iloc[0:sample][i] for i in bin_pivot.iloc[0:sample]]).T

    return table_data.iloc[0:sample], gmap
