import re
from typing import List, Tuple
from itertools import product

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import hdbscan
import umap.umap_ as umap

from helpers import DEFAULT_HOVER_LABEL, pivot_melted_df
from lift_chart import LiftChart


class PredictionExplanations(LiftChart):
    """
    Description:

    Attributes:
    -----------
    Inputs

    Methods:
    --------
    Funs

    """

    def __init__(
        self,
        project_id: str,
        model_id: str,
        deployment_id: str,
        segment_id: str,
        **kwargs,
    ):
        super().__init__(project_id, model_id, deployment_id, **kwargs)

        self.colors = px.colors.qualitative.Alphabet
        self.segment_id = segment_id
        self.all_series = f"All {self.segment_id}s"

    def _get_column_names_and_numbers(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[int]]:
        """
        Description

        Attributes:
        -----------
        Inputs

        Methods:
        --------
        Funs

        """

        rows = df.shape[0]
        pattern_base = re.compile("EXPLANATION_(.*?)")
        prediction_explanation_column_names = [
            col for col in df.columns if pattern_base.match(col)
        ]
        pattern = re.compile("EXPLANATION_(.*?)_FEATURE_NAME")

        populated_explanation_column_numbers = [
            pattern.match(col)[1]
            for col in prediction_explanation_column_names
            if pattern.match(col) and df[col].isna().sum() != rows
        ]
        return prediction_explanation_column_names, populated_explanation_column_numbers

    @staticmethod
    @st.cache_data
    def build_clusters(
        df,
        color_pallette,
        min_cluster_size=12,
        min_samples=5,
    ):
        """
        Runs UMAP and HDBSCAN on prediction explanations data
        """
        print("BUILDING CLUSTERS")
        np.random.seed(42)

        df_cluster = pivot_melted_df(df)

        standard_embedding = umap.UMAP(random_state=42).fit_transform(df_cluster.values)

        clusterable_embedding = umap.UMAP(
            n_neighbors=30,
            min_dist=0.0,
            n_components=2,
            random_state=42,
        ).fit_transform(df_cluster.values)

        df_cluster["labels"] = hdbscan.HDBSCAN(
            min_samples=int(
                min_samples
            ),  # The number of samples in a neighborhood for a point to be considered as a core point
            min_cluster_size=int(min_cluster_size),
        ).fit_predict(clusterable_embedding)

        colorscales = dict(zip(df_cluster["labels"].unique(), color_pallette))

        standard_embedding = pd.DataFrame(
            zip(df_cluster.index.values, standard_embedding),
            columns=["orig_row_num", "embeddings"],
        )
        standard_embedding["embedding_0"] = standard_embedding["embeddings"].apply(
            lambda x: x[0]
        )
        standard_embedding["embedding_1"] = standard_embedding["embeddings"].apply(
            lambda x: x[1]
        )

        return df_cluster, standard_embedding, colorscales

    def melt_prediction_explanations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Description

        Attributes:
        -----------
        Inputs

        Methods:
        --------
        Funs

        """
        print("MELTING PREDICTION EXPLANATIONS")
        column_names, column_numbers = self._get_column_names_and_numbers(df)

        feature_names = []
        feature_strengths = []
        feature_values = []

        for num in column_numbers:
            feature_names.append(f"EXPLANATION_{str(num)}_FEATURE_NAME")
            feature_strengths.append(f"EXPLANATION_{str(num)}_STRENGTH")
            feature_values.append(f"EXPLANATION_{str(num)}_ACTUAL_VALUE")

        df["orig_row_num"] = df.index
        name_mappings = self._get_column_name_mappings()

        melted_feature_names = df.melt(
            id_vars=["orig_row_num"],
            value_vars=feature_names,
            value_name="feature_name",
            var_name="variable_number",
        )

        melted_feature_strengths = df.melt(
            id_vars=["orig_row_num"],
            value_vars=feature_strengths,
            value_name="feature_strength",
            var_name="variable_number",
        )

        melted_feature_values = df.melt(
            id_vars=["orig_row_num"],
            value_vars=feature_values,
            value_name="feature_value",
            var_name="variable_number",
        )

        pattern = re.compile("[0-9]+")
        trim = lambda x: pattern.findall(x)[0]

        melted_feature_names["variable_number"] = melted_feature_names[
            "variable_number"
        ].map(trim)
        melted_feature_strengths["variable_number"] = melted_feature_strengths[
            "variable_number"
        ].map(trim)
        melted_feature_values["variable_number"] = melted_feature_values[
            "variable_number"
        ].map(trim)

        merged = (
            melted_feature_names.merge(
                melted_feature_strengths, on=["orig_row_num", "variable_number"]
            )
            .merge(melted_feature_values, on=["orig_row_num", "variable_number"])
            .drop(["variable_number"], axis=1)
        )

        # Merge prediction explanations with original dataframe
        combined = merged.merge(
            df,
            how="inner",
            left_on="orig_row_num",
            right_index=True,
            validate="many_to_one",
        ).copy()

        mean_strength = combined["feature_strength"].mean()
        std_strength = combined["feature_strength"].std()
        standardized_strength = (
            combined["feature_strength"] - mean_strength
        ) / std_strength
        combined["qualitative_strength"] = np.select(
            [
                standardized_strength < -2,
                standardized_strength < -1,
                standardized_strength < 0,
                standardized_strength < 1,
                standardized_strength < 2,
                standardized_strength >= 2,
            ],
            ["---", "--", "-", "+", "++", "+++"],
        )

        return combined

    @st.cache_data
    def get_melted_prediction_explanations(
        _self, df: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        preds = _self.get_batch_predictions(df, **kwargs)
        melted_preds = _self.melt_prediction_explanations(preds)
        return melted_preds

    def get_prediction_explanations_per_bin(
        self,
        df: pd.DataFrame,
        max_features=5,
        **kwargs,
    ) -> pd.DataFrame:
        def identify_type(row):
            """Figures out if a row in a series is a number or a string"""
            try:
                x = float(row)
                row_type = "Num"
            except:
                row_type = "Str"
            return row_type

        print("GETTING PREDICTION EXPLANATIONS PER BIN")

        ranked_features = (
            df.groupby("feature_name")["feature_strength"]
            .apply(lambda x: np.abs(x).sum())
            .reset_index()
            .sort_values(by="feature_strength", ascending=True)
        )

        features_to_keep = ranked_features[-max_features:]["feature_name"]

        binned_data = self.add_bins_to_data(df, **kwargs)

        def try_mean_else_mode(x):
            try:
                return x.astype(float).median()
            except:
                return x.value_counts().index[0]

        grouped = (
            binned_data.groupby(["bins", "feature_name"])[
                "feature_strength", "feature_value"
            ]
            .agg(
                feature_strength=("feature_strength", "sum"),
                feature_value=("feature_value", try_mean_else_mode),
            )
            .sort_values(by="feature_strength", ascending=True)
            .reset_index()
        )

        filtered_df = grouped.loc[grouped["feature_name"].isin(features_to_keep), :]
        filtered_df["bins"] += 1
        return filtered_df

    def plot_prediction_explanations_per_bin(
        self,
        df: pd.DataFrame,
    ):
        print("PLOTTING PREDICTION EXPLANATIONS PER BIN")

        fig = px.bar(
            df,
            x="bins",
            y="feature_strength",
            hover_data={"feature_value": True},
            height=400,
            color="feature_name",
            opacity=0.75,
        )
        fig.update_layout(
            legend=dict(
                bgcolor="rgba(255, 255, 255, 0)",
                bordercolor="rgba(255, 255, 255, 0)",
            ),
            legend_title="Features: ",
            hoverlabel=DEFAULT_HOVER_LABEL,
        )
        fig.update_yaxes(title="Cumulative Impact")
        fig.update_xaxes(title="Bins")
        return fig

    def plot_prediction_explanations_and_lift_chart(
        self,
        df: pd.DataFrame,
        grouped_df: pd.DataFrame,
        **kwargs,
    ):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        col = list(self._get_column_name_mappings().values())[0]
        # Add predictions
        fig.add_trace(
            go.Scatter(
                x=grouped_df["bins"],
                y=grouped_df[col],
                mode="lines+markers",
                name="Predictions",
                marker=dict(
                    size=5,
                    color="blue",
                    symbol="cross-open",
                    line=dict(
                        color="blue",
                        width=2,
                    ),
                ),
                line=dict(
                    color="blue",
                    width=2,
                ),
            ),
            secondary_y=True,
        )
        # Add actuals
        fig.add_trace(
            go.Scatter(
                x=grouped_df["bins"],
                y=grouped_df["actuals"],
                mode="lines+markers",
                name="Actuals",
                marker=dict(
                    size=5,
                    color="#ff7f0e",
                    symbol="circle-open",
                    line=dict(
                        color="#ff7f0e",
                        width=1,
                    ),
                ),
                line=dict(
                    color="#ff7f0e",
                    width=2,
                ),
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title={
                # "text": f"<b>Lift Chart</b>",
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            legend=dict(
                bgcolor="rgba(255, 255, 255, 0)",
                bordercolor="rgba(255, 255, 255, 0)",
            ),
            legend_title="Features: ",
            hoverlabel=DEFAULT_HOVER_LABEL,
        )
        fig.update_yaxes(title=f"Average Target")
        fig.update_xaxes(title=f"Bins")
        fig.update_layout(
            barmode="relative",
            yaxis2={"title": "Average Prediction", "tickformat": ",.0%"},
        )

        # Add bins
        features = np.sort(df["feature_name"].unique())
        colors = px.colors.qualitative.Plotly[0 : len(features)]
        marker_color = {column: color for column, color in zip(features, colors * 5)}

        for trace in features:
            dft = df[df["feature_name"] == trace]
            avg_val = (
                "<br>Most Frequent Value</b>: %{customdata[0]}"
                if isinstance(dft["feature_value"].iloc[0], str)
                else "<br>Average Value</b>: %{customdata[0]: .3}"
            )
            fig.add_traces(
                go.Bar(
                    x=dft["bins"],
                    y=dft["feature_strength"],
                    name=trace,
                    marker_color=marker_color[trace],
                    opacity=0.5,
                    customdata=dft[
                        ["feature_value", "feature_strength", "feature_name"]
                    ],
                    hovertemplate="<br>Bin</b>: %{x}"
                    + "<br>Feature</b>: %{customdata[2]}"
                    + "<br>Strength</b>: %{y: .2}"
                    + avg_val
                    + "<extra></extra>",
                    hoverlabel=DEFAULT_HOVER_LABEL,
                )
            )
        fig.update_yaxes(title="Feature Strength")
        fig.update_xaxes(title="Bins")
        fig.update_layout(
            yaxis2={"title": "Average Churn Prediction", "tickformat": ",.0%"},
            height=600,
            legend_title="Features: ",
            hoverlabel=DEFAULT_HOVER_LABEL,
        )
        return fig

    @staticmethod
    def observe_record(
        df,
        record_id,
    ):
        """
        Filters dataframe to one record, looking only at explanatory values

        Parameters
        ----------
        df : pd.DataFrame
            A dataframe that is not filtered
        """
        explanation_columns = {
            "feature_name": "Feature Name",
            "feature_value": "Feature Value",
            "feature_strength": "Strength",
            "qualitative_strength": "Qualitative Strength",
        }

        df = (
            df.copy()
            .loc[lambda x: x["orig_row_num"] == record_id]
            .rename(columns=explanation_columns)
            .assign(abs_strength=lambda x: abs(x["Strength"]))
            .sort_values(by="abs_strength", ascending=False)
        )
        return df[explanation_columns.values()], df["Feature Value"].values[0]

    @staticmethod
    def observe_records(
        df,
        record_id,
    ):
        """
        Filters dataframe to one client, looking only at explanatory values

        Parameters
        ----------
        df : pd.DataFrame
            A dataframe that is not filtered
        """
        explanation_columns = {
            "feature_name": "Feature Name",
            "feature_value": "Feature Value",
            "feature_strength": "Strength",
            "qualitative_strength": "Qualitative Strength",
        }

        df = (
            df.copy()
            .loc[lambda x: x[self.segment_id] == record_id]
            .rename(columns=explanation_columns)
            .assign(abs_strength=lambda x: abs(x["Strength"]))
            .sort_values(by="abs_strength", ascending=False)
        )

        return df[explanation_columns.values()], df["Feature Value"].values[0]

    def plot_prediction_explanations_per_feature(
        self,
        df: pd.DataFrame,
        record_selection: str = None,
        df_cluster: pd.DataFrame = None,
        columns: list = None,
        series_id: str = None,
    ):
        print("PLOTTING PREDICTION EXPLANATIONS PER FEATURE")
        np.random.seed(42)

        features = self.feature_impact.loc[
            self.feature_impact["feature_name"].isin(columns), "feature_name"
        ]

        fig = make_subplots(
            rows=len(features),
            cols=1,
            shared_xaxes=True,
            subplot_titles=tuple(features.unique()),
        )

        df_sample = df.merge(
            df_cluster[["labels"]],
            how="inner",
            left_on="orig_row_num",
            right_index=True,
        ).copy()

        print("CLUSTER LABELS: ", df_sample["labels"].unique())

        if record_selection == self.all_series:
            df_filtered = df_sample.copy()
        else:
            df_filtered = df_sample[
                lambda x: x[self.segment_id] == record_selection
            ].copy()

        hovertemplate = "%{text} <br><b>Feature Value</b>: %{customdata}</br><b>Strength</b>: %{x}<extra></extra>"
        for i, feature in enumerate(features[::-1]):
            df_feature = df_filtered.loc[lambda x: x["feature_name"] == feature]

            try:
                cluster_mean = df_feature["feature_value"].astype(float).mean()
                overall_mean = (
                    df.loc[lambda x: x["feature_name"] == feature, "feature_value"]
                    .astype(float)
                    .mean()
                )
            except ValueError:
                cluster_mean = df_feature["feature_value"].mode()
                overall_mean = df.loc[
                    lambda x: x["feature_name"] == feature, "feature_value"
                ].mode()

            overall_mean_strength = (
                df.loc[lambda x: x["feature_name"] == feature, "feature_strength"]
                .astype(float)
                .mean()
            )
            cluster_mean_strength = df_feature["feature_strength"].astype(float).mean()

            fig.add_trace(
                go.Scatter(
                    y=np.random.random(df_feature.shape[0]),
                    x=df_feature["feature_strength"],
                    mode="markers",
                    showlegend=False,
                    customdata=df_feature["feature_value"],
                    hovertemplate=hovertemplate,
                    text=[
                        f"<b>Row Number</b>: {a} </br><b>{series_id}<b>: ({b}) </br><b>Feature Name</b>: {feature}"
                        for a, b in zip(
                            df_feature["orig_row_num"], df_feature[self.segment_id]
                        )
                    ],
                    name=feature,
                    marker_color=[self.colorscales[i] for i in df_feature["labels"]],
                    marker=dict(size=5),
                    marker_line_color="black",
                    marker_line_width=1,
                    legendgroup=i,
                ),
                row=len(features) - i,
                col=1,
            )

            marker_direction = (
                "up"
                if round(cluster_mean_strength, 4) == round(overall_mean_strength, 4)
                else (
                    "left" if cluster_mean_strength < overall_mean_strength else "right"
                )
            )

            cluster_hover_template = "<b>Cluster Average</b></br>%{text}</br><b>Average Strength</b>: %{x}<extra></extra>"
            overall_hover_template = "<b>Overall Average</b></br>%{text}</br><b>Average Strength</b>: %{x}<extra></extra>"
            fig.add_trace(
                go.Scatter(
                    y=[0.5],
                    x=[cluster_mean_strength],
                    customdata=[cluster_mean],
                    text=[
                        f"<b>Cluster Average</b></br><b>Feature Name</b>: {feature}</br><b>Average Feature Value</b>: {cluster_mean}"
                    ],
                    hovertemplate=cluster_hover_template,
                    showlegend=False,
                    mode="markers",
                    marker_symbol=f"triangle-{marker_direction}",
                    marker_line_color="midnightblue",
                    marker_color="lightskyblue",
                    marker_line_width=2,
                    name=feature,
                    legendgroup=i,
                    marker=dict(size=16),
                ),
                row=len(features) - i,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    y=[0.5],
                    x=[overall_mean_strength],
                    customdata=[overall_mean],
                    text=[
                        f"<b>Overall Average</b></br><b>Feature Name</b>: {feature}</br><b>Average Feature Value</b>: {overall_mean}"
                    ],
                    hovertemplate=overall_hover_template,
                    showlegend=False,
                    mode="markers",
                    marker_symbol="diamond-tall",
                    marker_line_color="midnightblue",
                    marker_color="lightskyblue",
                    marker_line_width=2,
                    name=feature,
                    legendgroup=i,
                    marker=dict(size=15),
                ),
                row=len(features) - i,
                col=1,
            )
        fig.update_yaxes(
            color="#404040",
            linecolor="#adadad",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        )
        fig.update_layout(
            hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
            plot_bgcolor="#ffffff",
            margin=dict(l=0, r=20, b=20, t=50, pad=4),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=600,
        )
        fig["layout"]["yaxis"]["fixedrange"] = True

        return fig

    def prep_feature_impact(
        self,
        df: pd.DataFrame,
        n: int = 25,
    ):
        """
        Description

        Attributes:
        -----------
        Inputs

        Methods:
        --------
        Funs

        """

        df = df.groupby("feature_name")["feature_strength"].apply(
            lambda x: np.abs(x).sum()
        )
        df_subset = df.reset_index().sort_values(by="feature_strength", ascending=True)[
            -n:
        ]

        self.feature_impact = df_subset
        return df_subset

    def plot_feature_impact(
        self,
        df: pd.DataFrame,
        n: int = 25,
        title: str = "<b>Feature Impact<b>",
        height: int = 400,
    ):
        df_subset = self.prep_feature_impact(df, n)
        print("PLOTTING FEATURE IMPACT")
        fig = px.bar(
            df_subset,
            y="feature_name",
            x="feature_strength",
            orientation="h",
            height=height,
        )
        fig.update_traces(
            hovertemplate="<b>Feature Name:</b> %{y} <br><b>Feature Strength:</b> %{x}<extra></extra>"
        )

        fig.update_layout(
            title={"text": f"{title}"},
            hoverlabel=DEFAULT_HOVER_LABEL,
        )
        fig.update_yaxes(title="Feature Name")
        fig.update_xaxes(title="Impact")

        return fig

    def plot_signed_feature_impact(
        self,
        df: pd.DataFrame,
        n: int = 25,
        title: str = "<b>Feature Impact<b>",
        height: int = 400,
    ):
        """
        Description

        Attributes:
        -----------
        Inputs

        Methods:
        --------
        Funs

        """
        print("PLOTTING SIGNED FEATURE IMPACT")
        df = df.copy()
        df["positive_strength"] = np.where(
            df["feature_strength"] >= 0, "positive", "negative"
        )
        df = (
            df.groupby(["feature_name", "positive_strength"])["feature_strength"]
            .apply(lambda x: np.abs(x).sum())
            .reset_index()
        )
        df["abs_strength"] = df.groupby("feature_name")["feature_strength"].transform(
            lambda x: np.abs(x).sum()
        )
        strength_index = dict(df.groupby("feature_name")["abs_strength"].sum())

        names_df = pd.DataFrame(
            list(product(df.feature_name.unique(), ["positive", "negative"])),
            columns=["feature_name", "positive_strength"],
        ).assign(tmp=1)

        plot_ready_data = (
            df.merge(names_df, how="outer")
            .drop(columns="tmp")
            .fillna(0)
            .assign(sort_key=lambda x: x.feature_name.map(strength_index))
            .sort_values(by=["sort_key", "positive_strength"], ascending=True)
            .drop(columns=["sort_key"])
            .reset_index(drop=True)
        )
        y = plot_ready_data.feature_name.unique()[-n:]
        x_pos = plot_ready_data.loc[plot_ready_data.positive_strength == "positive"][
            -n:
        ]
        x_neg = plot_ready_data.loc[plot_ready_data.positive_strength == "negative"][
            -n:
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(y=y, x=x_pos.abs_strength, name="Positive Impact", orientation="h")
        )
        fig.add_trace(
            go.Bar(y=y, x=x_neg.abs_strength, name="Negative Impact", orientation="h")
        )

        fig.update_layout(
            barmode="stack",
            margin=dict(l=20, r=0, t=20, b=20),
            height=height,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        fig.update_layout(
            title={"text": f"{title}"},
            hoverlabel=DEFAULT_HOVER_LABEL,
        )
        fig.update_yaxes(title="Feature Name")
        fig.update_xaxes(title="Impact")

        return fig

    def visualize_clusters(
        self,
        df: pd.DataFrame,
        df_cluster: pd.DataFrame,
        standard_embedding: pd.DataFrame,
        record_selection: str = None,
    ):
        print("VISUALIZING CLUSTERS")
        np.random.seed(42)
        df = df.copy()
        df_top_explainers = (
            df.merge(
                df_cluster[["labels"]],
                how="inner",
                left_on="orig_row_num",
                right_index=True,
            )
            .assign(abs_strength=lambda x: np.abs(x["feature_strength"]))
            .sort_values(by=["orig_row_num", "abs_strength"], ascending=False)
            .groupby(["orig_row_num", self.segment_id, "labels"])
            .head(3)
            .reset_index()
        )

        def join_strings(x):
            concate = "".join(
                [
                    f"</br><b>Explanation {i+1}</b>: {a} ({b})"
                    for i, (a, b) in enumerate(zip(x.iloc[:, 0], x.iloc[:, 1]))
                ]
            )
            return "<b>Cluster:</b> " + str(x.iloc[0, 2]) + concate

        top_explain_concatenation = df_top_explainers.groupby("orig_row_num")[
            ["feature_name", "qualitative_strength", "labels"]
        ].apply(join_strings)

        selected_records = (
            df.reset_index()
            .loc[lambda x: x[self.segment_id] == record_selection]
            .index.values
        )
        segment_vals = df.groupby("orig_row_num")[self.segment_id].first()

        standard_embedding["segment"] = standard_embedding["orig_row_num"].map(
            segment_vals
        )
        standard_embedding["text"] = standard_embedding["orig_row_num"].map(
            top_explain_concatenation
        )

        standard_embedding["label"] = standard_embedding["orig_row_num"].map(
            dict(zip(df_cluster.index, df_cluster["labels"]))
        )

        standard_embedding = standard_embedding.assign(
            tooltip=lambda x: "<b>Row Number:</b>"
            + x.orig_row_num.astype(str)
            + "</br><b>"
            + self.segment_id
            + ":<b>"
            + x.segment.astype(str)
            + "<br>"
            + x.text
        ).assign(
            tooltip=lambda x: np.where(
                pd.isna(x.tooltip),
                "<b>Row Number:</b>"
                + x.orig_row_num.astype(str)
                + "</br><b>"
                + self.segment_id
                + ":<b>"
                + x.segment.astype(str)
                + "</br><b>Cluster:</b> "
                + x.label.astype(str),
                x.tooltip,
            )
        )

        # Filter to segment selection:
        if record_selection != self.all_series:
            standard_embedding = standard_embedding.merge(
                df[["orig_row_num", self.segment_id]],
                how="left",
                on="orig_row_num",
            )[lambda x: x[self.segment_id] == record_selection]

        hovertemplate = "</br>%{text}<extra></extra>"
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=standard_embedding["embedding_0"],
                y=standard_embedding["embedding_1"],
                hovertemplate=hovertemplate,
                text=standard_embedding.tooltip,
                showlegend=False,
                mode="markers",
                marker_color=standard_embedding.label.map(self.colorscales),
                marker_line_color="black",
                marker_line_width=1,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[
                    standard_embedding.loc[
                        standard_embedding["orig_row_num"].isin(selected_records),
                        "embedding_0",
                    ]
                ],
                y=[
                    standard_embedding.loc[
                        standard_embedding["orig_row_num"].isin(selected_records),
                        "embedding_1",
                    ]
                ],
                text=standard_embedding.loc[
                    standard_embedding["orig_row_num"].isin(selected_records),
                    "tooltip",
                ],
                hovertemplate=hovertemplate,
                showlegend=False,
                mode="markers",
                marker_symbol="star",
                marker_line_color="midnightblue",
                marker_color="lightskyblue",
                marker_line_width=1,
                marker=dict(size=15),
            )
        )

        fig.update_layout(
            title_text=None,
            height=300,
            plot_bgcolor="#ffffff",
            hoverlabel=DEFAULT_HOVER_LABEL,
            margin=dict(l=20, r=20, t=30, b=20),
        )

        return fig
