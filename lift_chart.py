import datarobot as dr
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from helpers import DEFAULT_HOVER_LABEL
from inference import BatchScoring


class LiftChart(BatchScoring):
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
        **kwargs,
    ):
        super().__init__(project_id, model_id, deployment_id, **kwargs)

    def add_bins_to_data(
        self,
        df: pd.DataFrame,
        bins: int = 10,
        data_subset: str = dr.enums.DATA_SUBSET.HOLDOUT,
        weights: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        self.bins = bins
        self.data_subset = data_subset
        self.weights = weights

        col = list(self._get_column_name_mappings().values())[0]
        # Create Quantiles
        if weights:
            df = df.loc[~np.isnan(df["Prediction"]), :].reset_index(drop=True)
            df["bins"] = weighted_qcut(
                df[col],
                df["weight"],
                q=bins,
                labels=False,
                duplicates="drop",
            )
            mean = lambda x: np.average(x, weights=df.loc[x.index, "weight"])
        else:
            df = df.loc[~pd.isna(df["Prediction"]), :].reset_index(drop=True)
            df["bins"] = pd.qcut(
                df[col],
                q=bins,
                labels=False,
                duplicates="drop",
            )
            mean = lambda x: np.average(x)

        self.func = {col: mean, "actuals": mean}

        return df

    def group_data_by_bin(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df["actuals"] = np.where(df[self.target] == self.project.positive_class, 1, 0)

        binned_data = df.groupby(["bins"]).agg(self.func).reset_index()
        binned_data["bins"] = range(1, self.bins + 1)
        return binned_data

    def plot_lift_chart(
        self,
        df: pd.DataFrame,
    ):
        col = list(self._get_column_name_mappings().values())[0]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["bins"],
                y=df[col],
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
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["bins"],
                y=df["actuals"],  # Change this marshall :)
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
            )
        )

        title = "Weighted " if self.weights else ""
        fig.update_layout(
            title={
                "text": f"<b>Lift Chart</b>",
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            legend=dict(
                bgcolor="rgba(255, 255, 255, 0)",
                bordercolor="rgba(255, 255, 255, 0)",
            ),
            legend_title=f"Data: ",
            hoverlabel=DEFAULT_HOVER_LABEL,
        )
        fig.update_yaxes(title=f"{title}Average Target")
        fig.update_xaxes(title=f"{title}Bins")

        return fig
