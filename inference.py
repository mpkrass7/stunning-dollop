import datarobot as dr
import pandas as pd
import streamlit as st


class BatchScoring:
    """
    Description

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
    ):
        self.project_id = project_id
        self.project = dr.Project.get(self.project_id)
        self.model_id = model_id
        self.model = dr.Model.get(self.project_id, self.model_id)
        self.target = self.project.target
        self.deployment_id = deployment_id
        self.project_type = self.project.target_type

    def _get_column_name_mappings(self):
        if self.project_type == "regression":
            name_remappings = {
                f"{self.target}_PREDICTION": "Prediction",
            }
        else:
            name_remappings = {
                f"{self.target}_{self.project.positive_class}_PREDICTION": "Class_1_Prediction",
                # f"{self.target}_True_PREDICTION": "Class_1_Prediction",
                f"{self.target}_PREDICTION": "Prediction",
            }

        return name_remappings

    def get_batch_predictions(
        self,
        df: pd.DataFrame,
        max_explanations: int = 20,
        max_wait: int = 1000,
    ) -> pd.DataFrame:

        print(f"Retrieving {max_explanations} prediction explanations")

        name_remappings = self._get_column_name_mappings()

        _, df = dr.BatchPredictionJob.score_pandas(
            df=df,
            deployment=self.deployment_id,
            max_explanations=max_explanations,
            download_timeout=max_wait,
            column_names_remapping=name_remappings,
        )
        df["Class_0_Prediction"] = 1.0 - df["Class_1_Prediction"]

        return df
