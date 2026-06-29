"""Model training and inference helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


@dataclass
class ModelArtifacts:
    model: Pipeline
    feature_columns: list[str]


def build_model(random_state: int = 42) -> Pipeline:
    """Build a robust baseline regressor pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, FEATURE_COLUMNS)]
    )

    regressor = GradientBoostingRegressor(
        random_state=random_state,
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        subsample=0.8,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


def train_model(train_df: pd.DataFrame, random_state: int = 42) -> ModelArtifacts:
    """Train model with historical data."""
    model = build_model(random_state=random_state)
    model.fit(train_df[FEATURE_COLUMNS], train_df["target_next_intraday"])
    return ModelArtifacts(model=model, feature_columns=FEATURE_COLUMNS.copy())


def predict(df: pd.DataFrame, artifacts: ModelArtifacts) -> pd.Series:
    """Predict next-day intraday return."""
    predictions = artifacts.model.predict(df[artifacts.feature_columns])
    return pd.Series(predictions, index=df.index, name="predicted_return")
