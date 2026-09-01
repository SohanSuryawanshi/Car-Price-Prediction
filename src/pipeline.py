"""
Assembles the full leakage-safe preprocessing + model pipeline.

Import build_pipeline() from here in train.py / predict.py rather than
redefining the pipeline structure in multiple places.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor

from . import config
from .transformers import (
    DomainSanityClipper,
    IQRClipper,
    CarAgeEngineer,
    RareCategoryGrouper,
)


def build_numeric_pipeline() -> Pipeline:
    """impute -> domain sanity clip -> car_age engineering -> IQR clip -> scale"""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("sanity_clip", DomainSanityClipper()),
        ("car_age", CarAgeEngineer()),
        ("outlier_clip", IQRClipper(factor=1.5)),
        ("scaler", StandardScaler()),
    ])


def build_categorical_pipeline() -> Pipeline:
    """impute (explicit 'unknown') -> group rare categories -> one-hot encode."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("rare_grouper", RareCategoryGrouper(top_n=config.RARE_CATEGORY_TOP_N)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", build_numeric_pipeline(), config.RAW_NUMERIC),
        ("cat", build_categorical_pipeline(), config.CATEGORICAL),
    ])


def build_pipeline() -> TransformedTargetRegressor:
    """Full pipeline: preprocessing -> feature selection -> model,
    wrapped to train/predict on log1p(price) and return raw-dollar predictions.
    """
    feature_pipeline = Pipeline(steps=[
        ("preprocessing", build_preprocessor()),
        ("feature_selection", SelectKBest(score_func=f_regression,
                                           k=config.FEATURE_SELECTION_K)),
        ("model", RandomForestRegressor(
            n_estimators=config.N_ESTIMATORS,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    return TransformedTargetRegressor(
        regressor=feature_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )
