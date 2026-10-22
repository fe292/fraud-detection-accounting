"""Leakage-safe preprocessing for the credit card fraud example.

All stateful preprocessing is part of the estimator pipeline, so
cross-validation fits StandardScaler and SMOTE only on each training fold.
"""
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


def split_raw_data(dataframe, *, random_state=42, test_size=0.2):
    """Make one untouched, stratified holdout split BEFORE any fitting."""
    if "Class" not in dataframe.columns:
        raise ValueError("Expected the binary target column 'Class'.")
    X = dataframe.drop(columns=["Class"])
    y = dataframe["Class"]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def make_fraud_pipeline(estimator, *, random_state=42, smote_ratio=0.10):
    """A new unfitted estimator with preprocessing contained in each CV fold.

    The 0.10 ratio avoids manufacturing hundreds of thousands of synthetic
    observations; it is a modeling choice, NOT an established optimum.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("smote", SMOTE(sampling_strategy=smote_ratio, random_state=random_state)),
            ("model", clone(estimator)),
        ]
    )
