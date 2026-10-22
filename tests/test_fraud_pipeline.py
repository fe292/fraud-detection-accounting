import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from fraud_pipeline import make_fraud_pipeline, split_raw_data


def sample_data():
    x, y = make_classification(
        n_samples=420, n_features=8, n_informative=5, n_redundant=0,
        weights=[0.95, 0.05], random_state=7,
    )
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(8)])
    df["Class"] = y
    return df


def test_holdout_split_is_disjoint_and_stratified():
    x_train, x_test, y_train, y_test = split_raw_data(sample_data())
    assert set(x_train.index).isdisjoint(set(x_test.index))
    assert len(x_train) + len(x_test) == 420
    assert abs(y_train.mean() - y_test.mean()) < 0.03


def test_scaler_learns_only_from_training_rows():
    x_train, x_test, y_train, _ = split_raw_data(sample_data())
    # Exaggerated test-only values reveal accidental preprocessing leakage.
    x_test = x_test.copy()
    x_test.iloc[:, 0] += 10000
    model = make_fraud_pipeline(LogisticRegression(max_iter=250))
    model.fit(x_train, y_train)
    np.testing.assert_allclose(
        model.named_steps["scaler"].mean_, x_train.mean().to_numpy(), atol=1e-10
    )
    assert not np.allclose(
        model.named_steps["scaler"].mean_,
        pd.concat([x_train, x_test]).mean().to_numpy(),
    )
    assert model.predict_proba(x_test[:5]).shape == (5, 2)


def test_cv_refits_scaler_and_smote_in_each_training_fold():
    x_train, _, y_train, _ = split_raw_data(sample_data())
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fitted = cross_validate(
        make_fraud_pipeline(LogisticRegression(max_iter=250)),
        x_train, y_train, cv=folds,
        scoring={"average_precision": "average_precision", "f1": "f1"},
        return_estimator=True, n_jobs=1,
    )
    assert np.isfinite(fitted["test_average_precision"]).all()
    assert len(fitted["estimator"]) == 3
    for (fit_idx, validation_idx), estimator in zip(
        folds.split(x_train, y_train), fitted["estimator"]
    ):
        expected = x_train.iloc[fit_idx].mean().to_numpy()
        np.testing.assert_allclose(
            estimator.named_steps["scaler"].mean_, expected, atol=1e-10
        )
        assert set(fit_idx).isdisjoint(set(validation_idx))
        assert estimator.named_steps["smote"].sampling_strategy_


def test_target_column_required():
    try:
        split_raw_data(sample_data().drop(columns=["Class"]))
    except ValueError as e:
        assert "Class" in str(e)
    else:
        raise AssertionError("A missing target must raise an error")
