"""Basic checks that run in GitHub Actions before every deployment (the CI step)."""
import json

import joblib
import pandas as pd

model = joblib.load("model.joblib")
info = json.load(open("feature_info.json"))
examples = json.load(open("examples.json"))


def test_model_uses_23_features():
    assert len(info["features"]) == 23
    assert list(model.feature_names_in_) == info["features"]


def test_predictions_are_valid_probabilities():
    X = pd.DataFrame(examples, columns=info["features"])
    p = model.predict_proba(X)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_default_input_gives_a_prediction():
    X = pd.DataFrame([info["defaults"]])[info["features"]]
    assert model.predict(X)[0] in (0, 1)
