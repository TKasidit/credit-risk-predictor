"""CI checks that run in GitHub Actions before every deployment."""
import json

import numpy as np

model = json.load(open("model.json"))
info = json.load(open("feature_info.json"))
examples = json.load(open("examples.json"))


def predict(row):
    """The same tree-following logic as the website (index.html)."""
    probs = []
    for t in model["trees"]:
        node = 0
        while t["l"][node] != -1:
            x = np.float32(row[t["f"][node]])
            node = t["l"][node] if x <= np.float32(t["t"][node]) else t["r"][node]
        probs.append(t["p"][node])
    return float(np.mean(probs))


def test_model_has_23_features_and_100_trees():
    assert model["features"] == info["features"]
    assert len(info["features"]) == 23
    assert len(model["trees"]) == 100


def test_predictions_match_scikit_learn():
    for row, expected in zip(examples["rows"], examples["expected"]):
        assert abs(predict(row) - expected) < 1e-4


def test_default_input_gives_a_valid_probability():
    p = predict([info["defaults"][f] for f in info["features"]])
    assert 0 <= p <= 1
