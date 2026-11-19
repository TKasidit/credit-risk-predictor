"""Credit risk predictor: a Gradio app for a Hugging Face Space.
Loads the tuned random forest from Assignment 8.6 and predicts the
probability that a credit card customer will default next month."""
import json

import gradio as gr
import joblib
import pandas as pd

model = joblib.load("model.joblib")                     # the trained model
info = json.load(open("feature_info.json"))             # feature names, defaults, descriptions
FEATURES = info["features"]
examples = json.load(open("examples.json"))             # a few real test customers


def predict(*values):
    """Turn the 23 inputs into one table row, and return the model's prediction."""
    X = pd.DataFrame([values], columns=FEATURES)
    p_default = float(model.predict_proba(X)[0, 1])
    if p_default >= 0.5:
        decision = f"High risk: likely to default ({p_default:.0%} probability)"
    else:
        decision = f"Low risk: likely to pay ({p_default:.0%} probability of default)"
    return {"Default": p_default, "No default": 1 - p_default}, decision


inputs = [gr.Number(label=f, value=info["defaults"][f], info=info["descriptions"].get(f, ""))
          for f in FEATURES]

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=[gr.Label(label="Predicted probability"), gr.Textbox(label="Decision")],
    examples=examples,
    title="Credit Card Default Risk Predictor",
    description=("Tuned random forest trained on the UCI Default of Credit Card Clients dataset "
                 "(Yeh, 2009). Enter a customer's details, or click an example below. "
                 "For coursework demonstration only, not for real credit decisions."),
)

if __name__ == "__main__":
    demo.launch()
