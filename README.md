---
title: Credit Card Default Risk Predictor
emoji: 💳
colorFrom: blue
colorTo: green
sdk: gradio
python_version: "3.11"
app_file: app.py
pinned: false
---

# Credit Card Default Risk Predictor

A Gradio app that serves the tuned random forest from my AAI6610 credit risk project
(class_weight = balanced, max_depth = 10, min_samples_leaf = 5, n_estimators = 100).
It predicts the probability that a customer will default on next month's payment.

Data: Yeh, I.-C. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C55S3H

For coursework demonstration only, not for real credit decisions.
