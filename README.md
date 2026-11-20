---
title: Credit Card Default Risk Predictor
emoji: 💳
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
---

# Credit Card Default Risk Predictor

A static web app that serves the tuned random forest from my AAI6610 credit risk project
(class_weight = balanced, max_depth = 10, min_samples_leaf = 5, n_estimators = 100).
The trained model was exported to JSON, and the predictions run in the visitor's browser,
so the app works on Hugging Face's free Static Space tier.

Data: Yeh, I.-C. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C55S3H

For coursework demonstration only, not for real credit decisions.
