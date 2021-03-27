import pickle
import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

import joblib


def get_data(path):
    _ROOT = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(_ROOT, "data", path)

data = pd.read_csv(get_data("sample.csv"))

def predict(data, model="calibrated_classifier_20200408_1552.pkl"):
    """"""
    model = joblib.load(get_data(model))
    data_initial = data[
        ["lowreslimit", "anomalousslope", "anomalousCC", "diffI", "diffF", "f"]
    ]
data_initial = data_initial.fillna(0)

unknown = data_initial.values

for line in unknown:
    y_pred = self.model.predict(line.reshape(1, -1))
    y_pred_proba = self.model.predict_proba(line.reshape(1, -1))
    fail_prob = round(y_pred_proba[0][0], 4) * 100
    succ_prob = round(y_pred_proba[0][1], 4) * 100
    y_pred_adj = [1 if x >= 0.9317 else 0 for x in y_pred_proba[:, 1]]

    print("Probability for experimental phasing outcome:")
    print("Failure: %s" % fail_prob)
    print("Success: %s" % succ_prob)
    print(
        "Predicted class after applying threshold 80.00%% for class 1: %s \n"
        % str(y_pred_adj)
    )
    print("*" * 80)

# make predictions
ynew = model.predict(unknown)
# show the inputs and predicted outputs
for i in range(len(unknown)):
    print("X=%s, Predicted=%s" % (unknown[i], ynew[i]))

# make probabilistic predictions
ynew_proba = model.predict_proba(unknown)
# show the inputs and predicted probabilities
for i in range(len(unknown)):
    print("X=%s, Predicted=%s" % (unknown[i], ynew_proba[i]))
