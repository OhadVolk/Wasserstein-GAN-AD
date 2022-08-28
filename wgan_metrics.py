import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from metrics import get_metrics

df = pd.read_csv("results/score.csv")
df
trainig_label = 1
labels = np.where(df["label"].values == trainig_label, 0, 1)
anomaly_score = df["anomaly_score"].values
probas = df["img_distance"].values
preds = np.where(probas >= 0.5, 1, 0)
z_distance = df["z_distance"].values

print(roc_curve(labels, probas))