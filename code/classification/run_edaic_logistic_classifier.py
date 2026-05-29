"""
run_edaic_logistic_classifier.py

Trains and evaluates a logistic regression classifier for depression detection
on the EDAIC (Extended Distress Analysis Interview Corpus) dataset.

Pre-extracted WavLM layer features (X_train, X_val, X_test) and binary labels
(y_train, y_val, y_test) are loaded from .npy files. Train and validation sets
are merged before fitting the model. Evaluation metrics (Accuracy, F1, ROC-AUC,
classification report, confusion matrix) are printed to stdout and saved to a
text file. A confusion matrix heatmap can optionally be saved by uncommenting
the matplotlib/seaborn section at the bottom.

Usage:
    python run_edaic_logistic_classifier.py
    (Paths are hard-coded; update feature_dir and output_dir as needed.)
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import random
import torch

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# load data paths
feature_dir = "features/features_edaic_layer6"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# load features and labels
X_train = np.load(os.path.join(feature_dir, "X_train.npy"))
y_train = np.load(os.path.join(feature_dir, "y_train.npy"))
X_val = np.load(os.path.join(feature_dir, "X_val.npy"))
y_val = np.load(os.path.join(feature_dir, "y_val.npy"))
X_test = np.load(os.path.join(feature_dir, "X_test.npy"))
y_test = np.load(os.path.join(feature_dir, "y_test.npy"))

# concatenate train + val
X_all = np.concatenate([X_train, X_val], axis=0)
y_all = np.concatenate([y_train, y_val], axis=0)

# build and train the model
model = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000)
model.fit(X_all, y_all)

# predict and evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

# print results
print("\nEDAIC Logistic Regression Results")
print(f"Accuracy:  {acc:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {auc:.4f}")
print("\nConfusion Matrix:")
print(cm)

# save results to file
with open(os.path.join(output_dir, "edaic_l9_results.txt"), "w") as f:
    f.write("EDAIC Logistic Regression Results\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"ROC-AUC: {auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

# plot confusion matrix
#plt.figure(figsize=(5,4))
#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#plt.xlabel("Predicted")
#plt.ylabel("Actual")
#plt.title("Confusion Matrix")
#plt.savefig(os.path.join(output_dir, "edaic_confusion_matrix.png"))

print("\nResults saved to output/edaic_results.txt")
