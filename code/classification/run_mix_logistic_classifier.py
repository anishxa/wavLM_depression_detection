"""
run_mix_logistic_classifier.py

Trains and evaluates a logistic regression classifier for depression detection
on the mixed (CMD-C + EDAIC) dataset.

Pre-extracted WavLM layer features (X_train, X_val, X_test) and binary labels
(y_train, y_val, y_test) are loaded from .npy files. Train and validation sets
are merged before fitting the model. Evaluation metrics (F1, Accuracy, ROC-AUC,
classification report, confusion matrix) are written to a text file.

Usage:
    python run_mix_logistic_classifier.py
    (Paths are hard-coded; update feature_dir and output_file as needed.)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report, confusion_matrix
import os
import random
import torch

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# set directory
feature_dir = "/scratch/s5944562/WavLM/features_mix_layer9"
output_file = "/scratch/s5944562/WavLM/output/mix_l9_results.txt"
os.makedirs("/scratch/s5944562/WavLM/output", exist_ok=True)

# load the files
y_train = np.load(os.path.join(feature_dir, "y_train.npy"))
y_val   = np.load(os.path.join(feature_dir, "y_val.npy"))
y_test  = np.load(os.path.join(feature_dir, "y_test.npy"))

X_train = np.load(os.path.join(feature_dir, "X_train.npy")).reshape(y_train.shape[0], -1)
X_val   = np.load(os.path.join(feature_dir, "X_val.npy")).reshape(y_val.shape[0], -1)
X_test  = np.load(os.path.join(feature_dir, "X_test.npy")).reshape(y_test.shape[0], -1)

# concatenate train + val
X_all = np.concatenate([X_train, X_val], axis=0)
y_all = np.concatenate([y_train, y_val], axis=0)

# train the model
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_all, y_all)

# evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

f1 = f1_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# outputs
with open(output_file, "w") as f:
    f.write("MIX Logistic Regression Results\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"ROC-AUC: {auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))

print("Evaluation complete. Results saved to", output_file)
