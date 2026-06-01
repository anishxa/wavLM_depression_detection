"""
split_metadata.py

Splits a metadata CSV into train/val/test subsets using a Stratified Speaker-Independent Split.
This prevents Speaker Identity Leakage by ensuring that all segments from a single
participant end up entirely in Train, Val, or Test. It uses StratifiedGroupKFold
to ensure the ratio of healthy vs. depressed participants is perfectly balanced across splits.

Args (command-line):
    --input_csv (str):    Path to the input metadata CSV (required). Must
                          contain columns 'file_path' and 'label'.
    --output_csv (str):   Path for the output CSV (optional). Defaults to
                          input filename with '_split' appended before .csv.

Output:
    A CSV file identical to the input but with an additional 'split' column
    ('train', 'val', or 'test') for each row.
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", type=str, required=True, help="Path to input metadata CSV")
parser.add_argument("--output_csv", type=str, default=None, help="Path to output CSV with split column")
args = parser.parse_args()

# load metadata
df = pd.read_csv(args.input_csv)

# Extract Participant ID from the filename to use as the Group ID.
# Uses regex to find the first sequence of digits in the filename (e.g. "300" in "edaic_chunks_300_1.wav").
df["participant_id"] = df["file_path"].apply(lambda x: re.search(r'\d+', os.path.basename(x)).group())

# We use 5-fold Stratified Group Split. Each fold gets 20% of the participants.
# Fold 0 = Test (20%)
# Fold 1 = Val (20%)
# Folds 2, 3, 4 = Train (60%)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(sgkf.split(df, df["label"], groups=df["participant_id"]))

test_idx = folds[0][1]
val_idx = folds[1][1]
train_idx = np.concatenate([folds[2][1], folds[3][1], folds[4][1]])

# add split labels
df["split"] = ""
df.loc[train_idx, "split"] = "train"
df.loc[val_idx, "split"] = "val"
df.loc[test_idx, "split"] = "test"

# merge, shuffle, and clean up the temporary participant_id column
df_split = df.sample(frac=1, random_state=42).drop(columns=["participant_id"])

# save output
output_path = args.output_csv or args.input_csv.replace(".csv", "_split.csv")
df_split.to_csv(output_path, index=False)
print(f"saved Stratified Speaker-Independent split metadata to: {output_path}")
