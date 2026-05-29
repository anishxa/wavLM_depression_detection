"""
split_metadata.py

Splits a metadata CSV into train/val/test subsets using a Speaker-Independent Split.
This prevents Speaker Identity Leakage by ensuring that all segments from a single
participant end up entirely in Train, Val, or Test, and are never mixed.

Args (command-line):
    --input_csv (str):    Path to the input metadata CSV (required). Must
                          contain columns 'file_path' and 'label'.
    --output_csv (str):   Path for the output CSV (optional). Defaults to
                          input filename with '_split' appended before .csv.
    --train_ratio (float): Fraction of data assigned to 'train' (default: 0.6).
    --val_ratio (float):  Fraction of data assigned to 'val' (default: 0.2).

Output:
    A CSV file identical to the input but with an additional 'split' column
    ('train', 'val', or 'test') for each row.
"""

import os
import re
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", type=str, required=True, help="Path to input metadata CSV")
parser.add_argument("--output_csv", type=str, default=None, help="Path to output CSV with split column")
parser.add_argument("--train_ratio", type=float, default=0.6)
parser.add_argument("--val_ratio", type=float, default=0.2)
args = parser.parse_args()

# load metadata
df = pd.read_csv(args.input_csv)

# Extract Participant ID from the filename to use as the Group ID.
# Uses regex to find the first sequence of digits in the filename (e.g. "300" in "edaic_chunks_300_1.wav").
df["participant_id"] = df["file_path"].apply(lambda x: re.search(r'\d+', os.path.basename(x)).group())

# step 1: train vs val+test (Group Split to prevent speaker leakage)
gss_train = GroupShuffleSplit(n_splits=1, train_size=args.train_ratio, random_state=42)
train_idx, temp_idx = next(gss_train.split(df, groups=df["participant_id"]))

df_train = df.iloc[train_idx].copy()
df_temp = df.iloc[temp_idx].copy()

# step 2: val vs test (split from temp)
val_ratio_relative = args.val_ratio / (1 - args.train_ratio)
gss_val = GroupShuffleSplit(n_splits=1, train_size=val_ratio_relative, random_state=42)
val_idx, test_idx = next(gss_val.split(df_temp, groups=df_temp["participant_id"]))

df_val = df_temp.iloc[val_idx].copy()
df_test = df_temp.iloc[test_idx].copy()

# add split labels
df_train["split"] = "train"
df_val["split"] = "val"
df_test["split"] = "test"

# merge, shuffle, and clean up the temporary participant_id column
df_split = pd.concat([df_train, df_val, df_test]).sample(frac=1, random_state=42)
df_split = df_split.drop(columns=["participant_id"])

# save output
output_path = args.output_csv or args.input_csv.replace(".csv", "_split.csv")
df_split.to_csv(output_path, index=False)
print(f"saved Speaker-Independent split metadata to: {output_path}")
