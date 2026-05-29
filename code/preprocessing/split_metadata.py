"""
split_metadata.py

Splits a metadata CSV into stratified train/val/test subsets and appends a
'split' column indicating each row's assignment.

Uses scikit-learn's train_test_split with stratification on the 'label'
column to ensure class balance is preserved across splits. The split ratios
are configurable via command-line arguments (default: 60/20/20). The combined
DataFrame (with the 'split' column added) is shuffled and written to a new
CSV file.

Args (command-line):
    --input_csv (str):    Path to the input metadata CSV (required). Must
                          contain columns 'file_path' and 'label'.
    --output_csv (str):   Path for the output CSV (optional). Defaults to
                          input filename with '_split' appended before .csv.
    --train_ratio (float): Fraction of data assigned to 'train' (default: 0.6).
    --val_ratio (float):  Fraction of data assigned to 'val' (default: 0.2).
                          The remainder is assigned to 'test'.

Output:
    A CSV file identical to the input but with an additional 'split' column
    ('train', 'val', or 'test') for each row.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", type=str, required=True, help="Path to input metadata CSV")
parser.add_argument("--output_csv", type=str, default=None, help="Path to output CSV with split column")
parser.add_argument("--train_ratio", type=float, default=0.6)
parser.add_argument("--val_ratio", type=float, default=0.2)
args = parser.parse_args()

# load metadata (must include file_path and label columns)
df = pd.read_csv(args.input_csv)

# step 1: train vs val+test
df_train, df_temp = train_test_split(
    df, stratify=df["label"], test_size=(1 - args.train_ratio), random_state=42
)

# step 2: val vs test (split from temp)
val_size = args.val_ratio / (1 - args.train_ratio)
df_val, df_test = train_test_split(
    df_temp, stratify=df_temp["label"], test_size=(1 - val_size), random_state=42
)

# add split labels
df_train["split"] = "train"
df_val["split"] = "val"
df_test["split"] = "test"

# merge and shuffle
df_split = pd.concat([df_train, df_val, df_test]).sample(frac=1, random_state=42)

# save output
output_path = args.output_csv or args.input_csv.replace(".csv", "_split.csv")
df_split.to_csv(output_path, index=False)
print(f"saved split metadata to: {output_path}")
