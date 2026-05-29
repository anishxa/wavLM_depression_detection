"""
create_cmdc_balanced_table.py

Creates a class-balanced utterance table for the CMD-C dataset by
random under-sampling both classes to match the size of the smaller class.

Reads utterance_table_cmdc_updated.csv, determines the minority class count
(min of HC and MDD), down-samples both classes to that count, shuffles the
result, and saves it to utterance_table_cmdc_balanced.csv. Unlike
balance_utterance_table.py (which targets the EDAIC dataset and only
down-samples the majority class), this script balances both classes symmetrically.

Input:
    utterance_table_cmdc_updated.csv -- CSV with columns 'file_path' and
    'label' (0 = HC, 1 = MDD).

Output:
    utterance_table_cmdc_balanced.csv -- Balanced CSV with equal class counts.
"""

import pandas as pd

# load the updated CSV
df = pd.read_csv("utterance_table_cmdc_updated.csv")

# separate the two classes
df_0 = df[df["label"] == 0]  # HC
df_1 = df[df["label"] == 1]  # MDD

# determine balance count from smaller class
min_count = min(len(df_0), len(df_1))

# down-sample both classes to min_count
df_0_sampled = df_0.sample(n=min_count, random_state=42)
df_1_sampled = df_1.sample(n=min_count, random_state=42)

# merge and shuffle
df_balanced = pd.concat([df_0_sampled, df_1_sampled]).sample(frac=1, random_state=42)

# save balanced table
df_balanced.to_csv("utterance_table_cmdc_balanced.csv", index=False)

print(f"balanced set created: {min_count} samples per class, total {len(df_balanced)}")
