"""
build_mixed_metadata.py

Builds a mixed-language metadata CSV by combining equal-sized samples from
the CMD-C (Chinese) and EDAIC (English) segmented split tables.

The total target size is fixed at 7712 utterances, split 60/20/20 into
train/val/test. Each split draws half its samples from CMD-C and half from
EDAIC (stratified random sampling within each split), and a 'language' column
is appended to track the source language ('zh' or 'en'). The resulting
DataFrame is saved to utterance_table_mix_segmented_split.csv.

Inputs:
    utterance_table_cmdc_segmented_split.csv -- CMD-C split metadata.
    utterance_table_edaic_segmented_split.csv -- EDAIC split metadata.
    Both CSVs must contain columns 'file_path', 'label', and 'split'.

Output:
    utterance_table_mix_segmented_split.csv -- Combined metadata with an
    additional 'language' column ('zh' or 'en').
"""

import pandas as pd
import os

# source file paths
cmdc_path = "/scratch/s5944562/WavLM/utterance_table_cmdc_segmented_split.csv"
edaic_path = "/scratch/s5944562/WavLM/utterance_table_edaic_segmented_split.csv"
out_path = "/scratch/s5944562/WavLM/utterance_table_mix_segmented_split.csv"

# load data
df_cmdc = pd.read_csv(cmdc_path)
df_edaic = pd.read_csv(edaic_path)

# total count and split ratios
TOTAL = 7712
train_n = int(0.6 * TOTAL)  # 4627
val_n = int(0.2 * TOTAL)    # 1542
test_n = TOTAL - train_n - val_n  # 1543
half_train = train_n // 2
half_val = val_n // 2
half_test = test_n // 2

# random sampling (stratified within language splits)
df_train = pd.concat([
    df_cmdc[df_cmdc["split"] == "train"].sample(n=half_train, random_state=42),
    df_edaic[df_edaic["split"] == "train"].sample(n=half_train, random_state=42)
])
df_val = pd.concat([
    df_cmdc[df_cmdc["split"] == "val"].sample(n=half_val, random_state=42),
    df_edaic[df_edaic["split"] == "val"].sample(n=half_val, random_state=42)
])
df_test = pd.concat([
    df_cmdc[df_cmdc["split"] == "test"].sample(n=half_test, random_state=42),
    df_edaic[df_edaic["split"] == "test"].sample(n=half_test, random_state=42)
])

# add language labels
df_train["language"] = ["zh"] * half_train + ["en"] * half_train
df_val["language"] = ["zh"] * half_val + ["en"] * half_val
df_test["language"] = ["zh"] * half_test + ["en"] * half_test

# merge and save
df_mix = pd.concat([df_train, df_val, df_test], ignore_index=True)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df_mix.to_csv(out_path, index=False)
print(f"mixed metadata saved to {out_path} with {len(df_mix)} rows (60/20/20 split of 7712 total)")
