"""
balance_utterance_table.py

Creates a class-balanced utterance table for the EDAIC dataset by
random under-sampling the majority class (label 0, healthy controls).

Reads an existing utterance CSV, separates the two classes, and down-samples
the healthy-control rows so that both classes have the same count as the
depression class (label 1). The shuffled, balanced DataFrame is saved to
utterance_table_balanced.csv.

Input:
    /scratch/.../edaic/utterance_table.csv -- CSV with columns 'file_path'
    and 'label' (0 = HC, 1 = depressed).

Output:
    utterance_table_balanced.csv -- Balanced CSV with equal class counts.
"""

import pandas as pd

# read metadata
df = pd.read_csv("utterance_table.csv")

# separate classes
df_0 = df[df["label"] == 0]
df_1 = df[df["label"] == 1]

# down-sample class 0 to match class 1
df_0_sampled = df_0.sample(n=len(df_1), random_state=42)

# merge and shuffle
df_balanced = pd.concat([df_0_sampled, df_1]).sample(frac=1, random_state=42)

# save balanced table
df_balanced.to_csv("utterance_table_balanced.csv", index=False)
