"""
prepare_local_modma.py

Scans the local MODMA dataset directory layout (audio_lanzhou_2015)
and maps subjects to their depression label (MDD=1, HC=0) from the Excel sheet.
It records files whose duration falls within [MIN_SEC, MAX_SEC] and outputs
the utterance CSV (utterance_table_modma_updated.csv).
"""

import os
import csv
import pandas as pd
import torchaudio
from tqdm import tqdm

root_dir = "audio_lanzhou_2015"
excel_path = os.path.join(root_dir, "subjects_information_audio_lanzhou_2015.xlsx")
output_csv = "utterance_table_modma_updated.csv"

# Load the Excel file to get label mapping
df_excel = pd.read_excel(excel_path)
df_excel["folder_name"] = df_excel["subject id"].apply(lambda x: f"0{x}")

label_map = dict(zip(df_excel["folder_name"], df_excel["type"]))

# Duration filter (seconds)
MIN_SEC = 2
MAX_SEC = 60

data = []

# Walk through each folder in audio_lanzhou_2015
for folder in tqdm(sorted(os.listdir(root_dir)), desc="Scanning local MODMA folders"):
    folder_path = os.path.join(root_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    
    # Check if folder name is mapped to a subject label
    if folder not in label_map:
        continue
    
    # Map MDD -> 1, HC -> 0
    raw_label = label_map[folder]
    if raw_label == "MDD":
        label = 1
    elif raw_label == "HC":
        label = 0
    else:
        print(f"Skipping unknown type {raw_label} for speaker {folder}")
        continue
        
    # Scan wav files
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            wav_path = os.path.join(folder_path, file)
            try:
                waveform, sr = torchaudio.load(wav_path)
                duration = waveform.shape[1] / sr
                if MIN_SEC <= duration <= MAX_SEC:
                    data.append({"file_path": wav_path, "label": label})
            except Exception as e:
                print(f"failed to read {wav_path}: {e}")

# Write to CSV
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["file_path", "label"])
    writer.writeheader()
    writer.writerows(data)

print(f"Successfully processed {len(data)} entries and saved to {output_csv}")
