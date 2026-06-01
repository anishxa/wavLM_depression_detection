"""
create_modma_utterance_table.py

Scans the CMD-C dataset directory structure and builds an utterance-level
metadata CSV, filtering out audio files that are too short or too long.

Walks the HC and MDD subdirectories under root_dir, checks the duration of
every .wav file using torchaudio, and records files whose duration falls
within [MIN_SEC, MAX_SEC]. Each entry stores the absolute file path and a
binary label (0 = HC, 1 = MDD). The resulting table is written to
utterance_table_modma_updated.csv.

Expected directory layout:
    root_dir/
        HC/
            <speaker_id>/
                *.wav
        MDD/
            <speaker_id>/
                *.wav

Output:
    utterance_table_modma_updated.csv -- CSV with columns 'file_path' and
    'label' (0 = HC, 1 = MDD), filtered to [MIN_SEC, MAX_SEC] duration.
"""

import os
import csv
import torchaudio

# set paths and output file
root_dir = "/scratch/s5944562/WavLM/datasets/modma"
output_csv = "utterance_table_modma_updated.csv"
data = []

# duration filter (seconds)
MIN_SEC = 2
MAX_SEC = 60  # raised from 30 to keep more audio data

# two classes: HC=0, MDD=1
for group, label in [("HC", 0), ("MDD", 1)]:
    group_dir = os.path.join(root_dir, group)
    for speaker in sorted(os.listdir(group_dir)):
        speaker_path = os.path.join(group_dir, speaker)
        if not os.path.isdir(speaker_path):
            continue
        for file in os.listdir(speaker_path):
            if file.endswith(".wav"):
                wav_path = os.path.join(speaker_path, file)
                try:
                    duration = torchaudio.info(wav_path).num_frames / torchaudio.info(wav_path).sample_rate
                    if MIN_SEC <= duration <= MAX_SEC:
                        data.append({"file_path": wav_path, "label": label})
                except Exception as e:
                    print(f"failed to read {wav_path}: {e}")

# write to CSV
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["file_path", "label"])
    writer.writeheader()
    writer.writerows(data)

print(f"filtered and wrote {len(data)} entries to {output_csv}")
