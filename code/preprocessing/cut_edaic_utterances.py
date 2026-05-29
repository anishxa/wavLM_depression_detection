"""
cut_edaic_utterances.py

Segments EDAIC participant audio files into individual utterances using
transcript timing information and exports each segment as a .wav file.

For each valid participant ID in the EDAIC dataset, the script reads the
corresponding transcript CSV (which contains Start_Time and End_Time columns)
and the full-session audio file. Utterances shorter than 200 ms are skipped.
Each surviving segment is exported to a dedicated output directory and logged
in utterance_table.csv along with its binary depression label (PHQ-8 binary
score mapped from metadata_mapped.csv).

Requires ffmpeg to be available; the path is hard-coded via
AudioSegment.converter.

Inputs:
    <base_dir>/<pid>/<pid>_P/<pid>_Transcript.csv -- Utterance timing.
    <base_dir>/<pid>/<pid>_P/<pid>_AUDIO.wav      -- Full session audio.
    metadata_mapped.csv -- Participant-level labels with columns
        'Participant_ID' and 'PHQ_Binary'.

Output:
    <output_dir>/<pid>_<row_index>.wav -- Individual utterance audio files.
    utterance_table.csv -- Metadata CSV with columns 'file_path' and 'label'.
"""

import os
import pandas as pd
import csv
from pydub import AudioSegment
AudioSegment.converter = "ffmpeg"

# directory paths
base_dir = "/Users/anishapattanayak/Documents/SLT/Dep_Det/WavLM_Depression_Detection/wwwedaic/data"
output_dir = "/Users/anishapattanayak/Documents/SLT/Dep_Det/WavLM_Depression_Detection/edaic_chunks"
os.makedirs(output_dir, exist_ok=True)

# load label metadata
label_df = pd.read_csv("/Users/anishapattanayak/Documents/SLT/Dep_Det/WavLM_Depression_Detection/wwwedaic/metadata_mapped.csv")
label_dict = dict(zip(label_df["Participant_ID"], label_df["PHQ_Binary"]))

# exclude missing participants
valid_ids = [i for i in range(300, 426) if i not in [342, 394, 398]]

# store metadata
metadata = []

for pid in valid_ids:
    try:
        transcript_path = f"{base_dir}/{pid}_P/{pid}_Transcript.csv"
        audio_path = f"{base_dir}/{pid}_P/{pid}_AUDIO.wav"

        if not os.path.exists(transcript_path) or not os.path.exists(audio_path):
            print(f"skipping {pid} - missing transcript or audio")
            continue

        df = pd.read_csv(transcript_path)
        audio = AudioSegment.from_wav(audio_path)
        # df = df[df["Speaker"] == "Participant"]  # no Speaker column, skip this step
        df = df.dropna(subset=["Start_Time", "End_Time"])  # keep only rows with valid timestamps

        for i, row in df.iterrows():
            start_ms = int(float(row["Start_Time"]) * 1000)
            end_ms = int(float(row["End_Time"]) * 1000)
            if end_ms - start_ms < 200:
                continue

            utt = audio[start_ms:end_ms]
            utt_path = f"{output_dir}/{pid}_{i}.wav"
            utt.export(utt_path, format="wav")

            label = label_dict.get(pid)
            if label is not None:
                metadata.append({"file_path": utt_path, "label": label})

        print(f"processed {pid}")

    except Exception as e:
        print(f"error with {pid}: {e}")

# write utterance_table.csv
with open("utterance_table.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["file_path", "label"])
    writer.writeheader()
    writer.writerows(metadata)
print("finished writing utterance_table.csv")
print(f"total utterances extracted: {len(metadata)}")
print(f"total participants processed: {len(set([os.path.basename(row['file_path']).split('_')[0] for row in metadata]))}")
