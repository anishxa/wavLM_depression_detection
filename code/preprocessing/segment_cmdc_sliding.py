"""
segment_cmdc_sliding.py

Applies a sliding-window segmentation to CMD-C utterance audio files,
producing fixed-length overlapping segments for feature extraction.

Reads a balanced utterance table CSV, loads each .wav file, resamples to
16 kHz if necessary, and slices the waveform into 3-second windows with a
1.5-second stride (50% overlap). Files shorter than one window are skipped.
Each segment is saved as a .wav file and logged in a new metadata CSV that
can be used directly by the feature extraction scripts.

Input:
    utterance_table_cmdc_balanced.csv -- CSV with columns 'file_path' and
    'label' pointing to the balanced CMD-C utterance audio files.

Output:
    <output_dir>/<parent>_<base>_seg<i>.wav -- Individual segment audio files.
    utterance_table_cmdc_segmented.csv -- Metadata CSV with columns
    'file_path' and 'label' for every generated segment.
"""

import os
import torchaudio
from tqdm import tqdm
import pandas as pd

# parameters
source_csv = "/scratch/s5944562/WavLM/cmdc/utterance_table_cmdc_balanced.csv"
output_dir = "/scratch/s5944562/WavLM/datasets/cmdc_segments"
window_length = 3.0  # seconds
stride = 1.5         # seconds (50% overlap)
target_sr = 16000

# create output directory
os.makedirs(output_dir, exist_ok=True)

# load table
df = pd.read_csv(source_csv)
segments = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    wav_path = row["file_path"]
    label = row["label"]

    try:
        waveform, sr = torchaudio.load(wav_path)
        if sr != target_sr:
            resample = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resample(waveform)
            sr = target_sr

        duration = waveform.shape[1] / sr
        if duration < window_length:
            continue  # skip if too short

        step = int(stride * sr)
        size = int(window_length * sr)
        total_samples = waveform.shape[1]
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        parent_name = os.path.basename(os.path.dirname(wav_path))

        for i, start in enumerate(range(0, total_samples - size + 1, step)):
            end = start + size
            segment = waveform[:, start:end]
            seg_name = f"{parent_name}_{base_name}_seg{i}.wav"
            seg_path = os.path.join(output_dir, seg_name)
            torchaudio.save(seg_path, segment, sr)
            segments.append({"file_path": seg_path, "label": label})

    except Exception as e:
        print(f"failed on {wav_path}: {e}")

# save new metadata
seg_df = pd.DataFrame(segments)
seg_df.to_csv("utterance_table_cmdc_segmented.csv", index=False)
print(f"done! total segments: {len(seg_df)}")
