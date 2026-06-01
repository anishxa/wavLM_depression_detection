"""
extract_edaic_layer.py

Extracts WavLM hidden-state features from EDAIC audio utterances and saves
them as numpy arrays for downstream classification.

The script loads the pre-trained facebook/wavlm-base-ls960 model and iterates
over the train/val/test splits defined in a metadata CSV provided via command
line. For each audio file it resamples to 16 kHz if needed, downmixes stereo
to mono, runs the waveform through WavLM, and mean-pools the hidden states of
the specified layer to produce a 768-dimensional feature vector.

This script detects Apple Silicon (MPS) or CUDA GPUs to accelerate extraction.
"""

import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm
import argparse

# === command-line arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, default=6, help="WavLM layer index to extract")
parser.add_argument("--metadata", type=str, required=True, help="Path to CSV metadata file")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for features")
args = parser.parse_args()

# === device selection ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} for feature extraction")

# === load model and feature extractor ===
model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus", output_hidden_states=True).to(device).eval()
extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")

os.makedirs(args.output_dir, exist_ok=True)


def extract_features(df_split, split_name):
    """Extract and save WavLM features for a single data split.

    Iterates over all rows in df_split, loads each audio file, preprocesses
    the waveform (stereo downmix and resampling), and extracts the mean-pooled
    hidden state from the WavLM layer specified by args.layer. Results are
    appended to X_{split_name}.npy and y_{split_name}.npy incrementally,
    allowing the job to resume from a previous checkpoint if those files exist.
    """
    X_list, y_list = [], []

    X_path = os.path.join(args.output_dir, f"X_{split_name}.npy")
    y_path = os.path.join(args.output_dir, f"y_{split_name}.npy")

    # === resume support ===
    processed_files = set()
    if os.path.exists(X_path) and os.path.exists(y_path):
        print(f"resuming {split_name} from saved features...")
        X_list = list(np.load(X_path))
        y_list = list(np.load(y_path))
        processed_files = set(df_split.iloc[:len(X_list)]["file_path"])  # already processed files

    for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Extracting {split_name}"):
        path = row["file_path"]
        label = row["label"]

        if path in processed_files:
            continue

        try:
            waveform, sr = torchaudio.load(path)

            if waveform.shape[0] == 2:
                waveform = waveform.mean(dim=0, keepdim=True)

            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

            inputs = extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
                layer_feat = out.hidden_states[args.layer]
                pooled = layer_feat.mean(dim=1).squeeze().cpu().numpy()

            if pooled.shape == (768,):
                X_list.append(pooled)
                y_list.append(label)
            else:
                print(f"skipped {path} due to invalid shape {pooled.shape}")

        except Exception as e:
            print(f"failed on {path}: {e}")

    np.save(X_path, np.array(X_list))
    np.save(y_path, np.array(y_list))
    print(f"saved {split_name}: {len(X_list)} samples")


# === main ===
df = pd.read_csv(args.metadata)
for split in ["train", "val", "test"]:
    df_split = df[df["split"] == split]
    extract_features(df_split, split)
