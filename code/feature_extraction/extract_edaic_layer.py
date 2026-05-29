"""
extract_edaic_layer.py

Extracts WavLM hidden-state features from EDAIC audio utterances and saves
them as numpy arrays for downstream classification.

The script loads the pre-trained facebook/wavlm-base-ls960 model and iterates
over the train/val/test splits defined in a metadata CSV provided via command
line. For each audio file it resamples to 16 kHz if needed, downmixes stereo
to mono, runs the waveform through WavLM, and mean-pools the hidden states of
the specified layer to produce a 768-dimensional feature vector. Partial
progress is saved after each split so the job can be safely resumed if
interrupted.

Args (command-line):
    --layer (int):        Index of the WavLM hidden layer to extract (default: 6).
    --metadata (str):     Path to the CSV metadata file (required). Must contain
                          columns 'file_path', 'label', and 'split'.
    --output_dir (str):   Directory where output .npy files will be written (required).

Outputs:
    X_{split}.npy  -- Feature matrix of shape (N, 768) for each split.
    y_{split}.npy  -- Binary label array of shape (N,) for each split.
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

# === load model and feature extractor ===
model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus", output_hidden_states=True).eval()
extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")

os.makedirs(args.output_dir, exist_ok=True)


def extract_features(df_split, split_name):
    """Extract and save WavLM features for a single data split.

    Iterates over all rows in df_split, loads each audio file, preprocesses
    the waveform (stereo downmix and resampling), and extracts the mean-pooled
    hidden state from the WavLM layer specified by args.layer. Results are
    appended to X_{split_name}.npy and y_{split_name}.npy incrementally,
    allowing the job to resume from a previous checkpoint if those files exist.

    Args:
        df_split (pd.DataFrame): Metadata slice for one split. Must contain
            columns 'file_path' (str) and 'label' (int, binary 0/1).
        split_name (str): Name of the split, e.g. 'train', 'val', or 'test'.
            Used to name the output .npy files.

    Side effects:
        Writes X_{split_name}.npy and y_{split_name}.npy to args.output_dir.
        Prints progress and any skipped/failed files to stdout.
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
            with torch.no_grad():
                out = model(**inputs)
                layer_feat = out.hidden_states[args.layer]
                pooled = layer_feat.mean(dim=1).squeeze().numpy()

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
