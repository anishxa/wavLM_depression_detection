import os
import re
import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import WavLMModel
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} for accelerated ablation feature extraction")

# Load WavLM Model
print("Loading microsoft/wavlm-base-plus model...")
model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus", output_hidden_states=True).to(device).eval()

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.file_paths = df["file_path"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            waveform, sr = torchaudio.load(path)
            # Downmix stereo to mono
            if waveform.shape[0] == 2:
                waveform = waveform.mean(dim=0, keepdim=True)
            # Resample if not 16000
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            waveform = waveform.squeeze(0)
            
            # Standardize length to exactly 48000 samples (3.0 seconds at 16kHz)
            if waveform.shape[0] < 48000:
                pad_len = 48000 - waveform.shape[0]
                waveform = torch.cat([waveform, torch.zeros(pad_len)], dim=0)
            elif waveform.shape[0] > 48000:
                waveform = waveform[:48000]
                
            return waveform, label, path
        except Exception as e:
            # Return silence on error so we don't crash
            return torch.zeros(48000), label, path

def extract_for_dataset(metadata_csv, dataset_name, batch_size=32):
    if not os.path.exists(metadata_csv):
        print(f"Metadata CSV not found: {metadata_csv}")
        return
        
    df = pd.read_csv(metadata_csv)
    print(f"\n==========================================")
    print(f"Processing Dataset: {dataset_name} ({len(df)} rows)")
    
    # Initialize output structures for layers 7, 8, 9
    layers = [7, 8, 9]
    for layer in layers:
        os.makedirs(f"features/features_{dataset_name}_layer{layer}", exist_ok=True)
        
    for split in ["train", "val", "test"]:
        df_split = df[df["split"] == split].reset_index(drop=True)
        if len(df_split) == 0:
            continue
            
        print(f"Extracting {split} split ({len(df_split)} items)...")
        dataset = SpeechDataset(df_split)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Prepare list to accumulate features for each layer
        X_accum = {layer: [] for layer in layers}
        y_accum = []
        
        with torch.no_grad():
            for waveforms, labels, paths in tqdm(dataloader, desc=f"{dataset_name} - {split}"):
                waveforms = waveforms.to(device)
                
                # Perform standard batch normalization on GPU
                mean = waveforms.mean(dim=-1, keepdim=True)
                var = waveforms.var(dim=-1, keepdim=True)
                waveforms = (waveforms - mean) / torch.sqrt(var + 1e-7)
                
                # Forward pass
                out = model(input_values=waveforms)
                
                # Extract pooled features for each layer
                for layer in layers:
                    layer_feat = out.hidden_states[layer]
                    pooled = layer_feat.mean(dim=1).cpu().numpy()  # shape (batch_size, 768)
                    X_accum[layer].append(pooled)
                    
                y_accum.append(labels.numpy())
                
        # Stack and save features for each layer
        y_stacked = np.concatenate(y_accum, axis=0)
        for layer in layers:
            X_stacked = np.concatenate(X_accum[layer], axis=0)
            X_path = f"features/features_{dataset_name}_layer{layer}/X_{split}.npy"
            y_path = f"features/features_{dataset_name}_layer{layer}/y_{split}.npy"
            np.save(X_path, X_stacked)
            np.save(y_path, y_stacked)
            print(f"  Layer {layer} saved: X shape {X_stacked.shape}, y shape {y_stacked.shape}")

if __name__ == "__main__":
    # Run extraction on all three datasets
    extract_for_dataset("utterance_table_modma_segmented_split.csv", "modma")
    extract_for_dataset("utterance_table_mix_segmented_split.csv", "mix")
    extract_for_dataset("utterance_table_edaic_segmented_split.csv", "edaic")

    print("\nAll feature extractions completed successfully!")
