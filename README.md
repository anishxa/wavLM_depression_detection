# Cross-Lingual Depression Detection (WavLM + CLeaD)

This repository contains the codebase for cross-lingual zero-shot depression detection from speech, specifically targeting the transfer gap between Germanic (English) and Tonal (Mandarin) languages. 

This architecture was designed for submission to SLT.

## Architecture

1. **Feature Extractor:** We use `microsoft/wavlm-base-plus` (Layer 6) to extract robust, noise-augmented speech representations.
2. **Classifier (CLeaD):** A custom PyTorch architecture implementing **Contrastive Learning for Depression Detection (CLeaD)**. 
    - The model uses Supervised Contrastive Loss (SupCon) to pull all "Depressed" speech signatures into a shared latent space, forcing the network to ignore the language spoken (English vs. Mandarin) and focus purely on acoustic biomarkers of depression (e.g., psychomotor retardation).

## Datasets
- **E-DAIC:** English corpus used for baseline training and evaluation.
- **MODMA / CMDC:** Mandarin corpora used to validate zero-shot cross-lingual alignment.

*(Note: Massive audio chunks and `.npy` feature arrays are tracked via `.gitignore` and are not included in this repository).*

## How to Run the Pipeline

### 1. Preprocessing
To cut, balance, and segment the transcripts into 10-second sliding windows:
```bash
python3 code/preprocessing/cut_edaic_utterances.py
python3 code/preprocessing/balance_utterance_table.py
python3 code/preprocessing/segment_edaic_sliding.py
python3 code/preprocessing/split_metadata.py --input_csv utterance_table_edaic_segmented.csv
```

### 2. Feature Extraction
To run the segmented `.wav` files through the WavLM transformer:
```bash
python3 code/feature_extraction/extract_edaic_layer.py --metadata utterance_table_edaic_segmented_split.csv --output_dir features/features_edaic_layer6 --layer 6
```
*(Repeat for `extract_cmdc_layer.py` when using Mandarin data).*

### 3. Model Training & Evaluation
To train the CLeaD contrastive alignment network on the extracted features:
```bash
python3 code/classification/run_contrastive_alignment.py
```
*(To run the cross-lingual transfer, simply modify the training script to include `--cmdc_features`).*

## Results
The baseline CLeaD model achieves the following segment-level metrics on the **E-DAIC English-to-English** zero-shot test set:
- **Accuracy:** 92.74%
- **F1 Score:** 0.90
- **ROC AUC:** 0.98
