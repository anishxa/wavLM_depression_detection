import os
import re
import pandas as pd

def analyze_split(csv_path, dataset_name):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Extract participant/speaker ID from the filename using regex
    # (first sequence of digits in the file basename)
    df["participant_id"] = df["file_path"].apply(lambda x: re.search(r'\d+', os.path.basename(x)).group())
    
    print(f"\n==========================================")
    print(f"Dataset: {dataset_name} ({csv_path})")
    print(f"Total Rows (Segments): {len(df)}")
    
    # Speaker level info
    speaker_df = df.drop_duplicates(subset=["participant_id"])
    total_speakers = len(speaker_df)
    mdd_speakers = len(speaker_df[speaker_df["label"] == 1])
    hc_speakers = len(speaker_df[speaker_df["label"] == 0])
    
    print(f"Total Unique Speakers: {total_speakers} (MDD: {mdd_speakers}, HC: {hc_speakers})")
    
    # Breakdown by split
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        split_speaker_df = split_df.drop_duplicates(subset=["participant_id"])
        
        num_segments = len(split_df)
        seg_mdd = len(split_df[split_df["label"] == 1])
        seg_hc = len(split_df[split_df["label"] == 0])
        
        num_speakers = len(split_speaker_df)
        spk_mdd = len(split_speaker_df[split_speaker_df["label"] == 1])
        spk_hc = len(split_speaker_df[split_speaker_df["label"] == 0])
        
        print(f"\n  Split: '{split.upper()}'")
        print(f"    Speakers: {num_speakers} (MDD/Depressed: {spk_mdd}, HC/Healthy: {spk_hc})")
        print(f"    Segments: {num_segments} (MDD/Depressed: {seg_mdd}, HC/Healthy: {seg_hc})")

analyze_split("utterance_table_modma_segmented_split.csv", "MODMA / MODMA")
analyze_split("utterance_table_edaic_segmented_split.csv", "E-DAIC")
analyze_split("utterance_table_mix_segmented_split.csv", "MIX")
