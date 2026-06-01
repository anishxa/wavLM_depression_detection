import os
import shutil
import re

def replace_in_file(file_path):
    if not os.path.exists(file_path):
        return
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Preserve case: cmdc -> modma, CMDC -> MODMA
    new_content = re.sub(r'cmdc', 'modma', content)
    new_content = re.sub(r'CMDC', 'MODMA', new_content)
    # Also replace any direct links/paths inside files
    
    if new_content != content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"  Updated contents of: {file_path}")

def rename_file_or_dir(src, dst):
    if os.path.exists(src):
        if os.path.exists(dst):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
        os.rename(src, dst)
        print(f"  Renamed: {src} -> {dst}")

def main():
    print("=== Starting CMDC to MODMA Refactoring ===\n")
    
    # 1. Update file contents for all python, markdown, and csv files first
    files_to_update = [
        "README.md",
        "extract_ablation_features.py",
        "run_classification_ablation.py",
        "get_speaker_level_metrics.py",
        "inspect_splits.py",
        "compile_layer6.py",
        "utterance_table_cmdc_segmented.csv",
        "utterance_table_cmdc_segmented_split.csv",
        "utterance_table_cmdc_balanced.csv",
        "utterance_table_cmdc_updated.csv",
        "utterance_table_mix_segmented_split.csv",
        "code/preprocessing/segment_cmdc_sliding.py",
        "code/feature_extraction/extract_cmdc_layer.py",
        "code/preprocessing/create_cmdc_utterance_table.py",
        "code/preprocessing/create_cmdc_balanced_table.py",
        "code/preprocessing/prepare_local_cmdc.py",
        "code/preprocessing/build_mixed_metadata.py",
        "code/preprocessing/split_metadata.py"
    ]
    
    for f in files_to_update:
        replace_in_file(f)
        
    # 2. Rename directories
    print("\nRenaming Directories...")
    for layer in [6, 7, 8, 9]:
        rename_file_or_dir(f"features/features_cmdc_layer{layer}", f"features/features_modma_layer{layer}")
    rename_file_or_dir("cmdc_segments", "modma_segments")
    
    # 3. Rename files
    print("\nRenaming Files...")
    rename_file_or_dir("utterance_table_cmdc_segmented.csv", "utterance_table_modma_segmented.csv")
    rename_file_or_dir("utterance_table_cmdc_segmented_split.csv", "utterance_table_modma_segmented_split.csv")
    rename_file_or_dir("utterance_table_cmdc_balanced.csv", "utterance_table_modma_balanced.csv")
    rename_file_or_dir("utterance_table_cmdc_updated.csv", "utterance_table_modma_updated.csv")
    
    rename_file_or_dir("code/preprocessing/segment_cmdc_sliding.py", "code/preprocessing/segment_modma_sliding.py")
    rename_file_or_dir("code/feature_extraction/extract_cmdc_layer.py", "code/feature_extraction/extract_modma_layer.py")
    rename_file_or_dir("code/preprocessing/create_cmdc_utterance_table.py", "code/preprocessing/create_modma_utterance_table.py")
    rename_file_or_dir("code/preprocessing/create_cmdc_balanced_table.py", "code/preprocessing/create_modma_balanced_table.py")
    rename_file_or_dir("code/preprocessing/prepare_local_cmdc.py", "code/preprocessing/prepare_local_modma.py")
    
    print("\n=== Refactoring Completed Successfully! ===")

if __name__ == "__main__":
    main()
