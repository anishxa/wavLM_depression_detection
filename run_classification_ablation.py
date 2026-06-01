import os
import re
import sys
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Add code/classification to sys.path
sys.path.append(os.path.join(os.getcwd(), "code", "classification"))
from models import ContrastiveAlignmentNet, SupConLoss

device = torch.device("cpu")
print(f"Using device for CLeaD training: {device}")

# Path to datasets
metadata_modma_path = "utterance_table_modma_segmented_split.csv"

# Load metadata to map segments back to speaker/participant IDs for MODMA speaker evaluation
modma_metadata = pd.read_csv(metadata_modma_path)
modma_test_df = modma_metadata[modma_metadata["split"] == "test"].copy()
modma_test_df["speaker_id"] = modma_test_df["file_path"].apply(lambda x: re.search(r'\d+', os.path.basename(x)).group())

def evaluate_speaker_level(preds_segment, probs_segment, df_test):
    df = df_test.copy()
    df["pred"] = preds_segment
    df["prob"] = probs_segment
    
    speaker_results = []
    for spk_id, group in df.groupby("speaker_id"):
        true_label = group["label"].iloc[0]
        # Majority voting
        maj_vote = 1 if group["pred"].mean() >= 0.5 else 0
        # Average probability
        avg_prob = group["prob"].mean()
        prob_vote = 1 if avg_prob >= 0.5 else 0
        
        speaker_results.append({
            "speaker_id": spk_id,
            "true_label": true_label,
            "maj_vote": maj_vote,
            "avg_prob": avg_prob,
            "prob_vote": prob_vote
        })
        
    df_spk = pd.DataFrame(speaker_results)
    
    acc_maj = accuracy_score(df_spk["true_label"], df_spk["maj_vote"])
    f1_maj = f1_score(df_spk["true_label"], df_spk["maj_vote"], zero_division=0)
    acc_prob = accuracy_score(df_spk["true_label"], df_spk["prob_vote"])
    f1_prob = f1_score(df_spk["true_label"], df_spk["prob_vote"], zero_division=0)
    
    try:
        auc_spk = roc_auc_score(df_spk["true_label"], df_spk["avg_prob"])
    except:
        auc_spk = 0.5
        
    num_correct_mdd = int(df_spk[(df_spk["true_label"] == 1) & (df_spk["maj_vote"] == 1)].shape[0])
    num_correct_hc = int(df_spk[(df_spk["true_label"] == 0) & (df_spk["maj_vote"] == 0)].shape[0])
    
    return f"{num_correct_mdd}/5 MDD, {num_correct_hc}/5 HC", f1_maj, acc_maj, auc_spk

def run_baseline(train_dir, test_dir, exp_name):
    # Train Logistic Regression
    X_train = np.concatenate([
        np.load(os.path.join(train_dir, "X_train.npy")),
        np.load(os.path.join(train_dir, "X_val.npy"))
    ], axis=0)
    y_train = np.concatenate([
        np.load(os.path.join(train_dir, "y_train.npy")),
        np.load(os.path.join(train_dir, "y_val.npy"))
    ], axis=0)
    
    X_test = np.load(os.path.join(test_dir, "X_test.npy"))
    y_test = np.load(os.path.join(test_dir, "y_test.npy"))
    
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs)
    except:
        auc = 0.5
        
    # If testing on MODMA, also calculate speaker metrics
    spk_str, spk_f1, spk_acc, spk_auc = "-", 0.0, 0.0, 0.0
    if "modma" in test_dir:
        spk_str, spk_f1, spk_acc, spk_auc = evaluate_speaker_level(preds, probs, modma_test_df)
        
    return acc, f1, auc, spk_str, spk_f1, spk_acc, spk_auc

def train_clead(train_dir, test_dir, epochs=30, batch_size=32):
    X_train = np.concatenate([
        np.load(os.path.join(train_dir, "X_train.npy")),
        np.load(os.path.join(train_dir, "X_val.npy"))
    ], axis=0)
    y_train = np.concatenate([
        np.load(os.path.join(train_dir, "y_train.npy")),
        np.load(os.path.join(train_dir, "y_val.npy"))
    ], axis=0)
    
    X_test = np.load(os.path.join(test_dir, "X_test.npy"))
    y_test = np.load(os.path.join(test_dir, "y_test.npy"))
    
    full_train = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long(), torch.zeros(y_train.shape).long())
    train_loader = DataLoader(full_train, batch_size=batch_size, shuffle=True, drop_last=True)
    
    class_counts = torch.bincount(torch.tensor(y_train).long())
    class_weights = len(y_train) / (len(class_counts) * class_counts.float())
    class_weights = class_weights.to(device)
    
    model = ContrastiveAlignmentNet(input_dim=768, proj_dim=128, num_classes=2).to(device)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    criterion_supcon = SupConLoss(temperature=0.07)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    model.train()
    for epoch in range(epochs):
        for features, labels, _ in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            proj, logits = model(features)
            ce_loss = criterion_ce(logits, labels)
            proj_unsqueezed = proj.unsqueeze(1)
            supcon_loss = criterion_supcon(proj_unsqueezed, labels=labels)
            loss = 0.5 * supcon_loss + 0.5 * ce_loss
            loss.backward()
            optimizer.step()
            
    model.eval()
    all_preds = []
    all_probs = []
    
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long()), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(device)
            _, logits = model(features)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    preds = np.array(all_preds)
    probs = np.array(all_probs)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs)
    except:
        auc = 0.5
        
    # If testing on MODMA, also calculate speaker metrics
    spk_str, spk_f1, spk_acc, spk_auc = "-", 0.0, 0.0, 0.0
    if "modma" in test_dir:
        spk_str, spk_f1, spk_acc, spk_auc = evaluate_speaker_level(preds, probs, modma_test_df)
        
    return acc, f1, auc, spk_str, spk_f1, spk_acc, spk_auc

def main():
    configs = [
        {"name": "EN -> EN", "train": "edaic", "test": "edaic"},
        {"name": "EN -> ZH", "train": "edaic", "test": "modma"},
        {"name": "ZH -> EN", "train": "modma", "test": "edaic"},
        {"name": "ZH -> ZH", "train": "modma", "test": "modma"},
        {"name": "MIX -> EN", "train": "mix", "test": "edaic"},
        {"name": "MIX -> ZH", "train": "mix", "test": "modma"}
    ]
    
    results = []
    
    # 1. Parse Layer 6 results from existing files if available
    print("\nParsing existing Layer 6 results...")
    for cfg in configs:
        train_ds = cfg["train"]
        test_ds = cfg["test"]
        cfg_name = cfg["name"]
        
        # Format names like existing ones (e.g. baseline_en_to_en_results.csv)
        train_norm = "mix" if train_ds == "mix" else ("en" if train_ds == "edaic" else "zh")
        test_norm = "mix" if test_ds == "mix" else ("en" if test_ds == "edaic" else "zh")
        
        # Baseline
        base_csv = f"output/baseline_{train_norm}_to_{test_norm}_results.csv"
        acc_b, f1_b, auc_b = 0.0, 0.0, 0.0
        if os.path.exists(base_csv):
            df_csv = pd.read_csv(base_csv)
            # Accuracy is at row index 1 (Accuracy), column 'Overall'
            acc_b = float(df_csv.loc[df_csv["Metric"] == "Accuracy", "Overall"].values[0])
            f1_b = float(df_csv.loc[df_csv["Metric"] == "F1 Score", "Overall"].values[0])
            auc_b = float(df_csv.loc[df_csv["Metric"] == "ROC AUC", "Overall"].values[0])
            
        # CLeaD
        clead_csv = f"output/clead_{train_norm}_to_{test_norm}_results.csv"
        if train_ds == "edaic" and test_ds == "edaic":
            # Wait, edaic is sometimes named clead_edaic_results.csv or clead_en_to_en_results.csv
            if not os.path.exists(clead_csv) and os.path.exists("output/clead_edaic_results.csv"):
                clead_csv = "output/clead_edaic_results.csv"
        
        acc_c, f1_c, auc_c = 0.0, 0.0, 0.0
        if os.path.exists(clead_csv):
            df_csv = pd.read_csv(clead_csv)
            acc_c = float(df_csv.loc[df_csv["Metric"] == "Accuracy", "Overall"].values[0])
            f1_c = float(df_csv.loc[df_csv["Metric"] == "F1 Score", "Overall"].values[0])
            auc_c = float(df_csv.loc[df_csv["Metric"] == "ROC AUC", "Overall"].values[0])
            
        # Get speaker level results for layer 6 using stored features
        spk_b, spk_c = "-", "-"
        if test_ds == "modma":
            # Run speaker level evaluation for Baseline
            try:
                _, _, _, spk_b, _, _, _ = run_baseline(f"features/features_{train_ds}_layer6", f"features/features_{test_ds}_layer6", "")
                _, _, _, spk_c, _, _, _ = train_clead(f"features/features_{train_ds}_layer6", f"features/features_{test_ds}_layer6")
            except Exception as e:
                print(f"Failed to calculate layer 6 speaker details for {cfg_name}: {e}")
                
        results.append({
            "Layer": 6,
            "Config": cfg_name,
            "Model": "Baseline",
            "Accuracy": acc_b,
            "F1": f1_b,
            "AUC": auc_b,
            "Speaker_Vote": spk_b
        })
        results.append({
            "Layer": 6,
            "Config": cfg_name,
            "Model": "CLeaD",
            "Accuracy": acc_c,
            "F1": f1_c,
            "AUC": auc_c,
            "Speaker_Vote": spk_c
        })
        
    # 2. Run for Layers 7, 8, 9
    for layer in [7, 8, 9]:
        print(f"\n--- Evaluating Layer {layer} ---")
        for cfg in configs:
            cfg_name = cfg["name"]
            train_ds = cfg["train"]
            test_ds = cfg["test"]
            
            train_dir = f"features/features_{train_ds}_layer{layer}"
            test_dir = f"features/features_{test_ds}_layer{layer}"
            
            if not os.path.exists(os.path.join(train_dir, "X_train.npy")) or not os.path.exists(os.path.join(test_dir, "X_test.npy")):
                print(f"Features for Layer {layer} ({cfg_name}) not found. Skipping.")
                continue
                
            print(f"  Running Baseline for {cfg_name} (Layer {layer})...")
            acc_b, f1_b, auc_b, spk_b, _, _, _ = run_baseline(train_dir, test_dir, f"baseline_{train_ds}_to_{test_ds}_layer{layer}")
            results.append({
                "Layer": layer,
                "Config": cfg_name,
                "Model": "Baseline",
                "Accuracy": acc_b,
                "F1": f1_b,
                "AUC": auc_b,
                "Speaker_Vote": spk_b
            })
            
            print(f"  Running CLeaD for {cfg_name} (Layer {layer})...")
            acc_c, f1_c, auc_c, spk_c, _, _, _ = train_clead(train_dir, test_dir)
            results.append({
                "Layer": layer,
                "Config": cfg_name,
                "Model": "CLeaD",
                "Accuracy": acc_c,
                "F1": f1_c,
                "AUC": auc_c,
                "Speaker_Vote": spk_c
            })
            
            print(f"    Baseline: Segment F1={f1_b:.4f}, AUC={auc_b:.4f} | Speaker={spk_b}")
            print(f"    CLeaD:    Segment F1={f1_c:.4f}, AUC={auc_c:.4f} | Speaker={spk_c}")
            
    # Save all results to a CSV table
    df_res = pd.DataFrame(results)
    df_res.to_csv("output/ablation_raw_results.csv", index=False)
    print("\nSaved raw results to output/ablation_raw_results.csv")
    
    # Compile a beautiful markdown table
    md_content = "# Layer Ablation Study Results (Layers 6, 7, 8, 9)\n\n"
    md_content += "This table summarizes the segment-level classification metrics and speaker-level majority vote results across all four WavLM layers.\n\n"
    md_content += "| Layer | Config | Model | Segment Accuracy | Segment F1 | Segment AUC | Speaker Vote (MDD/HC Correct) |\n"
    md_content += "| :---: | :--- | :--- | :---: | :---: | :---: | :--- |\n"
    
    for _, row in df_res.iterrows():
        md_content += f"| {row['Layer']} | {row['Config']} | {row['Model']} | {row['Accuracy']:.4f} | {row['F1']:.4f} | {row['AUC']:.4f} | {row['Speaker_Vote']} |\n"
        
    with open("output/ablation_summary.md", "w") as f:
        f.write(md_content)
        
    print("Saved beautiful summary table to output/ablation_summary.md")
    
if __name__ == "__main__":
    main()
