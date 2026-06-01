import os
import re
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# Add code/classification to path to import models
sys.path.append(os.path.join(os.getcwd(), "code", "classification"))
from models import ContrastiveAlignmentNet, SupConLoss

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load test metadata to map segments back to speaker/participant IDs
modma_metadata = pd.read_csv("utterance_table_modma_segmented_split.csv")
modma_test_df = modma_metadata[modma_metadata["split"] == "test"].copy()
modma_test_df["speaker_id"] = modma_test_df["file_path"].apply(lambda x: re.search(r'\d+', os.path.basename(x)).group())

print(f"Total test segments in MODMA: {len(modma_test_df)}")
speakers_list = modma_test_df["speaker_id"].tolist()
labels_list = modma_test_df["label"].tolist()

# Quick speaker-level mapping
unique_speaker_metadata = modma_test_df.drop_duplicates(subset=["speaker_id"])
print("Test Speakers:")
for _, r in unique_speaker_metadata.iterrows():
    seg_cnt = len(modma_test_df[modma_test_df["speaker_id"] == r["speaker_id"]])
    print(f"  Speaker ID: {r['speaker_id']}, Label: {'MDD' if r['label'] == 1 else 'HC'}, Segments: {seg_cnt}")

# Define evaluation function
def evaluate_speaker_level(preds_segment, probs_segment, df_test, label_str):
    df = df_test.copy()
    df["pred"] = preds_segment
    df["prob"] = probs_segment
    
    speaker_results = []
    for spk_id, group in df.groupby("speaker_id"):
        true_label = group["label"].iloc[0]
        seg_count = len(group)
        
        # Majority voting
        maj_vote = 1 if group["pred"].mean() >= 0.5 else 0
        
        # Average probability
        avg_prob = group["prob"].mean()
        prob_vote = 1 if avg_prob >= 0.5 else 0
        
        speaker_results.append({
            "speaker_id": spk_id,
            "true_label": true_label,
            "seg_count": seg_count,
            "maj_vote": maj_vote,
            "avg_prob": avg_prob,
            "prob_vote": prob_vote
        })
        
    df_spk = pd.DataFrame(speaker_results)
    
    # Calculate metrics
    acc_maj = accuracy_score(df_spk["true_label"], df_spk["maj_vote"])
    f1_maj = f1_score(df_spk["true_label"], df_spk["maj_vote"], zero_division=0)
    
    acc_prob = accuracy_score(df_spk["true_label"], df_spk["prob_vote"])
    f1_prob = f1_score(df_spk["true_label"], df_spk["prob_vote"], zero_division=0)
    
    try:
        auc_spk = roc_auc_score(df_spk["true_label"], df_spk["avg_prob"])
    except:
        auc_spk = 0.0
        
    print(f"\n[{label_str}] SPEAKER-LEVEL RESULTS:")
    for _, r in df_spk.iterrows():
        print(f"  Speaker {r['speaker_id']} ({'MDD' if r['true_label'] == 1 else 'HC'}): "
              f"Vote Pred={r['maj_vote']} | Prob Pred={r['prob_vote']} (Avg Prob={r['avg_prob']:.4f}) | Segments={r['seg_count']}")
    
    print(f"Speaker-Level Metrics (Majority Vote): F1={f1_maj:.4f}, Acc={acc_maj:.4f}")
    print(f"Speaker-Level Metrics (Prob Vote): F1={f1_prob:.4f}, Acc={acc_prob:.4f}, AUC={auc_spk:.4f}")

# ----------------- BASELINE LOGISTIC REGRESSION -----------------
# 1. ZH -> ZH Baseline
X_train_modma = np.concatenate([
    np.load("features/features_modma_layer6/X_train.npy"),
    np.load("features/features_modma_layer6/X_val.npy")
], axis=0)
y_train_modma = np.concatenate([
    np.load("features/features_modma_layer6/y_train.npy"),
    np.load("features/features_modma_layer6/y_val.npy")
], axis=0)

X_test_modma = np.load("features/features_modma_layer6/X_test.npy")
y_test_modma = np.load("features/features_modma_layer6/y_test.npy")

print("\n\nTraining ZH -> ZH Baseline Logistic Regression...")
clf_zh = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
clf_zh.fit(X_train_modma, y_train_modma)
preds_zh_lr = clf_zh.predict(X_test_modma)
probs_zh_lr = clf_zh.predict_proba(X_test_modma)[:, 1]

evaluate_speaker_level(preds_zh_lr, probs_zh_lr, modma_test_df, "ZH -> ZH Baseline LR")

# 2. MIX -> ZH Baseline
X_train_mix = np.concatenate([
    np.load("features/features_mix_layer6/X_train.npy"),
    np.load("features/features_mix_layer6/X_val.npy")
], axis=0)
y_train_mix = np.concatenate([
    np.load("features/features_mix_layer6/y_train.npy"),
    np.load("features/features_mix_layer6/y_val.npy")
], axis=0)

print("\n\nTraining MIX -> ZH Baseline Logistic Regression...")
clf_mix = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
clf_mix.fit(X_train_mix, y_train_mix)
preds_mix_lr = clf_mix.predict(X_test_modma)
probs_mix_lr = clf_mix.predict_proba(X_test_modma)[:, 1]

evaluate_speaker_level(preds_mix_lr, probs_mix_lr, modma_test_df, "MIX -> ZH Baseline LR")


# ----------------- CLEAD CONTRASTIVE ALIGNMENT -----------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device for CLeaD: {device}")

def train_clead(X_train, y_train, X_test, y_test, epochs=30, batch_size=32):
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
            
    return np.array(all_preds), np.array(all_probs)

# 3. ZH -> ZH CLeaD
print("\n\nTraining ZH -> ZH CLeaD...")
preds_zh_clead, probs_zh_clead = train_clead(X_train_modma, y_train_modma, X_test_modma, y_test_modma)
evaluate_speaker_level(preds_zh_clead, probs_zh_clead, modma_test_df, "ZH -> ZH CLeaD")

# 4. MIX -> ZH CLeaD
print("\n\nTraining MIX -> ZH CLeaD...")
preds_mix_clead, probs_mix_clead = train_clead(X_train_mix, y_train_mix, X_test_modma, y_test_modma)
evaluate_speaker_level(preds_mix_clead, probs_mix_clead, modma_test_df, "MIX -> ZH CLeaD")
