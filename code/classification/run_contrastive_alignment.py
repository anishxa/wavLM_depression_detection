import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import pandas as pd

from models import ContrastiveAlignmentNet, SupConLoss

def load_train_val(feature_dir):
    """Loads X_train, y_train, X_val, y_val from feature_dir and concatenates them."""
    X_train = np.load(os.path.join(feature_dir, "X_train.npy"))
    y_train = np.load(os.path.join(feature_dir, "y_train.npy"))
    X_val = np.load(os.path.join(feature_dir, "X_val.npy"))
    y_val = np.load(os.path.join(feature_dir, "y_val.npy"))
    
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    d_all = np.zeros(y_all.shape) # Dummy domain
    
    return TensorDataset(torch.tensor(X_all).float(), torch.tensor(y_all).long(), torch.tensor(d_all).long())

def load_test(feature_dir):
    """Loads X_test, y_test from feature_dir."""
    try:
        X_test = np.load(os.path.join(feature_dir, "X_test.npy"))
        y_test = np.load(os.path.join(feature_dir, "y_test.npy"))
    except FileNotFoundError:
        print(f"Test set not found in {feature_dir}. Falling back to validation set.")
        X_test = np.load(os.path.join(feature_dir, "X_val.npy"))
        y_test = np.load(os.path.join(feature_dir, "y_val.npy"))
        
    d_test = np.zeros(y_test.shape) # Dummy domain
    return TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long(), torch.tensor(d_test).long())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Directory containing train/val features.")
    parser.add_argument("--test_data", type=str, required=True, help="Directory containing test features.")
    parser.add_argument("--exp_name", type=str, default="cross_lingual", help="Prefix for output result files.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lam", type=float, default=0.5, help="Lambda weight for SupConLoss vs CrossEntropy")
    args = parser.parse_args()

    # Load Train Data
    print(f"Loading training features from {args.train_data}...")
    full_train = load_train_val(args.train_data)
    
    # Load Test Data
    print(f"Loading testing features from {args.test_data}...")
    full_test = load_test(args.test_data)

    train_loader = DataLoader(full_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(full_test, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Compute class weights from training labels to handle imbalance
    y_all_tensor = full_train.tensors[1]
    class_counts = torch.bincount(y_all_tensor)
    total = y_all_tensor.size(0)
    class_weights = total / (len(class_counts) * class_counts.float())
    class_weights = class_weights.to(device)
    print(f"Using device: {device}")

    model = ContrastiveAlignmentNet(input_dim=768, proj_dim=128, num_classes=2).to(device)
    
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    criterion_supcon = SupConLoss(temperature=0.07)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Training Loop
    print("\n--- Training ---")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        total_ce = 0
        total_supcon = 0
        for features, labels, domains in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            
            proj, logits = model(features)
            
            # CE Loss
            ce_loss = criterion_ce(logits, labels)
            
            # SupCon Loss requires shape [bsz, n_views, proj_dim]
            # Since we don't have explicit augmented views, we treat the batch itself as views
            # and SupCon aligns features with the same class label
            proj_unsqueezed = proj.unsqueeze(1)
            supcon_loss = criterion_supcon(proj_unsqueezed, labels=labels)
            
            # Joint Loss
            loss = args.lam * supcon_loss + (1 - args.lam) * ce_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_supcon += supcon_loss.item()
            
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f} | CE: {total_ce/len(train_loader):.4f} | SupCon: {total_supcon/len(train_loader):.4f}")

    # Evaluation Loop
    print("\n--- Evaluation ---")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels, domains in test_loader:
            features, labels = features.to(device), labels.to(device)
            _, logits = model(features)
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Try AUC if both classes exist in test set
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
        
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC:  {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    
    report = classification_report(all_labels, all_preds, zero_division=0)
    print(report)
    
    # Save results to file
    os.makedirs("output", exist_ok=True)
    txt_path = os.path.join("output", f"{args.exp_name}_results.txt")
    with open(txt_path, "w") as f:
        f.write("CLeaD Contrastive Alignment Results\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC AUC:  {auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        
    # Generate dynamic report dictionary for CSV (Clean 4-Column Layout)
    report_dict = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)
    csv_data = {
        "Metric": ["Accuracy", "F1 Score", "ROC AUC", "Precision", "Recall", "Support"],
        "Class_0_Healthy": ["", report_dict['0']['f1-score'], "", report_dict['0']['precision'], report_dict['0']['recall'], report_dict['0']['support']],
        "Class_1_Depressed": ["", report_dict['1']['f1-score'], "", report_dict['1']['precision'], report_dict['1']['recall'], report_dict['1']['support']],
        "Overall": [acc, f1, auc, "", "", report_dict['macro avg']['support']]
    }
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join("output", f"{args.exp_name}_results.csv")
    df.to_csv(csv_path, index=False)
        
    print(f"\nResults automatically saved to {txt_path} and {csv_path}")

if __name__ == "__main__":
    main()
