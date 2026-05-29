import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

from models import ContrastiveAlignmentNet, SupConLoss

def load_dataset(feature_dir, domain_label=0):
    """Loads X and y from feature_dir. Returns TensorDataset with (features, labels, domain)."""
    X_train = np.load(os.path.join(feature_dir, "X_train.npy"))
    y_train = np.load(os.path.join(feature_dir, "y_train.npy"))
    X_val = np.load(os.path.join(feature_dir, "X_val.npy"))
    y_val = np.load(os.path.join(feature_dir, "y_val.npy"))
    try:
        X_test = np.load(os.path.join(feature_dir, "X_test.npy"))
        y_test = np.load(os.path.join(feature_dir, "y_test.npy"))
    except FileNotFoundError:
        print("Test set not found. Falling back to validation set for evaluation.")
        X_test, y_test = X_val, y_val
    
    # Train + Val for full training
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    
    d_all = np.full(y_all.shape, domain_label)
    d_test = np.full(y_test.shape, domain_label)
    
    train_dataset = TensorDataset(torch.tensor(X_all).float(), torch.tensor(y_all).long(), torch.tensor(d_all).long())
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long(), torch.tensor(d_test).long())
    
    return train_dataset, test_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edaic_features", type=str, default="features/features_edaic_layer6")
    parser.add_argument("--cmdc_features", type=str, default=None, help="Path to CMDC features. If None, trains only on EDAIC.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lam", type=float, default=0.5, help="Lambda weight for SupConLoss vs CrossEntropy")
    args = parser.parse_args()

    # Load English Data
    print(f"Loading EDAIC features from {args.edaic_features}...")
    train_edaic, test_edaic = load_dataset(args.edaic_features, domain_label=0)
    
    # Load Mandarin Data if provided
    if args.cmdc_features and os.path.exists(args.cmdc_features):
        print(f"Loading CMDC features from {args.cmdc_features}...")
        train_cmdc, test_cmdc = load_dataset(args.cmdc_features, domain_label=1)
        # Concat datasets
        full_train = torch.utils.data.ConcatDataset([train_edaic, train_cmdc])
        full_test = torch.utils.data.ConcatDataset([test_edaic, test_cmdc])
    else:
        print("No CMDC features provided or found. Training on EDAIC only.")
        full_train = train_edaic
        full_test = test_edaic

    train_loader = DataLoader(full_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(full_test, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ContrastiveAlignmentNet(input_dim=768, proj_dim=128, num_classes=2).to(device)
    
    criterion_ce = nn.CrossEntropyLoss()
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
    with open(os.path.join("output", "clead_edaic_results.txt"), "w") as f:
        f.write("CLeaD Contrastive Alignment Results\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC AUC:  {auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        
    # Save results to Excel-compatible CSV
    metrics_data = {
        "Metric": ["Accuracy", "F1 Score", "ROC AUC", "Precision (Healthy)", "Precision (Depressed)", "Recall (Healthy)", "Recall (Depressed)", "Support (Healthy)", "Support (Depressed)"],
        "Value": [acc, f1, auc, 0.95, 0.89, 0.93, 0.91, 3885, 2147] # Note: Hardcoded classification report metrics will be dynamic in actual runs based on sklearn report dict, but keeping simple for this update
    }
    
    # Generate dynamic report dictionary for CSV
    report_dict = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)
    csv_data = {
        "Metric": ["Accuracy", "F1 Score (Overall)", "ROC AUC", "Precision (Healthy)", "Precision (Depressed)", "Recall (Healthy)", "Recall (Depressed)", "Support (Healthy)", "Support (Depressed)"],
        "Value": [
            acc, 
            f1, 
            auc, 
            report_dict['0']['precision'], 
            report_dict['1']['precision'], 
            report_dict['0']['recall'], 
            report_dict['1']['recall'], 
            report_dict['0']['support'], 
            report_dict['1']['support']
        ]
    }
    df = pd.DataFrame(csv_data)
    df.to_csv(os.path.join("output", "clead_edaic_results.csv"), index=False)
        
    print("\nResults automatically saved to output/clead_edaic_results.txt and .csv")

if __name__ == "__main__":
    main()
