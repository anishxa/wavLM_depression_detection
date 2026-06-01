import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

def load_train_val_numpy(feature_dir):
    """Loads X_train, y_train, X_val, y_val from feature_dir and concatenates them."""
    X_train = np.load(os.path.join(feature_dir, "X_train.npy"))
    y_train = np.load(os.path.join(feature_dir, "y_train.npy"))
    X_val = np.load(os.path.join(feature_dir, "X_val.npy"))
    y_val = np.load(os.path.join(feature_dir, "y_val.npy"))
    
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    return X_all, y_all

def load_test_numpy(feature_dir):
    """Loads X_test, y_test from feature_dir."""
    try:
        X_test = np.load(os.path.join(feature_dir, "X_test.npy"))
        y_test = np.load(os.path.join(feature_dir, "y_test.npy"))
    except FileNotFoundError:
        print(f"Test set not found in {feature_dir}. Falling back to validation set.")
        X_test = np.load(os.path.join(feature_dir, "X_val.npy"))
        y_test = np.load(os.path.join(feature_dir, "y_val.npy"))
    return X_test, y_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Directory containing train/val features.")
    parser.add_argument("--test_data", type=str, required=True, help="Directory containing test features.")
    parser.add_argument("--exp_name", type=str, default="baseline_cross_lingual", help="Prefix for output result files.")
    args = parser.parse_args()

    # Load Train Data
    print(f"Loading training features from {args.train_data}...")
    X_train, y_train = load_train_val_numpy(args.train_data)
    
    # Load Test Data
    print(f"Loading testing features from {args.test_data}...")
    X_test, y_test = load_test_numpy(args.test_data)
    
    # Train Logistic Regression
    print("\n--- Training Baseline Logistic Regression ---")
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Predict
    print("\n--- Evaluation ---")
    all_preds = clf.predict(X_test)
    all_probs = clf.predict_proba(X_test)[:, 1]
    all_labels = y_test
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
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
    
    report = classification_report(all_labels, all_preds, zero_division=0)
    print("\nClassification Report:")
    print(report)
    
    # Save results to file
    os.makedirs("output", exist_ok=True)
    txt_path = os.path.join("output", f"{args.exp_name}_results.txt")
    with open(txt_path, "w") as f:
        f.write("Baseline Logistic Regression Results\n")
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
