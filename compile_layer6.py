import os
import re

configs = [
    {"name": "EN -> EN", "baseline": "baseline_en_to_en_results.txt", "clead": "clead_en_to_en_results.txt"},
    {"name": "EN -> ZH", "baseline": "baseline_en_to_zh_results.txt", "clead": "clead_en_to_zh_results.txt"},
    {"name": "ZH -> EN", "baseline": "baseline_zh_to_en_results.txt", "clead": "clead_zh_to_en_results.txt"},
    {"name": "ZH -> ZH", "baseline": "baseline_zh_to_zh_results.txt", "clead": "clead_zh_to_zh_results.txt"},
    {"name": "MIX -> EN", "baseline": "baseline_mix_to_en_results.txt", "clead": "clead_mix_to_en_results.txt"},
    {"name": "MIX -> ZH", "baseline": "baseline_mix_to_zh_results.txt", "clead": "clead_mix_to_zh_results.txt"}
]

def parse_metrics(file_path):
    if not os.path.exists(file_path):
        # Fall back if path is slightly different (e.g. clead_edaic_results.txt instead of clead_en_to_en_results.txt)
        if "clead_en_to_en" in file_path and os.path.exists("output/clead_edaic_results.txt"):
            file_path = "output/clead_edaic_results.txt"
        else:
            return None, None, None
            
    acc, f1, auc = None, None, None
    with open(file_path, "r") as f:
        for line in f:
            if "Accuracy:" in line:
                acc = float(line.split(":")[-1].strip())
            elif "F1 Score:" in line:
                f1 = float(line.split(":")[-1].strip())
            elif "ROC AUC:" in line:
                auc = float(line.split(":")[-1].strip())
    return acc, f1, auc

# Header
table = []
table.append("=========================================================================================")
table.append("                             LAYER 6 SEGMENT-LEVEL RESULTS SUMMARY                      ")
table.append("=========================================================================================")
table.append(f"{'Configuration':<15} | {'Model':<10} | {'Accuracy':<10} | {'F1 Score':<10} | {'ROC AUC':<10}")
table.append("-----------------------------------------------------------------------------------------")

for cfg in configs:
    cfg_name = cfg["name"]
    
    # Baseline
    base_path = os.path.join("output", cfg["baseline"])
    b_acc, b_f1, b_auc = parse_metrics(base_path)
    b_acc_str = f"{b_acc:.4f}" if b_acc is not None else "N/A"
    b_f1_str = f"{b_f1:.4f}" if b_f1 is not None else "N/A"
    b_auc_str = f"{b_auc:.4f}" if b_auc is not None else "N/A"
    
    table.append(f"{cfg_name:<15} | {'Baseline':<10} | {b_acc_str:<10} | {b_f1_str:<10} | {b_auc_str:<10}")
    
    # CLeaD
    clead_path = os.path.join("output", cfg["clead"])
    c_acc, c_f1, c_auc = parse_metrics(clead_path)
    c_acc_str = f"{c_acc:.4f}" if c_acc is not None else "N/A"
    c_f1_str = f"{c_f1:.4f}" if c_f1 is not None else "N/A"
    c_auc_str = f"{c_auc:.4f}" if c_auc is not None else "N/A"
    
    table.append(f"{'':<15} | {'CLeaD':<10} | {c_acc_str:<10} | {c_f1_str:<10} | {c_auc_str:<10}")
    table.append("-----------------------------------------------------------------------------------------")

table.append("=========================================================================================")

table_str = "\n".join(table)
print(table_str)

# Write to file
with open("output/layer6_results_table.txt", "w") as f:
    f.write(table_str)
