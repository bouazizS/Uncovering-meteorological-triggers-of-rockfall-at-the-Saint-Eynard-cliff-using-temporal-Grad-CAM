import os
import json
import pandas as pd
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

hp=1
norm = "stscaler"
resolution = "hourly" 

if resolution == "hourly":
    suf = "_litt_kernels"
    resol=""

base_path = f"Rockfall_benchmarking/combinations/HP_{hp}_{norm}{suf}"

combs = [
        'Rain',
        'Temp',
        'Charge',
        'Freeze',
        'Rain_Temp',
        'Rain_Charge',
        'Rain_Freeze',
        'Temp_Charge',
        'Temp_Freeze',
        'Charge_Freeze',
        'Rain_Temp_Charge',
        'Rain_Temp_Freeze',
        'Rain_Charge_Freeze',
        'Temp_Charge_Freeze',
        'Rain_Temp_Charge_Freeze'
        ]



results = []



for comb in combs:
    model_path = os.path.join(base_path, comb)
    
    # load file of metrics
    with open(os.path.join(model_path, "results_vote.json"), "r") as f:
        res = json.load(f)[0]  # dict in a list

    # load file of execution time
    with open(os.path.join(model_path, "metrics_hp_1_all_test_years_averaged.json"), "r") as f:
        metrics = json.load(f)
    
    cm = res["confusion_matrix"]
    tn, fp = cm[0]
    fn, tp = cm[1]

    row = {
        f"Input{resol}": comb ,
        "Balanced Accuracy": round(res["balanced accuracy"], 3),
        "F1-score": round(res["f1_score"], 3),
        # Métriques par classe 
        "Precision_class_0": round(res["precision_class_0"], 3),
        "Recall_class_0": round(res["recall_class_0"], 3),
        "F1_class_0": round(res["f1_class_0"], 3),

        "Precision_class_1": round(res["precision_class_1"], 3),
        "Recall_class_1": round(res["recall_class_1"], 3),
        "F1_class_1": round(res["f1_class_1"], 3),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Execution_Time": round(metrics["execution_time_seconds"], 3)
    }
    
    results.append(row)

df = pd.DataFrame(results)
#save
df.to_csv(f"{base_path}/Performance_combinations_2024.csv", index=False)
