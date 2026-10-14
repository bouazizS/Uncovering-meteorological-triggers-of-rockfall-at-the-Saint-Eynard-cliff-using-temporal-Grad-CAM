import os
import json
import pandas as pd


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())



base_path = "/Rockfall_benchmarking/Rain_Temp_all_models"

models = [ "CNN", "RF", "SVM","NARMAX","TimeTransformer_32", "TimeTransformer_64", "InceptionTime_6" , "InceptionTime_1"]
hp=1
norm = "stscaler"

results = []

for model in models:
    model_path = os.path.join(base_path, model+f"/HP_{hp}_{norm}")
    
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
        "Model": model,
        "Balanced Accuracy": round(res["balanced accuracy"], 3),
        "F1-score" : round(res["f1_score"], 3),
        # Metrics by class
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
        "Execution_Time": round(metrics["execution_time"], 3)
    }
    
    results.append(row)

df = pd.DataFrame(results)
#save
df.to_csv(f"{base_path}/benchmark_models.csv", index=False)
