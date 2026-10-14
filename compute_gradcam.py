#from gradcam_Inception.inception_time_multi import InceptionTimeModel
from Models_Archi.gradcam import MultiBranchGradCAM
from Models_Archi.utils import  process_rockfall_data_multi, charge, preprocess_segments_test
import pandas as pd 
import numpy as np 
import time
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, recall_score, precision_score,accuracy_score
from glob import glob
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import pickle
import os 
import cv2
from matplotlib.ticker import StrMethodFormatter
#for model with attention
from Models_Archi.one_module_inception_attention_temporal_multi_modality import InceptionTimeModel 
from matplotlib.lines import Line2D
import matplotlib.dates as mdates    
from matplotlib.patches import Patch

""" Code for computing Grad-CAM on all data using 90 models """

plt.rcParams.update({'axes.labelsize': 13,    # Taille du texte des labels des axes
                     'axes.titlesize': 12,   # Taille du texte du titre des axes
                     'xtick.labelsize': 14,  # Taille du texte des labels des ticks X
                     'ytick.labelsize': 14,  # Taille du texte des labels des ticks Y
                     'font.size': 13,        # Taille générale du texte
                     'legend.fontsize': 12,
                     'text.usetex': False})

def get_sample_role(sample_year, test_year, val_year, blind_years=[2024]):
    if sample_year in blind_years:
        return "blind"
    if sample_year == test_year:
        return "test"
    elif sample_year == val_year:
        return "val"
    else:
        return "train"

def entropy_from_probs(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p))

def to_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_serializable(v) for v in obj]
    elif hasattr(obj, "tolist"):  # numpy arrays
        return obj.tolist()
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    else:
        return str(obj)

# --------------------------------------------------------------MAIN---------------------------------------------------

resolution = "hour"   
normalisation = "stscaler"
suf="_1_36_72_144_kernels_Lateversion_average"

litterature_kernels = False
average_kernels = True

blind_test = True
folder = "blind_test" 
comb= "Temp_Charge_Freeze" 
suf_comb="" 

df_sismo_test= pd.read_csv('.data/events/ev_sismo2_2024.csv', delimiter=',')
df_RR1_test = pd.read_csv('.data/events/1.0_RR1_2024.csv', delimiter=',')
df_temp_test = pd.read_csv('.data/events/1.0_T_st_H_2024.csv', delimiter=',')
# transformer la colonne ' T' de str à float
df_temp_test[' T'] = pd.to_numeric(df_temp_test[' T'], errors='coerce').astype(np.float64)
df_precip_test = pd.read_csv('.data/events/data_meteo_Sabrine2_2024.csv', delimiter=',')
df_precip_test['AAAAMMJJHH'] = df_RR1_test['AAAAMMJJHH'].values
df_precip_test['AAAAMMJJHH'] = pd.to_datetime(df_precip_test['AAAAMMJJHH'])
lamb=0.2
df_precip_test['H'] = charge(df_precip_test,lamb)
df_gel_test= pd.read_csv('.data/events/data_Temp_depth_sabrine_st-H_2024.csv', delimiter=',')



# Prevision horizons 
hp =  1
compute = False # to compute Grad-CAM
show_gradcam = True
show_prediction_scenarios = False
compute_stats_prediction = False 

device = torch.device("cpu")
if resolution == "hour": 
        
    #Rain
    df_RR1_test = process_rockfall_data_multi(df_RR1_test, df_sismo_test, x_days=hp, col_name='RR1',  concat_with_R=True, jour_pluie=14)
    df_RR1_test['sample_id'] = df_RR1_test.index
    #Temperature 
    df_temp_test = process_rockfall_data_multi(df_temp_test, df_sismo_test, x_days=hp, col_name=' T', concat_with_R=False, jour_pluie=14)
    df_temp_test['sample_id'] = df_temp_test.index
    #Charge 
    df_charge_test = process_rockfall_data_multi(df_precip_test, df_sismo_test, x_days=hp, col_name='H', concat_with_R=False, jour_pluie=14)
    df_charge_test['sample_id'] = df_charge_test.index
    #Depth freeze 
    df_freeze_test = process_rockfall_data_multi(df_gel_test, df_sismo_test, x_days=hp, col_name='pgel',  concat_with_R=False, jour_pluie=14)
    df_freeze_test['sample_id'] = df_freeze_test.index

    df_test = pd.merge(df_RR1_test, df_freeze_test, on='sample_id')
    df_test = pd.merge(df_test, df_temp_test, on='sample_id')
    df_test = pd.merge(df_test, df_charge_test, on='sample_id')
    df_test = df_test.drop(columns=['sample_id'])
    df_test.set_index('AAAAMMJJHH', inplace=True)

    # print(df_test)

    # Recupure the columns of rain and temp
    X_rain_test = df_test[[f"RR1_hour_{i}" for i in range(336)]].values  #----> colonnes de précipitations
    X_T_test = df_test[[f" T_hour_{i}" for i in range(336)]].values  #----> colonnes de temperature
    X_charge_test = df_test[[f"H_hour_{i}" for i in range(336)]].values  #----> colonnes de charge/discharge
    X_depthgel_test = df_test[[f"pgel_hour_{i}" for i in range(336)]].values  #----> colonnes de  profondeur de gel

    originals = {
            "Rain": X_rain_test,
            "Temp": X_T_test,
            "Charge": X_charge_test,
            "Freeze": X_depthgel_test
        }
    
    # Normalisation 
    with open(f"Rockfall_benchmarking/combinations/scaler_rr1_{normalisation}.pkl", "rb") as f:
        scaler_rain = pickle.load(f)
    with open(f"Rockfall_benchmarking/combinations/scaler_temp_{normalisation}.pkl", "rb") as f:
        scaler_temp = pickle.load(f)
    with open(f"Rockfall_benchmarking/combinations/scaler_h_{normalisation}.pkl", "rb") as f:
        scaler_charge = pickle.load(f)
    with open(f"Rockfall_benchmarking/combinations/scaler_depth_gel_{normalisation}.pkl", "rb") as f:
        scaler_depthgel = pickle.load(f)

    x_rain_test_scaled = preprocess_segments_test(X_rain_test, scaler_rain)
    x_T_test_scaled = preprocess_segments_test(X_T_test, scaler_temp)
    x_charge_test_scaled = preprocess_segments_test(X_charge_test, scaler_charge)
    x_depthgel_test_scaled = preprocess_segments_test(X_depthgel_test, scaler_depthgel)

    y_test = df_test['rockfall_target'].values
    encoder = LabelEncoder()
    y_encoded_test = encoder.fit_transform(y_test)

    #conversion to PyTorch tensors
    X_tensor_rain_blind_test = torch.FloatTensor(x_rain_test_scaled).view(-1, 1, x_rain_test_scaled.shape[1]).to(device)
    X_tensor_T_blind_test = torch.FloatTensor(x_T_test_scaled).view(-1, 1, x_T_test_scaled.shape[1]).to(device)
    X_tensor_H_blind_test = torch.FloatTensor(x_charge_test_scaled).view(-1, 1, x_charge_test_scaled.shape[1]).to(device)
    X_tensor_depthgel_blind_test = torch.FloatTensor(x_depthgel_test_scaled).view(-1, 1, x_depthgel_test_scaled.shape[1]).to(device)
    y_tensor_blind_test = torch.LongTensor(y_encoded_test).to(device)
    
    # Original signal's tensor
    X_test_tensor_orig_rain = torch.FloatTensor(X_rain_test).view(-1, 1, X_rain_test.shape[1]).to(device)
    X_test_tensor_orig_T = torch.FloatTensor(X_T_test).view(-1, 1, X_T_test.shape[1]).to(device)
    X_test_tensor_orig_H = torch.FloatTensor(X_charge_test).view(-1, 1, X_charge_test.shape[1]).to(device)
    X_test_tensor_orig_depthgel = torch.FloatTensor(X_depthgel_test).view(-1, 1, X_depthgel_test.shape[1]).to(device)


if compute == True :

    # Time of execution 
    start_time = time.time()
    results = {}
    model_id = 0

    from collections import defaultdict
    from glob import glob

    if litterature_kernels: 
        kernels_modalites = {'Rain': [9, 19, 39], 'Charge': [9, 19, 39],
                             'Temp': [9, 19, 39], 'Freeze':[9, 19, 39]}

    if average_kernels:
        kernels_modalites = {'Rain': [17, 55, 117], 'Charge': [17, 55, 117],
                             'Temp': [19, 49, 111], 'Freeze':[19, 49, 111]}


    feature_dict = {
                    "Rain": X_tensor_rain_blind_test,
                    "Temp": X_tensor_T_blind_test,
                    "Charge": X_tensor_H_blind_test,
                    "Freeze": X_tensor_depthgel_blind_test
                    }
    

    names_subset  = comb.split("_")
    in_channels_list = [1] * len(names_subset)
    X_BLIND_test = [feature_dict[name] for name in names_subset]
    print(names_subset)

    results = {}
    model_global_idx = 0

    for test_year in range(2014, 2024):
        print(f"year {test_year} processing calcul")

        if average_kernels or litterature_kernels:
            kernel_sizes_list = [kernels_modalites[name] for name in names_subset]

        model_paths = glob(
            f"Rockfall_benchmarking/combinations/HP_{hp}_{normalisation}{suf}/{comb}{suf_comb}"
            f"/test_year_{test_year}/val_*.pt")

        # print(model_paths)

        for model_path in model_paths:


            val_year = int(model_path[-7:-3])  # extracting the val year from the path
            model_id = f"M{model_global_idx}"
            model_global_idx += 1
            print(f"Processing {model_id} | test={test_year}, val={val_year}")

            #load model 
            model = InceptionTimeModel( in_channels_list=in_channels_list, num_classes=len(np.unique(y_encoded_test)),
                    n_blocks=1, kernel_sizes=kernel_sizes_list , merge_mode='concat', use_attention=False)
            
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and ('model_state_dict' in checkpoint or 'state_dict' in checkpoint):
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict'))
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
            model.eval()

            results[model_id] = {}

            with torch.no_grad():
                outputs = model(*X_BLIND_test)
                probs = torch.softmax(outputs, dim=1)
                y_pred = probs.argmax(dim=1).cpu().numpy()


            #sample's loop
            for idx in range(len(y_test)):

                p = probs[idx].cpu().numpy()     # shape (2,)
                entropy = entropy_from_probs(p)
                p = np.round(p.astype(np.float64), 2).tolist()
                sample_tensors = [feat[idx].unsqueeze(0).to(device) for feat in X_BLIND_test]
                gradcams = {}

                for branch_index, name in enumerate(names_subset):
                    gradcam = MultiBranchGradCAM( model, target_branch=model.branches[branch_index])
                    gradcams[name] = {}

                    for cls in [0, 1]:
                        heatmap = gradcam(*sample_tensors, target_class=torch.tensor([cls]).to(device))
                        gradcams[name][cls] = heatmap.squeeze()


                results[model_id][idx] = {
                    "gradcam": gradcams,
                    "pred": int(y_pred[idx]),
                    "truth": int(y_test[idx]),
                    "probs": p,
                    "entropy": round(float(entropy), 2),
                }


            os.makedirs(f"Rockfall_benchmarking/combinations/GradCAM/{comb}{suf_comb}/", exist_ok=True)

            json_path = (
                f"Rockfall_benchmarking/combinations/GradCAM/{comb}{suf_comb}/"
                f"{model_id}_test{test_year}_val{val_year}.json")


            with open(json_path, "w") as f:
                json.dump(to_json_serializable(results[model_id]), f, indent=2)

            del results[model_id]
            torch.cuda.empty_cache()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time = {elapsed_time} seconds")



def reduce_gradcam_dims(arr):
    a = np.array(arr, dtype=float)
    # if multi-channel, average channels -> produce 1D time axis
    if a.ndim == 2:
        # shape (channels, T) or (T, channels) -> try to detect
        if a.shape[0] <= a.shape[1] and a.shape[0] <= 16:  # likely channels first
            a = a.mean(axis=0)
        else:
            a = a.mean(axis=1)
    elif a.ndim > 2:
        # average all but last axis if uncertain
        a = a.reshape(a.shape[0], -1).mean(axis=0)
    return a

def load_all_gradcam(json_dir):
    files = sorted(glob(os.path.join(json_dir, "*.json")))
    entries = []  # list of tuples (model_file, sample_idx (int), pred, truth, gradcam_dict)
    for fn in files:
        with open(fn, "r") as f:
            data = json.load(f)
        # data keys are sample indices 
        for k, v in data.items():
            try:
                idx = int(k)
            except:
                idx = int(str(k))
            pred = int(v.get("pred"))
            truth = int(v.get("truth"))
            gradcam = v.get("gradcam", {})
            entries.append((os.path.basename(fn), idx, pred, truth, gradcam))
    return entries


def compute_stats(list_of_arrays):
    if len(list_of_arrays) == 0:
        return None, None, None, None
    stacked = np.vstack([np.array(x) for x in list_of_arrays])  # shape (N, T)
    mean = stacked.mean(axis=0)
    median = np.median(stacked, axis=0)
    p10 = np.percentile(stacked, 10, axis=0)
    p90 = np.percentile(stacked, 90, axis=0)
    return mean, median, p10, p90


C_RAIN    = "#378ADD"
C_EVENT   = "#D85A30"
C_NEUTRAL = "#888780"
C_TRAIN   = "#378ADD"
C_VAL     = "#D85A30"
C_GRAY    = "#B4B2A9"
C_TP      = "#1D9E75"
C_FP      = "#BA7517"
C_TN      = "#7F77DD"
C_FN      = "#D85A30"




SCENARIO_LABELS = {
    "1/1": ("TP", "Rockfall correctly detected",  C_TP),
    "0/0": ("TN", "Absence correctly rejected",  C_TN),
    "1/0": ("FP", "False alarm: no rockfall",  C_FP),
    "0/1": ("FN", "Rockfall missed",  C_FN),
}

def aggregate_by_scenario_majority_vote(entries, modalities, vote_threshold=45):
    """
    Aggregates Grad-CAMs based on majority voting.

    For each sample:
      - Count the number of models predicting class 1.
      - Set vote_final = 1 if the number of class 1 votes exceeds vote_threshold, otherwise 0.
      - Assign the scenario based on (vote_final / truth).
      - Keep only the Grad-CAMs from models that voted for the final majority class:
          * If vote_final = 1 → keep class 1 Grad-CAMs from models that predicted 1.
          * If vote_final = 0 → keep class 0 Grad-CAMs from models that predicted 0.

    Parameters:
    entries        : list of tuples (model_file, idx, pred, truth, gradcam_dict)
    modalities     : list of modalities, e.g., ["Temp", "Charge", "Freeze"]
    vote_threshold : threshold for majority voting (> threshold -> class 1), default 45
    """
    # Group all predictions by sample
    # sample_data[idx] = {"truth": int, "votes": [(pred, gradcam_dict), ...]}
    sample_data = {}
    for (_, idx, pred, truth, gradcam_dict) in entries:
        if idx not in sample_data:
            sample_data[idx] = {"truth": truth, "votes": []}
        sample_data[idx]["votes"].append((pred, gradcam_dict))

    # Apply majority voting and select the corresponding Grad-CAMs
    scenarios    = {"1/1": [], "0/0": [], "1/0": [], "0/1": []}
    collected    = {s: {m: [] for m in modalities} for s in scenarios}
    sample_indices = {s: set() for s in scenarios}

    for idx, info in sample_data.items():
        truth  = info["truth"]
        votes  = info["votes"]   # List of (pred, gradcam_dict) pairs across all models

        nb_votes_1 = sum(1 for (p, _) in votes if p == 1)
        vote_final = 1 if nb_votes_1 > vote_threshold else 0

        key = f"{vote_final}/{truth}"
        if key not in scenarios:
            continue

        # Select models that voted for the final majority class
        winning_gradcams = [gd for (p, gd) in votes if p == vote_final]

        for mod in modalities:
            for gradcam_dict in winning_gradcams:
                mod_dict = gradcam_dict.get(mod)
                if mod_dict is None:
                    continue
                cls_key = str(vote_final)
                if cls_key in mod_dict:
                    arr = reduce_gradcam_dims(mod_dict[cls_key])
                else:
                    try:
                        arr = reduce_gradcam_dims(mod_dict.get(int(cls_key)))
                    except:
                        continue
                collected[key][mod].append(arr)

        sample_indices[key].add(idx)

    return collected, sample_indices



def plot_modality_all_scenarios_latex(modality, all_grad_stats, all_orig_stats, out_dir, window_h, counts=None):

    scenarios_order = ["1/1", "1/0", "0/0", "0/1"]
    row_labels = ["(a)", "(b)", "(c)", "(d)"]
    n_days = window_h // 24

    fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex=False)

    for row, scenario in enumerate(scenarios_order):

        key, label, color = SCENARIO_LABELS[scenario]

        mean_g, median_g, p10_g, p90_g = all_grad_stats[scenario]
        mean_o, median_o, p10_o, p90_o = all_orig_stats[scenario]

        def safe_flip(arr):
            return np.flip(arr) if arr is not None else None

        mean_o, median_o, p10_o, p90_o = map(safe_flip, (mean_o, median_o, p10_o, p90_o))
        mean_g, median_g, p10_g, p90_g = map(safe_flip, (mean_g, median_g, p10_g, p90_g))

        if mean_o is not None:
            T = len(mean_o)
        elif mean_g is not None:
            T = len(mean_g)
        else:
            T = window_h

        x = np.linspace(-T, 0, T)

        # ================= RAW SIGNAL =================
        ax = axes[row, 0]
        if mean_o is not None:
            ax.fill_between(x, p10_o, p90_o, color="C0", alpha=0.25)
            ax.plot(x, mean_o, color="C0", linewidth=1.5)
            ax.plot(x, median_o, color="C0", linestyle="--", linewidth=1.2)

        ax.text(-0.18, 0.5, row_labels[row], transform=ax.transAxes, fontsize=14, va="center", ha="center")
        ax.set_xticks(np.linspace(-T, 0, 7))
        ax.set_xticklabels([f"{int(v)}" for v in np.linspace(-T, 0, 7)])


        # ================= GRAD-CAM =================
        ax = axes[row, 1]
        if mean_g is not None:
            ax.fill_between(x, p10_g, p90_g, color=color, alpha=0.15)
            ax.plot(x, mean_g, color=color, linewidth=1.5)
            ax.plot(x, median_g, color=color, linestyle="--", linewidth=1.2)
            for d in range(-n_days, 0):
                ax.axvline(d * 24, color="gray", linestyle=":", linewidth=0.4, alpha=0.3)
        ax.set_xticks(np.linspace(-T, 0, 7))
        ax.set_xticklabels([f"{int(v)}" for v in np.linspace(-T, 0, 7)])


    # ================= TITLES =================
    axes[0, 0].set_title(f"{modality} signal", fontsize=13, pad=45)
    axes[0, 1].set_title("Grad-CAM", fontsize=13, pad=45)

    # ================= LEGENDS =================
    raw_handles = [
        Line2D([0], [0], color="C0", lw=1.8, label="Mean"),
        Line2D([0], [0], color="C0", lw=1.8, linestyle="--", label="Median"),
        Patch(facecolor="C0", alpha=0.25, edgecolor="none", label="10–90%")
    ]
    scenario_handles = [
    Line2D([0], [0], color=SCENARIO_LABELS["1/1"][2], lw=3, label="TP"),
    Line2D([0], [0], color=SCENARIO_LABELS["1/0"][2], lw=3, label="FP"),
    Line2D([0], [0], color=SCENARIO_LABELS["0/0"][2], lw=3, label="TN"),
    Line2D([0], [0], color=SCENARIO_LABELS["0/1"][2], lw=3, label="FN")
]

    grad_handles = [
        Line2D([0], [0], color="gray", lw=1.8, label="Mean"),
        Line2D([0], [0], color="gray", lw=1.8, linestyle="--", label="Median"),
        Patch(facecolor="gray", alpha=0.2, edgecolor="none", label="10–90%")
    ]
    grad_handles_all = grad_handles + scenario_handles

    axes[0, 0].legend(handles=raw_handles, fontsize=11, loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=3, frameon=False)
    axes[0, 1].legend(handles=grad_handles_all, fontsize=11, loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=4, frameon=False)


    # ================= AXIS LABELS =================
    axes[-1, 0].set_xlabel("Time before prediction (hours)")
    axes[-1, 1].set_xlabel("Time before prediction (hours)")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"gradcam_all_scenarios_{modality}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_majority_vote_predictions_timeline(entries, comb, vote_threshold=45, out_dir=None):

    sample_data = {}

    for (_, idx, pred, truth, _) in entries:
        if idx not in sample_data:
            sample_data[idx] = {
                "truth": truth,
                "preds": []
            }

        sample_data[idx]["preds"].append(pred)


    results = []

    for idx, info in sample_data.items():
        truth = info["truth"]
        nb_votes_1 = sum(info["preds"])
        vote_final = 1 if nb_votes_1 > vote_threshold else 0
        results.append( (idx, f"{vote_final}/{truth}") )

    results.sort(key=lambda x: x[0])


    SCENARIO_ORDER = ["1/1", "0/1", "1/0", "0/0"]

    STYLE = {
        "1/1": {"color": C_TP, "label": "TP"},
        "0/1": {"color": C_FN, "label": "FN"},
        "1/0": {"color": C_FP, "label": "FP"},
        "0/0": {"color": C_TN, "label": "TN"},
    }


    Y_POS = {s: i for i, s in enumerate(reversed(SCENARIO_ORDER))}

    dates = df_test.index

    from collections import defaultdict
    grouped = defaultdict(list)

    for idx, key in results:

        if idx < len(dates):
            grouped[key].append(dates[idx])



    fig, ax = plt.subplots(figsize=(17, 4))


    # Scatter predictions
    for key, date_list in grouped.items():

        jitter = np.random.uniform( -0.15, 0.15,len(date_list))

        ax.scatter(
            date_list,
            np.full(len(date_list), Y_POS[key]) + jitter,
            color=STYLE[key]["color"],
            s=55,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            zorder=3
        )


    # Monthly background
    import calendar

    for month in range(1, 13):

        start = pd.Timestamp(year=2024,month=month,day=1)

        end = pd.Timestamp(year=2024,month=month,day=calendar.monthrange(2024, month)[1], hour=23)

        if month % 2 == 0:

            ax.axvspan(start,end,color="gray",alpha=0.07,zorder=0)


    # Scenario separation lines
    for y in [0.5, 1.5, 2.5]:

        ax.axhline(y, color="gray", linestyle="--", linewidth=0.6, alpha=0.4)


    # Y axis
    ax.set_yticks([0, 1, 2, 3])

    ax.set_yticklabels(["TN", "FP", "FN", "TP"],fontsize=13)


    for tick, key in zip(ax.get_yticklabels(),reversed(SCENARIO_ORDER)):

        tick.set_color(STYLE[key]["color"])

        tick.set_fontweight("bold")


    ax.set_ylim(-0.5, 3.5)


    # X axis
    ax.set_xlim(pd.Timestamp("2024-01-01"),pd.Timestamp("2024-12-31"))

    ax.xaxis.set_major_locator(mdates.MonthLocator())

    ax.xaxis.set_major_formatter( mdates.DateFormatter("%b"))

    fig.subplots_adjust(left=0.05)


    if out_dir:

        os.makedirs(out_dir, exist_ok=True)

        fig.savefig(
            os.path.join(
                out_dir,
                f"prediction_timeline_{comb}.png"
            ),
            dpi=300,
            bbox_inches='tight'
        )
    plt.close(fig)


def plot_ground_truth_timeline(out_dir=None):
    dates = df_test.index
    rockfall_events = np.array(df_test["rockfall_target"])

    fig, ax = plt.subplots(figsize=(17, 1.2))

    import calendar

    # Same background
    for month in range(1, 13):
        start = pd.Timestamp(year=2024, month=month, day=1)
        end = pd.Timestamp(year=2024, month=month, day=calendar.monthrange(2024, month)[1], hour=23)
        if month % 2 == 0:
            ax.axvspan(start, end, color="gray", alpha=0.07)

    # Rockfall events
    for date, val in zip(dates, rockfall_events):
        if int(val) == 1:
            ax.axvline(date, color="darkred", linewidth=1.0, alpha=0.7)

    ax.set_ylim(0, 1)

    # Reserve the same space as prediction y-axis labels
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["TN", "FP", "FN", "TP"], fontsize=13)

    for tick in ax.get_yticklabels():
        tick.set_alpha(0)

    ax.tick_params(axis="y", length=0)

    ax.set_xlim(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    ax.set_xlabel("Month (2024)", fontsize=14)

    # Same margins as the prediction figure
    fig.subplots_adjust(left=0.16)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, "ground_truth_timeline.png"), dpi=300, bbox_inches='tight')

    plt.close(fig)

def seasonal_tp_statistics( entries, df_test,  vote_threshold=45, winter_months=(11, 12, 1, 2), summer_months=(6, 7, 8),):
    sample_data = {}

    for (_, idx, pred, truth, _) in entries:

        if idx not in sample_data:
            sample_data[idx] = {
                "truth": truth,
                "preds": []
            }

        sample_data[idx]["preds"].append(pred)


    total_tp = 0
    total_rockfall = 0

    winter_tp = 0
    summer_tp = 0

    winter_rockfall = 0
    summer_rockfall = 0


    for idx, info in sample_data.items():

        truth = info["truth"]

        vote_final = int(sum(info["preds"]) > vote_threshold)

        date = df_test.index[idx]
        month = date.month

        is_winter = month in winter_months
        is_summer = month in summer_months

        if truth == 1:

            total_rockfall += 1

            if is_winter:
                winter_rockfall += 1

            if is_summer:
                summer_rockfall += 1

        if vote_final == 1 and truth == 1:

            total_tp += 1

            if is_winter:
                winter_tp += 1

            if is_summer:
                summer_tp += 1


    stats = {

        "winter TP": winter_tp,
        "summer TP": summer_tp,
        "total TP": total_tp,

        "winter rockfall": winter_rockfall,
        "summer rockfall": summer_rockfall,
        "total rockfall": total_rockfall,

        "winter detection (%)":
            100 * winter_tp / winter_rockfall if winter_rockfall else 0,

        "summer detection (%)":
            100 * summer_tp / summer_rockfall if summer_rockfall else 0,

        "overall detection (%)":
            100 * total_tp / total_rockfall if total_rockfall else 0
    }

    return stats


if show_prediction_scenarios:
    gradcam_dir = f"Rockfall_benchmarking/combinations/GradCAM/{comb}{suf_comb}/"
    entries = load_all_gradcam(gradcam_dir)
    modalities = comb.split("_")
    # visualisation temporelle des prédictions 
    plot_majority_vote_predictions_timeline( entries, comb, vote_threshold=45, out_dir=os.path.join(os.path.dirname(gradcam_dir), "aggregated_plots"))
    # plot_ground_truth_timeline(out_dir=os.path.join(os.path.dirname(gradcam_dir), "aggregated_plots"))

if compute_stats_prediction :
    gradcam_dir = f"Rockfall_benchmarking/combinations/GradCAM/{comb}{suf_comb}/"
    entries = load_all_gradcam(gradcam_dir)
    stats = seasonal_tp_statistics(entries, df_test, vote_threshold=45)

    for k, v in stats.items():
        print(f"{k:25s}: {v}")

if show_gradcam:
    gradcam_dir = f"Rockfall_benchmarking/combinations/GradCAM/{comb}{suf_comb}/"
    print("Loading gradcam jsons from:", gradcam_dir)
    entries = load_all_gradcam(gradcam_dir)
    modalities = comb.split("_")

    collected, sample_indices = aggregate_by_scenario_majority_vote(entries, modalities, vote_threshold=45)
    
    out_dir = os.path.join(os.path.dirname(gradcam_dir), "aggregated_plots")
    os.makedirs(out_dir, exist_ok=True)

    for mod in modalities:
        all_grad_stats = {}
        all_orig_stats = {}

        for scenario in ["1/1", "1/0", "0/0", "0/1"]:
            grad_list = collected[scenario][mod]
            all_grad_stats[scenario] = compute_stats(grad_list)
            idxs = sorted(list(sample_indices[scenario]))
            if len(idxs) == 0:
                all_orig_stats[scenario] = (None, None, None, None)
            else:
                orig_arr = originals.get(mod)
                if orig_arr is None:
                    all_orig_stats[scenario] = (None, None, None, None)
                else:
                    selected = orig_arr[idxs]
                    all_orig_stats[scenario] = (
                        selected.mean(axis=0),
                        np.median(selected, axis=0),
                        np.percentile(selected, 10, axis=0),
                        np.percentile(selected, 90, axis=0),
                    )

        counts = {s: len(sample_indices[s]) for s in ["1/1", "1/0", "0/0", "0/1"]}
        plot_modality_all_scenarios_latex(
            mod, all_grad_stats, all_orig_stats, out_dir,
            window_h=len(next(
                arr for lst in collected.values()
                for arr in lst.get(mod, [])
            )),
            counts=counts
        )






