import numpy as np
import pandas as pd
import time 
import os
import json
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
#CNN
from Models_Archi.CNN import CNNLateFusion 
# import of inceptionTime with 6 modules 
from Models_Archi.InceptionTime_6_attention import InceptionTimeModel_6 
# import of inceptionTime with 1 module
from Models_Archi.one_module_inception import InceptionTimeModel_1
#NARMAX
import statsmodels.api as sm
from Models_Archi.NARMAX import NARXModel, NARXDataset, build_y_lags
# Time transformer 
from Models_Archi.Time_transformer import BRANCH_A, BRANCH_B, DualBranchTimeTransformer

from imblearn.ensemble import BalancedRandomForestClassifier
from Models_Archi.utils import  process_rockfall_data_multi, preprocess_segments, preprocess_segments_test
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch.utils.data as data
import random 
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import f1_score, recall_score, precision_score , balanced_accuracy_score, accuracy_score
import pickle
import seaborn as sns
import torch.nn.init as init
import joblib  # to save sklearn models
from glob import glob

SEED_ = 42

def get_seed(outer_fold: int, inner_fold: int):
    #Seed unique pour chaque modele dans le fold
    return SEED_ + outer_fold * 100 + inner_fold

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed) # For GPU
    torch.backends.cudnn.deterministic = True # For GPU
    torch.backends.cudnn.benchmark = False # For GPU

def init_kaiming(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(v) for v in obj]
    else:
        return obj

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DualInputDataset:
    def __init__(self, x1, x2, y):
        self.x1 = x1  #1st time series tensor [n_samples, 1, 336]
        self.x2 = x2  #2nd time series tensor [n_samples, 1, 336]
        self.y = y    #Labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.x1[idx], self.x2[idx]), self.y[idx] #Tuples


use_attention=False

# Models 
models = ["TimeTransformer", "NARMAX", "CNN", "InceptionTime_6", "InceptionTime_1","RF","SVM" ]  
# z-score normalisation
normalisation = "stscaler"

# Forecasting horizon size 
hp = 1

# Training requirements :
# Load dataframes of rain, temperature and sismo events
df_sismo= pd.read_csv('.data/events/ev_sismo2.csv')
df_RR1= pd.read_csv('.data/events/1.0_RR1.csv', delimiter='\t')
df_temp = pd.read_csv('.data/events/1.0_ T_St_Hiliaire_filled_complete.csv', delimiter='\t')


# Associate rain and temperature data with rockfalls events 
df_1 = process_rockfall_data_multi(df_RR1, df_sismo, x_days=hp, col_name='RR1',  concat_with_R=True, jour_pluie=14)
df_1['sample_id'] = df_1.index
#Temperature 
df_2 = process_rockfall_data_multi(df_temp, df_sismo, x_days=hp, col_name=' T', concat_with_R=False, jour_pluie=14)
df_2['sample_id'] = df_2.index
# Merge dataframes
df = pd.merge(df_2, df_1, on='sample_id')
df = df.drop(columns=['sample_id'])
# Set index of df AAAAMMJJHH (to use it for year column)
df.set_index('AAAAMMJJHH', inplace=True)
# Generate a column of years of rockfall as groups
df['year'] = pd.to_datetime(df.index).year 
df = df[(df['year'] != 2013) & (df['year'] != 2024)]  # remove 2013 (not a complete year) and 2024 (blind test)

X_1 = df[[f"RR1_hour_{i}" for i in range(336)]].values  # Precipitations column
X_2 = df[[f" T_hour_{i}" for i in range(336)]].values  # Temperature column

# Normalisation for preprocessing
x1_scaled, scaler1 = preprocess_segments(X_1, normalisation) 
x2_scaled, scaler2 = preprocess_segments(X_2, normalisation)    #x_scaled de shape (n_samp, 336, 1) 
# Save scalers
with open(f"Rockfall_benchmarking/scaler_rr1_{normalisation}.pkl", "wb") as f:
    pickle.dump(scaler1, f)
with open(f"Rockfall_benchmarking/scaler_T_{normalisation}.pkl", "wb") as f:
    pickle.dump(scaler2, f)

# Target variable
y = df['rockfall_target'].values
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Conversion to pytorch tensors (useful for CNN, Inception models and ARMAX)
x1_tensor = torch.FloatTensor(x1_scaled).permute(0, 2, 1)   # de shape (n_samp, 1,336)
x2_tensor = torch.FloatTensor(x2_scaled).permute(0, 2, 1)   # de shape (n_samp, 1,336)
y_tensor = torch.LongTensor(y_encoded)

groups = df['year'].values  # groups by years
unique_years = np.unique(groups)
gkf = GroupKFold(n_splits=len(unique_years))  # number of groups = unique years

# Set device 
device = torch.device("cpu")
# Class weights 
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weight_tensor = torch.FloatTensor(class_weights).to(device)



# Blind test requirements :
vote = 'HARD' # majority vote 
#Load data for blind test
df_sismo_test= pd.read_csv('.data/events/ev_sismo2_2024.csv', delimiter=',')
df_RR1_test = pd.read_csv('.data/events/1.0_RR1_2024.csv', delimiter=',')
df_temp_test = pd.read_csv('.data/events/1.0_T_st_H_2024.csv', delimiter=',')
# transformer la colonne ' T' de str à float
df_temp_test[' T'] = pd.to_numeric(df_temp_test[' T'], errors='coerce').astype(np.float64)

#Rain
df_RR1_test = process_rockfall_data_multi(df_RR1_test, df_sismo_test, x_days=hp, col_name='RR1',  concat_with_R=True, jour_pluie=14)
df_RR1_test['sample_id'] = df_RR1_test.index
#Temperature 
df_temp_test = process_rockfall_data_multi(df_temp_test, df_sismo_test, x_days=hp, col_name=' T', concat_with_R=False, jour_pluie=14)
df_temp_test['sample_id'] = df_temp_test.index

df_test = pd.merge(df_temp_test, df_RR1_test, on='sample_id')
df_test = df_test.drop(columns=['sample_id'])

df_test.set_index('AAAAMMJJHH', inplace=True)
# Recupure the columns of rain and temp
X_rain = df_test[[f"RR1_hour_{i}" for i in range(336)]].values  #----> colonnes de précipitations
X_T = df_test[[f" T_hour_{i}" for i in range(336)]].values  #----> colonnes de temperature

# Normalisation 
with open(f"Rockfall_benchmarking/scaler_rr1_{normalisation}.pkl", "rb") as f:
    scaler_rain = pickle.load(f)

with open(f"Rockfall_benchmarking/scaler_T_{normalisation}.pkl", "rb") as f:
    scaler_temp = pickle.load(f)

x_T_scaled = preprocess_segments_test(X_T, scaler_temp)
x_rain_scaled = preprocess_segments_test(X_rain, scaler_rain)

y_test = df_test['rockfall_target'].values
encoder = LabelEncoder()
y_encoded_test = encoder.fit_transform(y_test)

#conversion en tenseurs PyTorch
X_tensor_rain_blind_test = torch.FloatTensor(x_rain_scaled).view(-1, 1, x_rain_scaled.shape[1]).to(device)
X_tensor_T_blind_test = torch.FloatTensor(x_T_scaled).view(-1, 1, x_T_scaled.shape[1]).to(device)
y_tensor_blind_test = torch.LongTensor(y_encoded_test).to(device)




for model_str in models : 
    start_time = time.time()

    print(f"\nrunning model {model_str} with hp = {hp}")

    path= f'Rockfall_benchmarking/Rain_Temp_all_models/{model_str}/HP_{hp}_{normalisation}/'
    os.makedirs(path, exist_ok=True)

    # Initialize to save the performance
    fold_results = []
    all_test_metrics = {}
    all_val_metrics = {}
    all_train_metrics = {}
    all_test_results = []  # pour stocker les resultats de toutes les tests pour ce hp

    # groupkfold
    if model_str in ["RF", "SVM"]: 
        train = True
        if train == True:
            #remove the last dimension 
            X_1_flat = x1_scaled.reshape(x1_scaled.shape[0], -1)  # --> (n_samples, 336)
            X_2_flat = x2_scaled.reshape(x2_scaled.shape[0], -1)  # --> (n_samples, 336)
            X_train = np.concatenate([X_1_flat, X_2_flat], axis=1)  # (n_samples, nbr_features=336x2)


            y_train = df['rockfall_target'].values
            encoder = LabelEncoder()
            y_train_encoded = encoder.fit_transform(y_train)
            
            for fold, (train_val_idx, test_idx) in enumerate(gkf.split(x1_tensor, y_train_encoded, groups=groups)):
                print(f"\n--- Fold {fold+1} ---")

                X_train_val, X_test = X_train[train_val_idx], X_train[test_idx]
                y_train_val, y_test = y_train_encoded[train_val_idx], y_train_encoded[test_idx]
                groups_train_val = groups[train_val_idx]

                test_year = np.unique(groups[test_idx])
                remaining_years = np.setdiff1d(unique_years, test_year)

                print("Test year:", test_year)

                # Create folder for this test year
                test_year_folder = os.path.join(path, f"test_year_{test_year[0]}")
                os.makedirs(test_year_folder, exist_ok=True)

                test_metrics=[] 

                for inner_fold, val_year in enumerate(remaining_years):
            
                    #Seed unique par modèle
                    seed = get_seed(outer_fold=fold, inner_fold=inner_fold)

                    test_metrics_per_model={}
                    print("Val year:", val_year)

                    # Train = all years except test_year and val_year
                    train_idx = np.where(groups_train_val != val_year)[0]
                    val_idx = np.where(groups_train_val == val_year)[0]

                    X_tr, y_tr = X_train_val[train_idx], y_train_val[train_idx]
                    X_val, y_val = X_train_val[val_idx], y_train_val[val_idx]

                    # Train RandomForest
                    if model_str == "RF":
                        model = BalancedRandomForestClassifier(n_estimators=600,min_samples_split=10,min_samples_leaf=4, 
                                                            max_features='log2', max_depth=None,                                                            
                                                            random_state=seed, class_weight="balanced",n_jobs=-1, 
                                                            sampling_strategy='all', bootstrap=False, replacement=True )

                    # Train SVM
                    elif model_str == "SVM":
                        model = SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=seed)


                    model.fit(X_tr, y_tr)
                    y_val_pred  = model.predict(X_val)

                    val_balanced_acc  = balanced_accuracy_score(y_val, y_val_pred)
                    val_precision_w   = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
                    val_recall_w      = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
                    val_f1_w          = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
                    val_prec_cls      = precision_score(y_val, y_val_pred, average=None, zero_division=0)
                    val_rec_cls       = recall_score(y_val, y_val_pred, average=None, zero_division=0)
                    val_f1_cls        = f1_score(y_val, y_val_pred, average=None, zero_division=0)
    
                    val_metrics = {
                        'val_balanced_accuracy':  val_balanced_acc,
                        'val_precision':          val_precision_w,
                        'val_recall':             val_recall_w,
                        'val_f1':                 val_f1_w,
                        'val_precision_class_0':  float(val_prec_cls[0]),
                        'val_recall_class_0':     float(val_rec_cls[0]),
                        'val_f1_class_0':         float(val_f1_cls[0]),
                        'val_precision_class_1':  float(val_prec_cls[1]),
                        'val_recall_class_1':     float(val_rec_cls[1]),
                        'val_f1_class_1':         float(val_f1_cls[1])
                    }

                    val_metrics_filename = os.path.join(test_year_folder, f'metrics_val_year{val_year}.json')
                    with open(val_metrics_filename, 'w') as f:
                        json.dump(val_metrics, f, indent=4)
                    # Save model
                    model_path = os.path.join(test_year_folder, f"val_{val_year}.joblib")
                    joblib.dump(model, model_path) if model_str in ["RF", "SVM"] else joblib.dump(results, model_path)


                    # ---- Testing----
                    y_test_pred = model.predict(X_test) if model_str in ["RF", "SVM"] else (results.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, exog=X_test) > 0.5).astype(int)

                    test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
                    test_precision_w  = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
                    test_recall_w     = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
                    test_f1_w         = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
                    test_prec_cls     = precision_score(y_test, y_test_pred, average=None, zero_division=0)
                    test_rec_cls      = recall_score(y_test, y_test_pred, average=None, zero_division=0)
                    test_f1_cls       = f1_score(y_test, y_test_pred, average=None, zero_division=0)
                    conf_matrix       = confusion_matrix(y_test, y_test_pred)

                    test_metrics_per_model = {
                            'test_balanced_accuracy':  test_balanced_acc,
                            'test_precision':          test_precision_w,
                            'test_recall':             test_recall_w,
                            'test_f1':                 test_f1_w,
                            'test_precision_class_0':  float(test_prec_cls[0]),
                            'test_recall_class_0':     float(test_rec_cls[0]),
                            'test_f1_class_0':         float(test_f1_cls[0]),
                            'test_precision_class_1':  float(test_prec_cls[1]),
                            'test_recall_class_1':     float(test_rec_cls[1]),
                            'test_f1_class_1':         float(test_f1_cls[1]),
                            'confusion_matrix':        conf_matrix.tolist()
                        }
        
                    test_metrics.append(test_metrics_per_model)

                    # save test metrics
                    test_metrics_filename = os.path.join(test_year_folder, f"test_metrics_val_year{val_year}.json")
                    
                    with open(test_metrics_filename, "w") as f:
                        json.dump(test_metrics_per_model, f, indent=4)

                    # print(
                    #         f"ValYear={val_year} | "
                    #         f"val_bal_acc={val_balanced_acc:.4f} | val_f1_c1={val_f1_cls[1]:.4f} | "
                    #         f"test_bal_acc={test_balanced_acc:.4f} | "
                    #         f"test_recall_c0={test_rec_cls[0]:.4f} | test_recall_c1={test_rec_cls[1]:.4f} | "
                    #     )
                    
                    print(f"Saved model and metrics for TestYear={test_year[0]} ValYear={val_year}")


            #avreaged metrics across all 9 models for each test year
            avg_metrics = {
                'avg_test_balanced_accuracy': np.mean([r['test_balanced_accuracy'] for r in test_metrics]),
                'avg_test_precision': np.mean([r['test_precision'] for r in test_metrics]),
                'avg_test_recall': np.mean([r['test_recall'] for r in test_metrics]),
                'avg_test_F1_score': np.mean([r['test_f1'] for r in test_metrics]),
                'avg_test_precision_class_0': np.mean([r['test_precision_class_0'] for r in test_metrics]),
                'avg_test_precision_class_1': np.mean([r['test_precision_class_1'] for r in test_metrics]),
                'avg_test_recall_class_0': np.mean([r['test_recall_class_0'] for r in test_metrics]),
                'avg_test_recall_class_1': np.mean([r['test_recall_class_1'] for r in test_metrics]),
                'avg_test_F1_score_class0': np.mean([r['test_f1_class_0'] for r in test_metrics]),
                'avg_test_F1_score_class1': np.mean([r['test_f1_class_1'] for r in test_metrics]),
            }
            
            all_test_results.append(avg_metrics)


            # Execution time
            end_time = time.time()
            execution_time = end_time - start_time
            print("execution time: ", end_time - start_time)
            # résultats finaux
            final_results_test = {'execution_time': execution_time,  
                            'test_results': all_test_results}
            
            #conversion de all_test_results before save    
            final_results = convert_ndarray(final_results_test)

            # Save en JSON 
            metrics_filename = f'{path}metrics_hp_{hp}_all_test_years_averaged.json'
            with open(metrics_filename, 'w') as f:
                json.dump(final_results, f, indent=4)
            print(f"All test metrics saved at {metrics_filename}")



        # Majority vote across the 90 models for each test year
        #remove the last dimension 
        X_1_flat = x_rain_scaled.reshape(x_rain_scaled.shape[0], -1)  # --> (n_samples, 336)
        X_2_flat = x_T_scaled.reshape(x_T_scaled.shape[0], -1)  # --> (n_samples, 336)
        X = np.concatenate([X_1_flat, X_2_flat], axis=1)  # (n_samples, nbr_features=336x2)
        
        #initialize lists
        all_preds = []
        all_probs=[]
        all_weighted_probs=[]
        w1_all=[]
        for test_year in [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]:
            model_paths = glob(f"Rockfall_benchmarking/Rain_Temp_all_models/{model_str}/HP_{hp}_{normalisation}/test_year_{test_year}/val_*.joblib")
            if not model_paths:
                print(f"Aucun modèle trouvé pour {test_year}")

            for model_path in model_paths:
                if model_str in ["RF", "SVM"]:
                    clf = joblib.load(model_path)
                    preds = clf.predict(X)   # prédictions de ce modèle


                all_preds.append(preds)


        results = []
        if vote =="HARD" : 
            all_preds_array = np.stack(all_preds)   # de shape (90, 351) 
            #sum of votes for class 1 across models 
            votes_for_class_1 = np.sum(all_preds_array == 1, axis=0) 
            #more than half the models predict 1
            final_preds_majority = (votes_for_class_1 > 90 / 2).astype(int)
            print(y_encoded)
            print(final_preds_majority)


        #Metrics
        accuracy = accuracy_score(y_encoded, final_preds_majority)
        balanced_acc = balanced_accuracy_score(y_encoded, final_preds_majority)

        precision = precision_score(y_encoded, final_preds_majority, average="weighted",zero_division=0)
        recall = recall_score(y_encoded, final_preds_majority, average="weighted",zero_division=0)
        f1 = f1_score(y_encoded, final_preds_majority, average="weighted")
        conf_matrix = confusion_matrix(y_encoded, final_preds_majority)

        #per class metrics
        precision_per_class = precision_score(y_encoded, final_preds_majority, average=None,zero_division=0)
        recall_per_class = recall_score(y_encoded, final_preds_majority, average=None,zero_division=0)
        f1_per_class=f1_score(y_encoded, final_preds_majority, average=None)

        results.append({
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "balanced accuracy" : balanced_acc,
        'precision_class_0': precision_per_class[0],
        'recall_class_0': recall_per_class[0],
        'precision_class_1': precision_per_class[1],
        'recall_class_1': recall_per_class[1],
        'f1_class_0': f1_per_class[0],
        'f1_class_1': f1_per_class[1],
        'confusion_matrix': conf_matrix.tolist()
        })

        print(f"balanced_acc ={balanced_acc:.3f} | Precision={precision:.3f} | Recall={recall:.3f} | F1={f1:.3f} | ")
        print(results)

        with open(os.path.join(path, f"results_vote.json" ), "w") as f:
            json.dump(results, f, indent=4)



    elif model_str in ["CNN", "NARMAX", "InceptionTime_1", "InceptionTime_6", "TimeTransformer"]:
        n_lags_y = 24  #For NARMAX model
        train = True
        if train == True:
            for fold, (train_val_idx, test_idx) in enumerate(gkf.split(x1_tensor, y_encoded, groups=groups)):
                print(f"\n--- fold {fold + 1} ---")

                #split data into train+val and test sets for dual input 
                x1_train_val, x1_test = x1_tensor[train_val_idx], x1_tensor[test_idx]
                x2_train_val, x2_test = x2_tensor[train_val_idx], x2_tensor[test_idx]
                y_train_val, y_test = y_tensor[train_val_idx], y_tensor[test_idx]
                groups_train_val = groups[train_val_idx]
                
                test_year = np.unique(groups[test_idx])
                remaining_years = np.setdiff1d(unique_years, [test_year])

                #select the validation year from the remaining years
                print('---------test_year', test_year)
                # print(remaining_years)

                #create a folder test_year to save the 9 models 
                test_year_folder = os.path.join(path, f'test_year_{test_year[0]}')
                os.makedirs(test_year_folder, exist_ok=True)

                test_metrics=[] #pour stocker la dedans les metriques de chaque test year, pour chaque val
                #initialize best model tracking for this test year
                best_overall_model = None
                best_val_year = None
                best_overall_val_loss = float('inf')


                # for val_year in remaining_years :
                for inner_fold, val_year in enumerate(remaining_years):
                    #Seed unique et reproductible pour ce modèle précis
                    seed = get_seed(outer_fold=fold, inner_fold=inner_fold)
                    set_seed(seed)

                    print('--------val_year', val_year)
                    remaining_train_year= np.setdiff1d(remaining_years, [val_year])

                    train_idx = np.where(groups_train_val != val_year)[0]
                    val_idx = np.where(groups_train_val == val_year)[0]
                    
                    #Diviser dual inputs
                    x1_train, x1_val = x1_train_val[train_idx], x1_train_val[val_idx]
                    x2_train, x2_val = x2_train_val[train_idx], x2_train_val[val_idx]
                    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

                    #load data with dataloader
                    bs=32
                    if model_str == "NARMAX":
                        y_lags_train = build_y_lags(y_train, n_lags_y, device=device)
                        y_lags_val   = build_y_lags(y_val,   n_lags_y, device=device)
                        y_lags_test  = build_y_lags(y_test,  n_lags_y, device=device)
                        train_dataset = NARXDataset(x1_train, x2_train, y_train, y_lags_train)
                        val_dataset   = NARXDataset(x1_val,   x2_val,   y_val,   y_lags_val)
                        test_dataset  = NARXDataset(x1_test,  x2_test,  y_test,  y_lags_test)
                    else:
                        train_dataset = DualInputDataset(x1_train, x2_train, y_train)
                        val_dataset   = DualInputDataset(x1_val,   x2_val,   y_val)
                        test_dataset  = DualInputDataset(x1_test,  x2_test,  y_test)



                    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
                    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
                
                    #configuration model with dual input for this fold

                    if model_str == "CNN":
                        model = CNNLateFusion(input_shape_1=(336, 1), input_shape_2=(336, 1),num_classes=len(np.unique(y_encoded)),
                            n_layers=5, base_filters=16, kernel_size=5, dropout=0.377)
                        model.apply(init_kaiming)

                    elif model_str == "NARMAX":
                        model = NARXModel( n_lags_y=24, n_lags_x=168, num_classes=len(np.unique(y_encoded)),
                            hidden_dims=[128, 64], dropout=0.3, )
                        model.apply(init_kaiming)

                    elif model_str == "InceptionTime_6":
                        model = InceptionTimeModel_6( in_channels_1=1, in_channels_2=1, num_classes=len(np.unique(y_encoded)), 
                                                    n_blocks=2, merge_mode='concat', use_attention=False)
                        model.apply(init_kaiming)
                    elif model_str == "InceptionTime_1":
                        model = InceptionTimeModel_1( in_channels_1=1, in_channels_2=1, num_classes=len(np.unique(y_encoded)),
                                                    n_blocks=1, merge_mode='concat', use_attention=False)
                        model.apply(init_kaiming)

                    elif model_str == "TimeTransformer":
                        # instantiate transformer that expects input [B, T, C] where C=2 (rain,temp)
                        model = DualBranchTimeTransformer(branch_a_cfg = BRANCH_A, branch_b_cfg = BRANCH_B,
                                hidden_dim  = 32, num_classes=len(np.unique(y_encoded)), dropout = 0.2 )
                        model.apply(init_kaiming)

                    model = model.to(device)
                    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
                    optimizer = optim.Adam(model.parameters(), lr=0.001)

                    val_losses = []
                    val_balanced_accuracies = []
                    train_accuracies = []
                    val_f1_scores = []
                    val_recalls = []
                    val_precision = []
                    val_precision_class_0 = []
                    val_recall_class_0 = []
                    val_f1_class_0 = []
                    val_precision_class_1 = []
                    val_recall_class_1 = []
                    val_f1_class_1 = []

                    #initilize each test metrics dict to save metrics for a test year and a specific val_year
                    test_metrics_per_model={}

                    #early stopping parameters
                    patience = 35  
                    best_val_loss = float('inf')  
                    counter = 0 
                    epochs_no_improve=0
                    #------------------------------------training for each fold
                    nb_epochs=200
                    for epoch in range(nb_epochs):
                        model.train()
                        train_loss = 0
                        correct = 0
                        total = 0
                        all_train_targets = []
                        all_train_predictions = []

                        for (data1,data2,*extra), target in train_loader:  #now we take (x1,x2) from data
                            optimizer.zero_grad()
                            if model_str == "NARMAX":
                                y_lags_batch = extra[0].to(device)
                                output = model(data1.to(device), data2.to(device), y_lags_batch)
                            elif model_str == "TimeTransformer":
                                # build [B, T, C] from (B,1,T) tensors
                                d1 = data1.to(device).permute(0,2,1)   # (B, T, 1)
                                d2 = data2.to(device).permute(0,2,1)   # (B, T, 1)
                                output = model(d1, d2)

                            else:
                                output = model(data1.to(device), data2.to(device))

                            loss = criterion(output, target.to(device))
                            loss.backward()
                            optimizer.step()

                            train_loss += loss.item()
                            _, predicted = torch.max(output , 1)
                            total += target.size(0)
                            correct += (predicted == target.to(device)).sum().item()
                            #save truth classes and predictions
                            all_train_targets.extend(target.cpu().numpy())
                            all_train_predictions.extend(predicted.cpu().numpy())


                        #metrics for the training set
                        precision_train = precision_score(all_train_targets, all_train_predictions, average='weighted',zero_division=0)
                        recall_train = recall_score(all_train_targets, all_train_predictions, average='weighted',zero_division=0)

                        class_0_correct = np.sum((all_train_targets == 0) & (all_train_predictions == 0))
                        class_0_total = np.sum(all_train_targets == 0)
                        class_1_correct = np.sum((all_train_targets == 1) & (all_train_predictions == 1))
                        class_1_total = np.sum(all_train_targets == 1)

                        precision_train_per_class = precision_score(all_train_targets, all_train_predictions, average=None,zero_division=0)
                        recall_train_per_class = recall_score(all_train_targets, all_train_predictions, average=None,zero_division=0)


                        
                        #---------------------------------------Validation-----------------------------------------------------
                        model.eval()
                        val_loss = 0
                        correct = 0
                        total = 0
                        all_val_targets = []
                        all_val_predictions = []

                        with torch.no_grad():
                            for (batch1,batch2, *extra), target in val_loader:
                                if model_str == "NARMAX":
                                    y_lags_batch = extra[0].to(device)
                                    output = model(batch1.to(device), batch2.to(device), y_lags_batch)

                                elif model_str == "TimeTransformer":
                                    d1 = batch1.to(device).permute(0,2,1)
                                    d2 = batch2.to(device).permute(0,2,1)
                                    output = model(d1, d2)

                                else:
                                    output = model(batch1.to(device), batch2.to(device))
                                loss = criterion(output, target.to(device))
                                val_loss += loss.item()
                                _, predicted = torch.max(output, 1)
                                total += target.size(0)
                                correct += (predicted == target.to(device)).sum().item()
                                #save targets and predictions for class-wise metrics
                                all_val_targets.extend(target.cpu().numpy())
                                all_val_predictions.extend(predicted.cpu().numpy())

            
                        #calculate class metrics
                        precision = precision_score(all_val_targets, all_val_predictions, average='weighted',zero_division=0)
                        recall = recall_score(all_val_targets, all_val_predictions, average='weighted',zero_division=0)
                        f1_score_metric = f1_score(all_val_targets, all_val_predictions, average='weighted',zero_division=0)
                        val_balanced_accuracy = balanced_accuracy_score(all_val_targets, all_val_predictions)
                        precision_per_class = precision_score(all_val_targets, all_val_predictions, average=None,zero_division=0)
                        recall_per_class = recall_score(all_val_targets, all_val_predictions, average=None,zero_division=0)
                        f1_per_class = f1_score(all_val_targets, all_val_predictions, average=None,zero_division=0)

                        all_val_targets = np.array(all_val_targets)
                        all_val_predictions = np.array(all_val_predictions)
                        class_0_correct = np.sum((all_val_targets == 0) & (all_val_predictions == 0))
                        class_0_total = np.sum(all_val_targets == 0)
                        class_1_correct = np.sum((all_val_targets == 1) & (all_val_predictions == 1))
                        class_1_total = np.sum(all_val_targets == 1)

                        #save global metrics
                        val_losses.append(val_loss / len(val_loader))
                        val_balanced_accuracies.append(val_balanced_accuracy)
                        val_precision.append(precision)
                        val_recalls.append(recall)
                        val_f1_scores.append(f1_score_metric)
                        #save per class metrics
                        val_precision_class_0.append(precision_per_class[0])
                        val_recall_class_0.append(recall_per_class[0])
                        val_f1_class_0.append(f1_per_class[0])

                        val_precision_class_1.append(precision_per_class[1])
                        val_recall_class_1.append(recall_per_class[1])
                        val_f1_class_1.append(f1_per_class[1])

                        print(f'epoch [{epoch + 1}/{nb_epochs}], train loss: {train_loss / len(train_loader):.4f}, '
                            f'val loss: {val_loss / len(val_loader):.4f}, val accuracy: {correct / total:.4f},  recall: {recall:.4f}, '
                            f'val class 0 recall: {recall_per_class[0]:.4f}, val class 1 recall: {recall_per_class[1]:.4f}'
                            )
                    
                        num_epochs_trained = epoch + 1 

                        #early stopping logic per val_year
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            model_state = model.state_dict()
                            
                            epochs_no_improve = 0
                            #track overall best across all val years
                            if val_loss < best_overall_val_loss:
                                best_overall_val_loss = val_loss
                                best_overall_model = model
                                best_val_year = val_year
                                best_overall_state = model.state_dict()
                        else:
                            epochs_no_improve += 1  

                        #if nombre d'epochs sans amelioration depasse patience
                        if epochs_no_improve >= patience:
                            print(f"Early stopping at epoch {epoch+1}")
                            break


                    # plot_training_curves_gkf(train_losses, val_losses, train_accuracies, val_accuracies, path_curves, hp,test_year[0], val_year )

                    val_metrics ={
                        'val_losses': np.mean(val_losses),
                        "F1_scores" : np.mean(val_f1_scores),
                        'val_accuracies': np.mean(val_balanced_accuracies),
                        'val_recalls': np.mean(val_recalls),
                        'val_precision': np.mean(val_precision),
                        'val_precision_class_0': np.mean(val_precision_class_0),
                        'val_recall_class_0': np.mean(val_recall_class_0),
                        'val_f1_class_0': np.mean(val_f1_class_0),
                        'val_precision_class_1': np.mean(val_precision_class_1),
                        'val_recall_class_1': np.mean(val_recall_class_1),
                        'val_f1_class_1': np.mean(val_f1_class_1),
                    }

                    #save each model (parmis les 9 modeles) at each fold of test year 
                    model_path = os.path.join(test_year_folder, f'val_{val_year}.pt')
                    torch.save( model_state, model_path)

                    metrics_filename = os.path.join(test_year_folder, f'metrics_val_year{val_year}.json')
                    with open(metrics_filename, 'w') as f:
                        json.dump(val_metrics, f, indent=4)


                    #------------------------------------------------testing
                    model.eval()
                    test_loss = 0
                    correct = 0
                    total = 0
                    all_test_targets = []
                    all_test_predictions = []
                    all_test_probs = []  #store predicted probabilities

                    with torch.no_grad():
                        for (data1,data2,*extra), target in test_loader:
                            if model_str == "NARMAX":
                                y_lags_batch = extra[0].to(device)
                                output = model(data1.to(device), data2.to(device), y_lags_batch)

                            elif model_str == "TimeTransformer":
                                d1 = data1.to(device).permute(0,2,1)
                                d2 = data2.to(device).permute(0,2,1)
                                output = model(d1, d2)

                            else:
                                output = model(data1.to(device), data2.to(device))
                            loss = criterion(output, target.to(device))
                            test_loss += loss.item()
                            _, predicted = torch.max(output, 1)
                            probs = torch.softmax(output, dim=1) 
                            total += target.size(0)
                            correct += (predicted == target.to(device)).sum().item()

                            #save targets and predictions
                            all_test_targets.extend(target.cpu().numpy())
                            all_test_predictions.extend(predicted.cpu().numpy())
                            all_test_probs.extend(probs.cpu().numpy())#[:, 1])

                    all_test_targets = np.array(all_test_targets)
                    all_test_predictions = np.array(all_test_predictions)
                    all_test_probs = np.array(all_test_probs) 


                    #calculate metrics for test 
                    # test_accuracy = correct / total     
                    test_balanced_accuracy = balanced_accuracy_score(all_test_targets, all_test_predictions)
                    test_precision = precision_score(all_test_targets, all_test_predictions, average='weighted', zero_division=0)
                    test_f1 = f1_score(all_test_targets, all_test_predictions, average='weighted', zero_division=0)
                    test_recall = recall_score(all_test_targets, all_test_predictions, average='weighted', zero_division=0)

                    test_precision_per_class = precision_score(all_test_targets, all_test_predictions, average=None, zero_division=0)
                    test_recall_per_class = recall_score(all_test_targets, all_test_predictions, average=None, zero_division=0)
                    test_f1_per_class = f1_score(all_test_targets, all_test_predictions, average=None, zero_division=0)

                    all_test_targets = np.array(all_test_targets)
                    all_test_predictions = np.array(all_test_predictions)
                    test_class_0_correct = np.sum((all_test_targets == 0) & (all_test_predictions == 0))
                    test_class_0_total = np.sum(all_test_targets == 0)
                    test_class_1_correct = np.sum((all_test_targets == 1) & (all_test_predictions == 1))
                    test_class_1_total = np.sum(all_test_targets == 1)
                    
                    conf_matrix = confusion_matrix(all_test_targets, all_test_predictions)
                    #plot_save_conf_matrix(conf_matrix, [0,1],  test_year_folder)
                    
                    precision_0, recall_0, _0 = precision_recall_curve(all_test_targets == 0, all_test_probs[:, 0])
                    precision_1, recall_1, _1 = precision_recall_curve(all_test_targets == 1, all_test_probs[:, 1])


                    test_metrics_per_model={
                        'test_balanced_accuracy': test_balanced_accuracy,
                        'test_precision': test_precision,
                        'test_recall': test_recall,
                        'test_f1': test_f1,
                        'test_precision_class_0': test_precision_per_class[0],
                        'test_recall_class_0': test_recall_per_class[0],
                        'test_f1_class_0': test_f1_per_class[0],
                        'test_precision_class_1': test_precision_per_class[1],
                        'test_recall_class_1': test_recall_per_class[1],
                        'test_f1_class_1': test_f1_per_class[1],
                        'confusion_matrix': conf_matrix.tolist()
                    }
                    
                    test_metrics.append(test_metrics_per_model)

                    print(f'test loss: {test_loss / len(test_loader):.4f}, test balanced accuracy: {test_balanced_accuracy:.4f}, '
                        f'test f1-score: {test_f1:.4f}, test recall: {test_recall:.4f}, '
                        f'test class 0 recall: {test_recall_per_class[0]:.4f}, test class 1 recall: {test_recall_per_class[1]:.4f}'
                        )
                    
                    metrics_test_filename = os.path.join(test_year_folder, f'test_metrics_val_year{val_year}.json')
                    with open(metrics_test_filename, 'w') as f:
                        json.dump(test_metrics_per_model, f, indent=4)



                #avreaged metrics across all 9 models for each test year
                avg_metrics = {
                    'test_year': test_year,
                    'avg_test_balanced_accuracy': np.mean([r['test_balanced_accuracy'] for r in test_metrics]),
                    'avg_test_precision': np.mean([r['test_precision'] for r in test_metrics]),
                    'avg_test_recall': np.mean([r['test_recall'] for r in test_metrics]),
                    'avg_test_F1_score': np.mean([r['test_f1'] for r in test_metrics]),
                    'avg_test_precision_class_0': np.mean([r['test_precision_class_0'] for r in test_metrics]),
                    'avg_test_precision_class_1': np.mean([r['test_precision_class_1'] for r in test_metrics]),
                    'avg_test_recall_class_0': np.mean([r['test_recall_class_0'] for r in test_metrics]),
                    'avg_test_recall_class_1': np.mean([r['test_recall_class_1'] for r in test_metrics]),
                    'avg_test_F1_score_class0': np.mean([r['test_f1_class_0'] for r in test_metrics]),
                    'avg_test_F1_score_class1': np.mean([r['test_f1_class_1'] for r in test_metrics]),
                }
                
                all_test_results.append(avg_metrics)

            # Execution time
            end_time = time.time()
            execution_time = end_time - start_time
            print("execution time: ", end_time - start_time )
            # résultats finaux
            final_results_test = {'execution_time': execution_time,  
                            'test_results': all_test_results}
            
            #conversion de all_test_results before save    
            final_results = convert_ndarray(final_results_test)

            # Save en JSON 
            metrics_filename = f'{path}metrics_hp_{hp}_all_test_years_averaged.json'
            with open(metrics_filename, 'w') as f:
                json.dump(final_results, f, indent=4)

            print(f"All test metrics saved at {metrics_filename}")



        # Majority voting across 90 models:
        #Initialize lists
        all_preds = []
        all_probs=[]
        all_weighted_probs=[]
        w1_all=[]
        for test_year in [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]:
            model_paths = glob(f"Rockfall_benchmarking/Rain_Temp_all_models/{model_str}/HP_{hp}_{normalisation}/test_year_{test_year}/val_*.pt")
            if not model_paths:
                print(f"Aucun modèle trouvé pour {test_year}")

            for model_path in model_paths:
                metric_path= model_path[:-11]+"test_metrics_val_year"+model_path[-7:-3]+".json"
                #configuration du modèle
                if model_str == "CNN":
                    model = CNNLateFusion(input_shape_1=(336, 1), input_shape_2=(336, 1),num_classes=len(np.unique(y_encoded)),
                        n_layers=5, base_filters=16, kernel_size=5, dropout=0.377)
                    
                elif model_str == "NARMAX":
                    model = NARXModel( n_lags_y=24, n_lags_x=168, num_classes=len(np.unique(y_encoded)),
                                     hidden_dims=[128, 64], dropout=0.3, )

                elif model_str == "InceptionTime_6":
                    model = InceptionTimeModel_6( in_channels_1=1, in_channels_2=1, num_classes=len(np.unique(y_encoded)), 
                                                n_blocks=2, merge_mode='concat', use_attention=False)
                elif model_str == "InceptionTime_1":
                    model = InceptionTimeModel_1( in_channels_1=1, in_channels_2=1, num_classes=len(np.unique(y_encoded)),
                                                n_blocks=1, merge_mode='concat', use_attention=False)
                elif model_str == "TimeTransformer":
                    model = DualBranchTimeTransformer(branch_a_cfg = BRANCH_A, branch_b_cfg = BRANCH_B,
                                hidden_dim  = 32, num_classes=len(np.unique(y_encoded)), dropout = 0.2 )

                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                #loadweights into model
                model.load_state_dict(checkpoint)
                model = model.to(device)
                model.eval()

                with torch.no_grad():
                    if model_str == "NARMAX":
                        y_lags_test  = build_y_lags(y_tensor_blind_test, n_lags_y, device=device)
                        outputs = model(X_tensor_rain_blind_test, X_tensor_T_blind_test, y_lags_test)
                    elif model_str == "TimeTransformer":
                        # blind test tensors are [N,1,T] -> permute to [N,T,1] and concat
                        d1 = X_tensor_rain_blind_test.permute(0,2,1).to(device)
                        d2 = X_tensor_T_blind_test.permute(0,2,1).to(device)
                        outputs = model(d1, d2)
                    else : 
                        outputs = model(X_tensor_rain_blind_test, X_tensor_T_blind_test)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()  # get probabilities
                    all_probs.append(probs) #.cpu().numpy()) 
                    preds = outputs.argmax(dim=1).cpu().numpy() #get predictions 
                    all_preds.append(preds)

        results = []
        if vote =="HARD" : 
            all_preds_array = np.stack(all_preds)   # de shape (90, 352) 
            #sum of votes for class 1 across models 
            votes_for_class_1 = np.sum(all_preds_array == 1, axis=0) 
            #more than half the models predict 1
            final_preds_majority = (votes_for_class_1 > 90 / 2).astype(int)     # 1 if the majority votes 1 else 0
            print(y_tensor_blind_test)
            print(final_preds_majority)

        # Metrics

        accuracy = accuracy_score(y_tensor_blind_test, final_preds_majority)
        balanced_acc = balanced_accuracy_score(y_tensor_blind_test, final_preds_majority)
        precision = precision_score(y_tensor_blind_test, final_preds_majority, average="weighted",zero_division=0)
        recall = recall_score(y_tensor_blind_test, final_preds_majority, average="weighted",zero_division=0)
        f1 = f1_score(y_tensor_blind_test, final_preds_majority, average="weighted")
        conf_matrix = confusion_matrix(y_tensor_blind_test, final_preds_majority)

        #per class metrics
        precision_per_class = precision_score(y_tensor_blind_test, final_preds_majority, average=None,zero_division=0)
        recall_per_class = recall_score(y_tensor_blind_test, final_preds_majority, average=None,zero_division=0)
        f1_per_class=f1_score(y_tensor_blind_test, final_preds_majority, average=None)

        results.append({
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "balanced accuracy" : balanced_acc,
        'precision_class_0': precision_per_class[0],
        'recall_class_0': recall_per_class[0],
        'precision_class_1': precision_per_class[1],
        'recall_class_1': recall_per_class[1],
        'f1_class_0': f1_per_class[0],
        'f1_class_1': f1_per_class[1],
        'confusion_matrix': conf_matrix.tolist()
        })

        print(f"Acuuracy={accuracy:.3f} | Precision={precision:.3f} | Recall={recall:.3f} | F1={f1:.3f} | ")
        print(results)

        with open(os.path.join(path, f"results_vote.json" ), "w") as f:
            json.dump(results, f, indent=4)

