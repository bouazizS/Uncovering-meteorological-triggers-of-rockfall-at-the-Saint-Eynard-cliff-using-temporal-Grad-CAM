import numpy as np
import pandas as pd
import time 
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Models_Archi.one_module_inception_attention_temporal_multi_modality import InceptionTimeModel 
from Models_Archi.one_module_inception_attention_temporal_multi_modality_attention import InceptionTimeModelAttention 
from Models_Archi.utils import  process_rockfall_data_multi, charge, preprocess_segments, preprocess_segments_test
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random 
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import f1_score, recall_score, precision_score , balanced_accuracy_score, accuracy_score

import json
import numpy as np
import seaborn as sns
import pickle 
from torch.utils.data import Dataset
from itertools import combinations
from glob import glob 


SEED_ = 42

def get_seed(outer_fold: int, inner_fold: int):
    #Seed unique pour chaque modele dans le fold
    return SEED_ + outer_fold * 100 + inner_fold

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)



def init_kaiming(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        # initialize weights loi normale adapted to Relu activations, 
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

#Convert ndarrays to lists
def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(v) for v in obj]
    else:
        return obj


class MultiModalDataset(Dataset):
    def __init__(self, X_list, y):
        self.X_list = X_list  # list of tensors, each one of shape [n_samples, 1, 336]
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        inputs = [X[idx] for X in self.X_list]
        return inputs, self.y[idx]
    

use_attention = False
use_attention_multihead = False
normalisation = "stscaler"
suf = "_kernels_average"

# Size of the forecasting horizon
hp=1 

# Device configuration 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

path= f'Rockfall_benchmarking/combinations/HP_{hp}_{normalisation}{suf}/'
os.makedirs(path, exist_ok=True)

# Train requirements 
# Load dataframes of Sismo events, Rain RR1 and Temperature
df_sismo= pd.read_csv('.data/events/ev_sismo2.csv')
df_RR1= pd.read_csv('.data/events/1.0_RR1.csv', delimiter='\t')
df_temp = pd.read_csv('.data/events/1.0_ T_St_Hiliaire_filled_complete.csv', delimiter='\t')
# Charge/discharge H 
df_precip = pd.read_csv('/mnt/SSD1/bouazizs/GradCam/Probabilities/data_meteo_Sabrine2.csv', delimiter=';')
df_precip['AAAAMMJJHH'] = df_RR1['AAAAMMJJHH'].values
df_precip['AAAAMMJJHH'] = pd.to_datetime(df_precip['AAAAMMJJHH'])
lamb=0.2
df_precip['H'] = charge(df_precip,lamb) 
# Load dataframe of depth freeze
df_gel= pd.read_csv('.data/events/data_Temp_depth_sabrine_st-H_train.csv', delimiter=',')

# Rain
df_1 = process_rockfall_data_multi(df_RR1, df_sismo, x_days=hp, col_name='RR1',  concat_with_R=True, jour_pluie=14)
df_1['sample_id'] = df_1.index
# Temperature
df_2 = process_rockfall_data_multi(df_temp, df_sismo, x_days=hp, col_name=' T', concat_with_R=False, jour_pluie=14)
df_2['sample_id'] = df_2.index
# Depth freeze 
df_3 = process_rockfall_data_multi(df_gel, df_sismo, x_days=hp, col_name='pgel',  concat_with_R=False, jour_pluie=14)
df_3['sample_id'] = df_3.index
# CHARGE H
df_4 = process_rockfall_data_multi(df_precip, df_sismo, x_days=hp, col_name='H', concat_with_R=False, jour_pluie=14)
df_4['sample_id'] = df_4.index

df = pd.merge(df_1, df_4, on='sample_id')
df = pd.merge(df, df_2, on='sample_id')
df = pd.merge(df, df_3, on='sample_id')
df = df.drop(columns=['sample_id'])

# Set index of df AAAAMMJJHH (to use it for year column!)
df.set_index('AAAAMMJJHH', inplace=True)
# Generate a column of years as groups
df['year'] = pd.to_datetime(df.index).year  # year of rockfall!
df = df[(df['year'] != 2013) & (df['year'] != 2024)]  # remove 2013 year and 2024

# Features
X_rain = df[[f"RR1_hour_{i}" for i in range(336)]].values  #----> column of precipitations
X_temp = df[[f" T_hour_{i}" for i in range(336)]].values  #----> column of temperature
X_charge = df[[f"H_hour_{i}" for i in range(336)]].values  #----> column of charge/discharge
X_depthgel = df[[f"pgel_hour_{i}" for i in range(336)]].values  #----> column of depth freeze

# Z-score normalisation 
x1_scaled, scaler1 = preprocess_segments(X_rain, normalisation) #x_scaled shape (n_samp, 336, 1) 
x2_scaled, scaler2 = preprocess_segments(X_temp, normalisation)    
x3_scaled, scaler3 = preprocess_segments(X_charge, normalisation) 
x4_scaled, scaler4 = preprocess_segments(X_depthgel, normalisation)    
# Save scalers
with open(f"Rockfall_benchmarking/combinations/scaler_depth_gel_{normalisation}.pkl", "wb") as f:
    pickle.dump(scaler4, f)
with open(f"Rockfall_benchmarking/combinations/scaler_h_{normalisation}.pkl", "wb") as f:
    pickle.dump(scaler3, f)
with open(f"Rockfall_benchmarking/combinations/scaler_rr1_{normalisation}.pkl", "wb") as f:
    pickle.dump(scaler1, f)
with open(f"Rockfall_benchmarking/combinations/scaler_temp_{normalisation}.pkl", "wb") as f:
    pickle.dump(scaler2, f)



# Encode labels
y = df['rockfall_target'].values
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

#conversion to pytorch tensors
x1_tensor = torch.FloatTensor(x1_scaled).permute(0, 2, 1)   # shape (n_samp, 1,336)
x2_tensor = torch.FloatTensor(x2_scaled).permute(0, 2, 1)   # shape (n_samp, 1,336)
x3_tensor = torch.FloatTensor(x3_scaled).permute(0, 2, 1)   # shape (n_samp, 1,336)
x4_tensor = torch.FloatTensor(x4_scaled).permute(0, 2, 1)   # shape (n_samp, 1,336)
y_tensor = torch.LongTensor(y_encoded)

# Groups for nested cross validation
groups = df['year'].values  # groups by years
unique_years = np.unique(groups)
gkf = GroupKFold(n_splits=len(unique_years))  # number of groups = unique years

# Class weights 
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weight_tensor = torch.FloatTensor(class_weights).to(device)



# ------------------------------------------------------------
# Blind test requirements :
vote = 'HARD' # majority vote 
#Load data for blind test on year 2024 
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


# Recupure the columns of rain and temp
X_rain_test = df_test[[f"RR1_hour_{i}" for i in range(336)]].values  #----> colonnes de précipitations
X_T_test = df_test[[f" T_hour_{i}" for i in range(336)]].values  #----> colonnes de temperature
X_charge_test = df_test[[f"H_hour_{i}" for i in range(336)]].values  #----> colonnes de charge/discharge
X_depthgel_test = df_test[[f"pgel_hour_{i}" for i in range(336)]].values  #----> colonnes de  profondeur de gel

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
#----------------------------------------------------------------------------------



# all_combinations = []
# for k in range(1, len(feature_names)+1):
#     all_combinations.extend(list(combinations(feature_names, k)))

# for comb in all_combinations:
#     print(comb)

feature_names = ['Rain', 'Temp', 'Charge', 'Freeze']

Rain_KERNELS = {2014: {'k1': 35, 'k2': 43, 'k3': 89}, 2015: {'k1': 13, 'k2': 65, 'k3': 119}, 
                2016: {'k1': 1, 'k2': 37, 'k3': 143}, 2017: {'k1': 27, 'k2': 55, 'k3': 91}, 
                2018: {'k1': 7, 'k2': 61, 'k3': 115}, 2019: {'k1': 29, 'k2': 67, 'k3': 127},
                2020: {'k1': 19, 'k2': 43, 'k3': 133}, 2021: {'k1': 9, 'k2': 67, 'k3': 115}, 
                2022: {'k1': 21, 'k2': 53, 'k3': 113}, 2023: {'k1': 9, 'k2': 49, 'k3': 133}}

Temp_KERNELS = {2014: {'k1': 15, 'k2': 61, 'k3': 107}, 2015: {'k1': 31, 'k2': 59, 'k3': 95},
                2016: {'k1': 21, 'k2': 37, 'k3': 129}, 2017: {'k1': 33, 'k2': 49, 'k3': 91}, 
                2018: {'k1': 17, 'k2': 67, 'k3': 133}, 2019: {'k1': 3, 'k2': 39, 'k3': 115}, 
                2020: {'k1': 25, 'k2': 41, 'k3': 131}, 2021: {'k1': 21, 'k2': 63, 'k3': 93}, 
                2022: {'k1': 17, 'k2': 37, 'k3': 93}, 2023: {'k1': 15, 'k2': 43, 'k3': 123}}

# If taking averages of kernels sizes across folds 
average_kernels = True 
litterature_kernels =  False

if average_kernels:
    # average kernels
    kernels_modalites = {'Rain': [17, 55, 117], 'Charge': [17, 55, 117],
                         'Temp': [19, 49, 111], 'Freeze':[19, 49, 111]}

elif litterature_kernels:  #kernel experts
    kernels_modalites = {'Rain': [9, 19, 39] , 'Charge':  [9, 19, 39] ,
                        'Temp':  [9, 19, 39] , 'Freeze': [9, 19, 39] }
    
# IF taking individual years kernels sizes
else :
    kernels_modalites = { 'Rain': Rain_KERNELS, 'Charge': Rain_KERNELS, 
                     'Temp': Temp_KERNELS, 'Freeze': Temp_KERNELS}


# kernel's experts
# kernels_modalites = {'Rain': [25, 73, 169] , 'Charge': [25, 73, 169] ,
# #                         'Temp': [25, 73, 169] , 'Freeze':[25, 73, 169] }


# Training loop with different inputs combinations
for k in range(1, len(feature_names)+1):

    for comb in combinations(range(len(feature_names)), k):  
        train = True
        blind_test = True  

        names_subset = [feature_names[i] for i in comb]
        print("Training with:", names_subset)      

        # Create the appropriate folder to save models of current combinations inputs 
        comb_str = "_".join(names_subset)
        path_comb = os.path.join(path, f'{comb_str}')
        # os.makedirs(path_comb, exist_ok=True)

        in_channels_list = [1] * len(comb)

        #each modality branch has its own kernels (for average kernels sizes case)
        if average_kernels :
            kernel_sizes_list = [kernels_modalites[name] for name in names_subset] 


        if comb_str in ['Rain_Temp']:
            train = False
            blind_test = False
            # os.makedirs(path_comb, exist_ok=True)

        if train == True:

            start_time = time.time()
            # Initialize lists to save the performance
            fold_results = []
            all_test_metrics = {}
            all_val_metrics={}
            all_train_metrics={}
            # List for saving test results
            all_test_results = []  

            # Groupkfold
            for fold, (train_val_idx, test_idx) in enumerate(gkf.split(x1_tensor, y_encoded, groups=groups)):
                print(f"\n--- fold {fold + 1} ---")

                #split data into train+val and test sets for dual input 
                x1_train_val, x1_test = x1_tensor[train_val_idx], x1_tensor[test_idx]
                x2_train_val, x2_test = x2_tensor[train_val_idx], x2_tensor[test_idx]
                x3_train_val, x3_test = x3_tensor[train_val_idx], x3_tensor[test_idx]
                x4_train_val, x4_test = x4_tensor[train_val_idx], x4_tensor[test_idx]
                
                y_train_val, y_test = y_tensor[train_val_idx], y_tensor[test_idx]
                groups_train_val = groups[train_val_idx]
                
                test_year = np.unique(groups[test_idx])
                remaining_years = np.setdiff1d(unique_years, [test_year])

                # Select the validation year from the remaining years
                print('---------test_year', test_year)

                # Create a folder test_year to save the 9 models 
                test_year_folder = os.path.join(path_comb, f'test_year_{test_year[0]}')
                os.makedirs(test_year_folder, exist_ok=True)

                test_metrics=[] #pour stocker la dedans les metriques de chaque test year, pour chaque val
                #initialize best model tracking for this test year
                best_overall_model = None
                best_val_year = None
                best_overall_val_loss = float('inf')

                # kernels for each fold
                if not average_kernels:
                    kernel_sizes_list = []
                    for name in names_subset:
                        kernel_sizes_list.append([kernels_modalites[name][test_year[0]]["k1"], kernels_modalites[name][test_year[0]]["k2"], kernels_modalites[name][test_year[0]]["k3"]])

                print(kernel_sizes_list)

                for inner_fold, val_year in enumerate(remaining_years):
                    print('--------val_year', val_year)
                    
                    # Seed unique et reproductible pour ce modèle précis
                    seed = get_seed(outer_fold=fold, inner_fold=inner_fold)
                    set_seed(seed)

                    remaining_train_year = np.setdiff1d(remaining_years, [val_year])

                    train_idx = np.where(groups_train_val != val_year)[0]
                    val_idx = np.where(groups_train_val == val_year)[0]
                    
                    # Divide dual inputs to train and val 
                    x1_train, x1_val = x1_train_val[train_idx], x1_train_val[val_idx]
                    x2_train, x2_val = x2_train_val[train_idx], x2_train_val[val_idx]
                    x3_train, x3_val = x3_train_val[train_idx], x3_train_val[val_idx]
                    x4_train, x4_val = x4_train_val[train_idx], x4_train_val[val_idx]
                    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

                    # Create a list of all features for train, val, and test sets in the same order of the list "feature_names" to pick the appropriate ones related to the combinations
                    
                    all_features_train = [x1_train, x2_train, x3_train, x4_train]
                    all_features_val = [x1_val, x2_val, x3_val, x4_val]
                    all_features_test = [x1_test, x2_test, x3_test, x4_test]


                    # Create subsets features based on the current combination
                    X_subset_train = [all_features_train[i] for i in comb]
                    X_subset_val = [all_features_val[i] for i in comb]
                    X_subset_test = [all_features_test[i] for i in comb]

                    # Load data with dataloader
                    bs=32
                    train_dataset = MultiModalDataset(X_subset_train, y_train)
                    val_dataset = MultiModalDataset(X_subset_val, y_val)
                    test_dataset = MultiModalDataset(X_subset_test, y_test)

                    g = torch.Generator()
                    g.manual_seed(seed)

                    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, generator=g)
                    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
                    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
                
                    #configuration model with inputs for the current combinations
                    in_channels_list = [1] * len(comb)

                    if use_attention == False :
                        model = InceptionTimeModel( in_channels_list=in_channels_list, num_classes=len(np.unique(y_encoded)),
                                                    n_blocks=1, kernel_sizes=kernel_sizes_list , merge_mode='concat', use_attention=False)
                    if use_attention == True :
                        model = InceptionTimeModelAttention( in_channels_list=in_channels_list, num_classes=len(np.unique(y_encoded)),
                                                    n_blocks=1, kernel_sizes=kernel_sizes_list , merge_mode='concat', use_modality_attention=True)
                    model.apply(init_kaiming)
                    model = model.to(device)
                    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
                    optimizer = optim.Adam(model.parameters(), lr=0.001)


                    val_losses = []
                    val_accuracies = []
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

                        # Unpack the data_list into separate variables
                        for data_list, target in train_loader: 

                            data_list = [d.to(device) for d in data_list]
                            optimizer.zero_grad()
                            # Pass combinations inputs to the model
                            if use_attention_multihead:
                                output, att = model(*data_list) # dépile la liste 
                            else:
                                output = model(*data_list)
                            loss = criterion(output, target.to(device))
                            loss.backward()
                            optimizer.step()

                            train_loss += loss.item()
                            _, predicted = torch.max(output , 1)
                            total += target.size(0)
                            correct += (predicted == target.to(device)).sum().item()
                            #save les vraies classes et les prédictions
                            all_train_targets.extend(target.cpu().numpy())
                            all_train_predictions.extend(predicted.cpu().numpy())

        
                        #---------------------------------------Validation-----------------------------------------------------
                        model.eval()
                        val_loss = 0
                        correct = 0
                        total = 0
                        all_val_targets = []
                        all_val_predictions = []

                        with torch.no_grad():
                            for batch_list, target in val_loader:
                                if use_attention_multihead:
                                    output, att = model(*[b.to(device) for b in batch_list])
                                else:
                                    output = model(*[b.to(device) for b in batch_list])
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
                        balanced_accuracy = balanced_accuracy_score(all_val_targets, all_val_predictions)

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
                        val_accuracies.append(correct / total)
                        val_balanced_accuracies.append(balanced_accuracy)
                        val_precision.append(precision)
                        val_recalls.append(recall)
                        val_f1_scores.append(f1_score_metric)
                        #save per class metrics
                        val_precision_class_0.append(precision_per_class[0])
                        val_recall_class_0.append(recall_per_class[0])

                        val_precision_class_1.append(precision_per_class[1])
                        val_recall_class_1.append(recall_per_class[1])

                        val_f1_class_0.append(f1_per_class[0])
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


                    val_metrics ={
                        'val_losses': np.mean(val_losses),
                        "F1_scores" : np.mean(val_f1_scores),
                        'val_accuracies': np.mean(val_accuracies),
                        'val_balanced_accuracies': np.mean(val_balanced_accuracies),
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
                        for data_list, target in test_loader:
                            if use_attention_multihead:
                                output,  att = model(*[d.to(device) for d in data_list])
                            else:
                                output = model(*[d.to(device) for d in data_list])
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

                    # print(f'test loss: {test_loss / len(test_loader):.4f}, test balanced accuracy: {test_balanced_accuracy:.4f}, '
                    #     f'test f1-score: {test_f1:.4f}, test recall: {test_recall:.4f}, '
                    #     f'test class 0 recall: {test_recall_per_class[0]:.4f}, test class 1 recall: {test_recall_per_class[1]:.4f}'
                    #     )
                    
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
                    #'all_val_splits': val_split_results
                }
                
                all_test_results.append(avg_metrics)

            end_time = time.time()
            execution_time = end_time - start_time
            print(f"execution time of training for {comb_str}: ", execution_time)

            final_results = {
                'combination': comb_str,
                'hp': hp,
                'execution_time_seconds': execution_time,
                'results_per_year': all_test_results
            }

            #conversion de all_test_results before save 
            final_results = convert_ndarray(final_results)

            #save en JSON
            metrics_filename = f'{path_comb}/metrics_hp_{hp}_all_test_years_averaged.json'
            with open(metrics_filename, 'w') as f:
                json.dump(final_results, f, indent=4)

            print(f"All test metrics for {comb_str} saved at {metrics_filename}")
            print(f"{comb_str} done !!")



        if blind_test == True:
            all_features_blind_test = [X_tensor_rain_blind_test, X_tensor_T_blind_test, X_tensor_H_blind_test, X_tensor_depthgel_blind_test]

            X_BLIND_test = [all_features_blind_test[i] for i in comb]


            # Majority voting across 90 models:
            #Initialize lists
            all_preds = []
            all_probs=[]
            all_weighted_probs=[]
            all_attentions = [] 
            for test_year in [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]:
                model_paths = glob(f"{path_comb}/test_year_{test_year}/val_*.pt")
                if not model_paths:
                    print(f"Aucun modèle trouvé pour {test_year}")

                if not average_kernels:
                    kernel_sizes_list = []
                    for name in names_subset:
                        kernel_sizes_list.append([kernels_modalites[name][test_year]["k1"], kernels_modalites[name][test_year]["k2"], kernels_modalites[name][test_year]["k3"]])
                else :
                    kernel_sizes_list = [kernels_modalites[name] for name in names_subset] 
                    
                for model_path in model_paths:
                    metric_path= model_path[:-11]+"test_metrics_val_year"+model_path[-7:-3]+".json"
                    #configuration du modèle
                    if use_attention == False :
                        model = InceptionTimeModel( in_channels_list=in_channels_list, num_classes=len(np.unique(y_encoded)),
                                                    n_blocks=1, kernel_sizes=kernel_sizes_list , merge_mode='concat', use_attention=False)
                    if use_attention == True :
                        model = InceptionTimeModelAttention( in_channels_list=in_channels_list, num_classes=len(np.unique(y_encoded)),
                                                    n_blocks=1, kernel_sizes=kernel_sizes_list , merge_mode='concat', use_modality_attention=True)
                    
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                    #loadweights into model
                    model.load_state_dict(checkpoint)
                    model = model.to(device)
                    model.eval()

                    with torch.no_grad():
                        if use_attention_multihead:
                            outputs, attentions_blind = model(*X_BLIND_test)
                            all_attentions.append(attentions_blind.cpu().numpy())
                        else:
                            outputs = model(*X_BLIND_test)
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
                final_preds_majority = (votes_for_class_1 > 90 / 2).astype(int)     # 1 si la majorité vote 1 sinon ça donne 0
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

            with open(os.path.join(path_comb, f"results_vote.json" ), "w") as f:
                json.dump(results, f, indent=4)


            if use_attention_multihead and len(all_attentions) > 0:
                   # stack  (n_models, N_samples, M, M)
                   ensemble_attns = np.stack(all_attentions, axis=0)
                   # mean over models (N_samples, M, M)
                   mean_over_models = ensemble_attns.mean(axis=0)
                   # mean over samples (M, M)
                   mean_over_samples = mean_over_models.mean(axis=0)
                   #mean over models and samples 
                   attn_mean = ensemble_attns.mean(axis=0).mean(axis=0)
                   # save arrays for inspection
                   np.save(os.path.join(path_comb, "ensemble_attentions_per_model.npy"), ensemble_attns)
                   np.save(os.path.join(path_comb, "ensemble_mean_per_sample.npy"), mean_over_models)
                   np.save(os.path.join(path_comb, "ensemble_mean_over_samples.npy"), mean_over_samples)
                   print("saved ensemble attention arrays")
