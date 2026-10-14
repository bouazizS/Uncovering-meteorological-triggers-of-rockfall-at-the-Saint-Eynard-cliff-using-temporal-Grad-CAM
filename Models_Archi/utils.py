
import wfdb
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

plt.rcParams.update(
    {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": "\n".join(
            [
                r"\usepackage[utf8x]{inputenc}",
                r"\usepackage[T1]{fontenc}",
                #r"\usepackage{cmbright}",
            ]
        ),
    }
)




def preprocess_segments(segments, norm,  transform='none'):
    """ 
    Normalize segments and reshape for the model.
    """
    if transform == 'log10p':
        segments = np.log1p(segments)

    if norm =="stscaler":
        scaler = StandardScaler()
    elif norm =="minmax":
        scaler = MinMaxScaler()  

    segments_scaled = scaler.fit_transform(segments.reshape(-1, segments.shape[-1])).reshape(segments.shape)
    return segments_scaled[..., np.newaxis], scaler  # Add a dimension for Conv1D

def preprocess_segments_test(segments, scaler,  transform='none'):
    """
    Apply a scaler (already fitted on training data) to a segment matrix.
    """
    if transform == 'log1p':
        segments = np.log1p(segments)
    segments_scaled = scaler.transform(segments.reshape(-1, segments.shape[-1])).reshape(segments.shape)
    return segments_scaled[..., np.newaxis]  


def preprocess_segments_test_test(segments, scaler, transform='none'):

    print("\n========== INSIDE preprocess_segments_test ==========")
    print("segments type:", type(segments))
    print("segments shape BEFORE:", segments.shape)
    print("scaler n_features:", scaler.n_features_in_)
    print("transform:", transform)

    if transform == 'log1p':
        segments = np.log1p(segments)

    print("segments shape AFTER transform:", segments.shape)

    segments_2d = segments.reshape(-1, segments.shape[-1])

    print("segments_2d shape:", segments_2d.shape)

    segments_scaled = scaler.transform(segments_2d)

    print("segments_scaled shape:", segments_scaled.shape)

    segments_scaled = segments_scaled.reshape(segments.shape)

    print("segments_scaled reshaped:", segments_scaled.shape)

    return segments_scaled[..., np.newaxis]

def preprocess_segments_multi(segments_temp, segments_rain):
    scaler_temp = StandardScaler()
    scaler_rain = StandardScaler()

    #normalize temperature and precipitation segments separately
    segments_temp_scaled = scaler_temp.fit_transform(segments_temp.reshape(-1, segments_temp.shape[-1])).reshape(segments_temp.shape)
    segments_rain_scaled = scaler_rain.fit_transform(segments_rain.reshape(-1, segments_rain.shape[-1])).reshape(segments_rain.shape)

    #combine both temperature and precipitation segments along the last axis (336, 2)
    segments_combined = np.stack([segments_temp_scaled, segments_rain_scaled], axis=-1)

    return segments_combined, scaler_temp, scaler_rain



def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, output_path, hp):
    epochs_range = range(1, len(train_losses) + 1)

    # Courbe de perte
    loss_filename = f'{output_path}plot_model_loss_acc_hp={hp}.pdf'
    plt.figure(figsize=(12, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Assuming accuracy is between 0 and 1
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Sauvegarder l'image contenant les deux courbes (perte et précision)
    plt.tight_layout()
    plt.savefig(loss_filename)
    plt.close()




def process_rockfall_data_year(df_RR, df_sismo, x_days, jour_pluie=14):
    df_RR['AAAAMMJJHH'] = pd.to_datetime(df_RR['AAAAMMJJHH'])
    df_RR.set_index('AAAAMMJJHH', inplace=True)
    df_RR_per_day = df_RR.resample('1D').sum()
    df_RR_per_day.reset_index('AAAAMMJJHH', inplace=True)
    df_RR.reset_index('AAAAMMJJHH', inplace=True)

    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
    df_sismo['AAAAMMJJHH'] = df_sismo['AAAAMMJJHH'].dt.strftime('%Y-%m-%d')
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])

    #dates communes entre df_sismo et df_RR_per_day
    start_date = max(df_sismo['AAAAMMJJHH'].min(), df_RR_per_day['AAAAMMJJHH'].min())
    end_date = min(df_sismo['AAAAMMJJHH'].max(), df_RR_per_day['AAAAMMJJHH'].max())

    df_sismo_ = df_sismo[(df_sismo['AAAAMMJJHH'] >= start_date) & (df_sismo['AAAAMMJJHH'] <= end_date)]
    df_RR_per_day = df_RR_per_day[(df_RR_per_day['AAAAMMJJHH'] >= start_date) & (df_RR_per_day['AAAAMMJJHH'] <= end_date)]
    df_RR = df_RR[(df_RR['AAAAMMJJHH'] >= start_date) & (df_RR['AAAAMMJJHH'] <= end_date)]

    df_sismo_.loc[:, 'AAAAMMJJHH'] = pd.to_datetime(df_sismo_['AAAAMMJJHH'])
    df_RR_per_day.set_index('AAAAMMJJHH', inplace=True)
    df_RR.set_index('AAAAMMJJHH', inplace=True)

    rockfall_x_days = []
    volume_x_days = []

    #boucle pour vérifier les rockfalls dans les jours suivants
    for date in df_RR_per_day.index:
        end_date_range = date + pd.Timedelta(days=x_days)
        rockfalls_in_x_days = df_sismo_[
            (df_sismo_['AAAAMMJJHH'].dt.date >= date.date()) &
            (df_sismo_['AAAAMMJJHH'].dt.date < end_date_range.date()) &
            (df_sismo_['type'] == 'R')
        ]
        if not rockfalls_in_x_days.empty:
            rockfall_x_days.append(1)
            volume_x_days.append(rockfalls_in_x_days['A(nm/s)'].sum())
        else:
            rockfall_x_days.append(0)
            volume_x_days.append(0)

    #ajout des résultats au dataframe
    df_RR_per_day['rockfall'] = rockfall_x_days
    df_RR_per_day['volume'] = volume_x_days

    #creer les colonnes pour les précipitations des 14 jours précédents
    for i in range(1, jour_pluie + 1):
        df_RR_per_day[f'RR1_day-{i}'] = df_RR_per_day['RR1'].shift(i)

    df_RR_per_day['rockfall_target'] = df_RR_per_day['rockfall'].shift(0).astype(int)
    df_RR_per_day['Amp_target'] = df_RR_per_day['volume'].shift(0)
    df_RR_per_day = df_RR_per_day.dropna()

    #selection des colonnes Rockfall et volume
    df_final_columns = [col for col in df_RR_per_day.columns if 'RR1_day-' in col]
    df_final_columns.extend(['rockfall_target', 'Amp_target'])

    df_hourP_R = df_RR_per_day[df_final_columns].copy()
    df_hourP_R.loc[:, 'AAAAMMJJHH'] = df_RR_per_day.index 

    #creer une plage horaire complète et remplir les heures manquantes
    all_hours = pd.date_range(start=df_RR.index.min(), end=df_RR.index.max(), freq='h')
    df_RR = df_RR.reindex(all_hours, fill_value=0)

    #creer un dataframe avec les précipitations décalées sur 14 jours
    window_size = jour_pluie * 24
    df_shifted = pd.DataFrame(index=df_RR.index)

    #creer un dictionnaire pour stocker les colonnes
    shifted_columns = {f'precip_hour_{i}': df_RR['RR1'].shift(i) for i in range(window_size)}

    #ajouter toutes les colonnes en une seule fois
    df_shifted = pd.concat(shifted_columns, axis=1).fillna(0)

    df_shifted = df_shifted[df_shifted.index.hour == 23]
    df_shifted = df_shifted.iloc[jour_pluie - 1:]  # pour éviter le décalage

    #concaténer avec les cibles rockfall et amplitude
    colonnes_a_concatener = df_hourP_R[['AAAAMMJJHH', 'rockfall_target', 'Amp_target']].reset_index(drop=True)
    df_concatene = pd.concat([df_shifted.reset_index(drop=True), colonnes_a_concatener], axis=1)

    #utiliser la colonne 'AAAAMMJJHH' comme index final
    df_concatene.set_index('AAAAMMJJHH', inplace=True)

    return df_concatene

def charge(dfpluie, lamb):
    pluv = dfpluie['pluvio'].values  
    H = np.zeros_like(pluv) 
    Dt = 1 / 24  
    for p in range(1, len(pluv)):
        H[p] = H[p-1] * np.exp(-Dt / lamb) + pluv[p]
    return H



def process_rockfall_data(df_RR, df_sismo, x_days, jour_pluie=14):
    df_RR['AAAAMMJJHH'] = pd.to_datetime(df_RR['AAAAMMJJHH'])
    df_RR.set_index('AAAAMMJJHH', inplace=True)
    df_RR_per_day = df_RR.resample('1D').sum()
    df_RR_per_day.reset_index('AAAAMMJJHH', inplace=True)
    df_RR.reset_index('AAAAMMJJHH', inplace=True)

    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
    df_sismo['AAAAMMJJHH']=df_sismo['AAAAMMJJHH'].dt.strftime('%Y-%m-%d')
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])


    #dates communes entre df_sismo et df_RR_per_day
    start_date = max(df_sismo['AAAAMMJJHH'].min(), df_RR_per_day['AAAAMMJJHH'].min())
    end_date = min(df_sismo['AAAAMMJJHH'].max(), df_RR_per_day['AAAAMMJJHH'].max())

    df_sismo_ = df_sismo[(df_sismo['AAAAMMJJHH'] >= start_date) & (df_sismo['AAAAMMJJHH'] <= end_date)]
    df_RR_per_day = df_RR_per_day[(df_RR_per_day['AAAAMMJJHH'] >= start_date) & (df_RR_per_day['AAAAMMJJHH'] <= end_date)]
    df_RR = df_RR[(df_RR['AAAAMMJJHH']>= start_date) & (df_RR['AAAAMMJJHH'] <= end_date)]

    #df_sismo_['AAAAMMJJHH'] = pd.to_datetime(df_sismo_['AAAAMMJJHH'])
    df_sismo_.loc[:, 'AAAAMMJJHH'] = pd.to_datetime(df_sismo_['AAAAMMJJHH'])
    df_RR_per_day.set_index('AAAAMMJJHH', inplace=True)
    df_RR.set_index('AAAAMMJJHH', inplace=True)

    rockfall_x_days = []
    volume_x_days = []

    #boucle pour vérifier les rockfalls dans les jours suivants
    for date in df_RR_per_day.index:
        end_date_range = date + pd.Timedelta(days=x_days)
        rockfalls_in_x_days = df_sismo_[
            (df_sismo_['AAAAMMJJHH'].dt.date >= date.date()) &
            (df_sismo_['AAAAMMJJHH'].dt.date < end_date_range.date()) &
            (df_sismo_['type'] == 'R')
        ]
        if not rockfalls_in_x_days.empty:
            rockfall_x_days.append(1)
            volume_x_days.append(rockfalls_in_x_days['A(nm/s)'].sum())
        else:
            rockfall_x_days.append(0)
            volume_x_days.append(0)

    df_RR_per_day['rockfall'] = rockfall_x_days
    df_RR_per_day['volume'] = volume_x_days

    #creation colonnes pour les précipitations des 14 jours précédents
    for i in range(1, jour_pluie+1):
        df_RR_per_day[f'RR1_day-{i}'] = df_RR_per_day['RR1'].shift(i)

    df_RR_per_day['rockfall_target'] = df_RR_per_day['rockfall'].shift(0).astype(int)
    df_RR_per_day['Amp_target'] = df_RR_per_day['volume'].shift(0)
    df_RR_per_day = df_RR_per_day.dropna()

    #colonnes Rockfall et volume
    df_final_columns = [col for col in df_RR_per_day.columns if 'RR1_day-' in col]
    df_final_columns.extend(['rockfall_target', 'Amp_target'])

    df_hourP_R = df_RR_per_day[df_final_columns]


    #creation plage horaire complète et remplir les heures manquantes
    all_hours = pd.date_range(start=df_RR.index.min(), end=df_RR.index.max(), freq='h')
    df_RR = df_RR.reindex(all_hours, fill_value=0)

    #creer dataframe avec les précipitations décalées sur 14 jours
    window_size = jour_pluie * 24
    df_shifted = pd.DataFrame(index=df_RR.index)

    #dictionnaire pour stocker les colonnes
    shifted_columns = {f'precip_hour_{i}': df_RR['RR1'].shift(i) for i in range(window_size)}

    #ajouter les colonnes en une seule fois
    df_shifted = pd.concat(shifted_columns, axis=1).fillna(0)


    df_shifted = df_shifted[df_shifted.index.hour == 23]
    df_shifted = df_shifted.iloc[jour_pluie - 1:]  # pour éviter le décalage

    #concat avec les targets rockfall et amplitude
    colonnes_a_concatener = df_hourP_R[['rockfall_target', 'Amp_target']]
    df_concatene = pd.concat([df_shifted.reset_index(drop=True), colonnes_a_concatener.reset_index(drop=True)], axis=1)

    return df_concatene



def process_rockfall_data_multi(df_RR, df_sismo, x_days, col_name='RR1',concat_with_R=True, jour_pluie=14):
    df_RR['AAAAMMJJHH'] = pd.to_datetime(df_RR['AAAAMMJJHH'])
    df_RR.set_index('AAAAMMJJHH', inplace=True)
    if col_name=='RR1':
      df_RR_per_day = df_RR.resample('1D').sum()
      df_RR_per_day.reset_index('AAAAMMJJHH', inplace=True)
      df_RR.reset_index('AAAAMMJJHH', inplace=True)
    else: 
      df_RR_per_day = df_RR.resample('1D').mean()
      df_RR_per_day.reset_index('AAAAMMJJHH', inplace=True)
      df_RR.reset_index('AAAAMMJJHH', inplace=True)   

    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
    df_sismo['AAAAMMJJHH']=df_sismo['AAAAMMJJHH'].dt.strftime('%Y-%m-%d')
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])


    #dates communes entre df_sismo et df_RR_per_day
    start_date = max(df_sismo['AAAAMMJJHH'].min(), df_RR_per_day['AAAAMMJJHH'].min())
    end_date = min(df_sismo['AAAAMMJJHH'].max(), df_RR_per_day['AAAAMMJJHH'].max())

    df_sismo_ = df_sismo[(df_sismo['AAAAMMJJHH'] >= start_date) & (df_sismo['AAAAMMJJHH'] <= end_date)]
    df_RR_per_day = df_RR_per_day[(df_RR_per_day['AAAAMMJJHH'] >= start_date) & (df_RR_per_day['AAAAMMJJHH'] <= end_date)]
    df_RR = df_RR[(df_RR['AAAAMMJJHH']>= start_date) & (df_RR['AAAAMMJJHH'] <= end_date)]

    #df_sismo_['AAAAMMJJHH'] = pd.to_datetime(df_sismo_['AAAAMMJJHH'])
    df_sismo_.loc[:, 'AAAAMMJJHH'] = pd.to_datetime(df_sismo_['AAAAMMJJHH'])
    df_RR_per_day.set_index('AAAAMMJJHH', inplace=True)
    df_RR.set_index('AAAAMMJJHH', inplace=True)

    rockfall_x_days = []
    volume_x_days = []

    #boucle pour vérifier les rockfalls dans les jours suivants
    for date in df_RR_per_day.index:
        end_date_range = date + pd.Timedelta(days=x_days)
        rockfalls_in_x_days = df_sismo_[
            (df_sismo_['AAAAMMJJHH'].dt.date >= date.date()) &
            (df_sismo_['AAAAMMJJHH'].dt.date < end_date_range.date()) &
            (df_sismo_['type'] == 'R')
        ]
        if not rockfalls_in_x_days.empty:
            rockfall_x_days.append(1)
            volume_x_days.append(rockfalls_in_x_days['A(nm/s)'].sum())
        else:
            rockfall_x_days.append(0)
            volume_x_days.append(0)

    df_RR_per_day['rockfall'] = rockfall_x_days
    df_RR_per_day['volume'] = volume_x_days

    #creation colonnes pour les précipitations des 14 jours précédents
    for i in range(1, jour_pluie+1):
        df_RR_per_day[f'{col_name}_day-{i}'] = df_RR_per_day[col_name].shift(i)

    df_RR_per_day['rockfall_target'] = df_RR_per_day['rockfall'].shift(0).astype(int)
    df_RR_per_day['Amp_target'] = df_RR_per_day['volume'].shift(0)
    df_RR_per_day = df_RR_per_day.dropna()

    #colonnes Rockfall et volume
    df_final_columns = [col for col in df_RR_per_day.columns if f'{col_name}_day-' in col]
    df_final_columns.extend(['rockfall_target', 'Amp_target'])

    df_hourP_R = df_RR_per_day[df_final_columns]
    df_hourP_R.loc[:, 'AAAAMMJJHH'] = df_RR_per_day.index 


    #creation plage horaire complète et remplir les heures manquantes
    all_hours = pd.date_range(start=df_RR.index.min(), end=df_RR.index.max(), freq='h')
    df_RR = df_RR.reindex(all_hours, fill_value=0)

    #creer dataframe avec les précipitations décalées sur 14 jours
    window_size = jour_pluie * 24
    df_shifted = pd.DataFrame(index=df_RR.index)

    #dictionnaire pour stocker les colonnes
    shifted_columns = {f'{col_name}_hour_{i}': df_RR[col_name].shift(i) for i in range(window_size)}

    #ajouter les colonnes en une seule fois
    df_shifted = pd.concat(shifted_columns, axis=1).fillna(0)


    df_shifted = df_shifted[df_shifted.index.hour == 23]
    df_shifted = df_shifted.iloc[jour_pluie - 1:]  # pour éviter le décalage

    #concat avec les targets rockfall et amplitude
    colonnes_a_concatener = df_hourP_R[['AAAAMMJJHH','rockfall_target', 'Amp_target']]
    df_concatene = pd.concat([df_shifted.reset_index(drop=True), colonnes_a_concatener.reset_index(drop=True)], axis=1)
    #utiliser la colonne 'AAAAMMJJHH' comme index final
    #df_concatene.set_index('AAAAMMJJHH', inplace=True)

    if concat_with_R:
     return df_concatene
    else :
      return df_shifted.reset_index(drop=True)





def process_rockfall_data_multi_day_min_max_mean_median( df_TEMP,   df_sismo,  x_days, col_name='TEMP',jour_pluie=14):

    df_TEMP['AAAAMMJJHH'] = pd.to_datetime(df_TEMP['AAAAMMJJHH'])
    df_TEMP.set_index('AAAAMMJJHH', inplace=True)

    #agrégation journalière : min et max
    df_TEMP_per_day = df_TEMP.resample('1D').agg({ col_name: ['min', 'max', 'mean', 'median']})

    df_TEMP_per_day.columns = [f'{col_name}_min', f'{col_name}_max', f'{col_name}_mean', f'{col_name}_median']
    df_TEMP_per_day.reset_index(inplace=True)

    df_TEMP.reset_index(inplace=True)

    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'].dt.date)

    # dates communes 
    start_date = max(df_sismo['AAAAMMJJHH'].min(), df_TEMP_per_day['AAAAMMJJHH'].min())
    end_date   = min(df_sismo['AAAAMMJJHH'].max(), df_TEMP_per_day['AAAAMMJJHH'].max())

    df_sismo_ = df_sismo[   (df_sismo['AAAAMMJJHH'] >= start_date) & (df_sismo['AAAAMMJJHH'] <= end_date) ]

    df_TEMP_per_day = df_TEMP_per_day[(df_TEMP_per_day['AAAAMMJJHH'] >= start_date) & (df_TEMP_per_day['AAAAMMJJHH'] <= end_date) ]

    df_TEMP_per_day.set_index('AAAAMMJJHH', inplace=True)

    # rockfall HP
    rockfall_x_days = []
    volume_x_days = []

    for date in df_TEMP_per_day.index:
        end_date_range = date + pd.Timedelta(days=x_days)

        rockfalls = df_sismo_[
            (df_sismo_['AAAAMMJJHH'] >= date) &
            (df_sismo_['AAAAMMJJHH'] < end_date_range) &
            (df_sismo_['type'] == 'R') ]

        if not rockfalls.empty:
            rockfall_x_days.append(1)
            volume_x_days.append(rockfalls['A(nm/s)'].sum())
        else:
            rockfall_x_days.append(0)
            volume_x_days.append(0)

    df_TEMP_per_day['rockfall_target'] = rockfall_x_days
    df_TEMP_per_day['Amp_target'] = volume_x_days

    #création des features min/max temp 
    for i in range(1, jour_pluie + 1):
        df_TEMP_per_day[f'{col_name}_min_day_{i}'] = df_TEMP_per_day[f'{col_name}_min'].shift(i)
        df_TEMP_per_day[f'{col_name}_max_day_{i}'] = df_TEMP_per_day[f'{col_name}_max'].shift(i)
        df_TEMP_per_day[f'{col_name}_mean_day_{i}'] = df_TEMP_per_day[f'{col_name}_mean'].shift(i)
        df_TEMP_per_day[f'{col_name}_median_day_{i}'] = df_TEMP_per_day[f'{col_name}_median'].shift(i)

    df_TEMP_per_day = df_TEMP_per_day.dropna()

    #colonnes finales ---
    feature_cols = (
        [f'{col_name}_min_day_{i}' for i in range(1, jour_pluie + 1)] +
        [f'{col_name}_max_day_{i}' for i in range(1, jour_pluie + 1)]  +
        [f'{col_name}_mean_day_{i}' for i in range(1, jour_pluie + 1)]  +
        [f'{col_name}_median_day_{i}' for i in range(1, jour_pluie + 1)]
    )

    feature_cols += ['rockfall_target', 'Amp_target']

    df_final = df_TEMP_per_day[feature_cols].copy()
    df_final['AAAAMMJJHH'] = df_TEMP_per_day.index
    df_final.set_index('AAAAMMJJHH', inplace=True)

    return df_final




def process_rockfall_data_multi_day(df_RR, df_sismo, x_days, col_name='RR1', jour_pluie=14):
    df_RR['AAAAMMJJHH'] = pd.to_datetime(df_RR['AAAAMMJJHH'])
    df_RR.set_index('AAAAMMJJHH', inplace=True)
    if col_name=='RR1':
      df_RR_per_day = df_RR.resample('1D').sum()
      df_RR_per_day.reset_index('AAAAMMJJHH', inplace=True)
      df_RR.reset_index('AAAAMMJJHH', inplace=True)
    else:
      df_RR_per_day = df_RR.resample('1D').mean()
      df_RR_per_day.reset_index('AAAAMMJJHH', inplace=True)
      df_RR.reset_index('AAAAMMJJHH', inplace=True)

    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
    df_sismo['AAAAMMJJHH']=df_sismo['AAAAMMJJHH'].dt.strftime('%Y-%m-%d')
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])


    #dates communes entre df_sismo et df_RR_per_day
    start_date = max(df_sismo['AAAAMMJJHH'].min(), df_RR_per_day['AAAAMMJJHH'].min())
    end_date = min(df_sismo['AAAAMMJJHH'].max(), df_RR_per_day['AAAAMMJJHH'].max())

    df_sismo_ = df_sismo[(df_sismo['AAAAMMJJHH'] >= start_date) & (df_sismo['AAAAMMJJHH'] <= end_date)]
    df_RR_per_day = df_RR_per_day[(df_RR_per_day['AAAAMMJJHH'] >= start_date) & (df_RR_per_day['AAAAMMJJHH'] <= end_date)]
    df_RR = df_RR[(df_RR['AAAAMMJJHH']>= start_date) & (df_RR['AAAAMMJJHH'] <= end_date)]

    #df_sismo_['AAAAMMJJHH'] = pd.to_datetime(df_sismo_['AAAAMMJJHH'])
    df_sismo_.loc[:, 'AAAAMMJJHH'] = pd.to_datetime(df_sismo_['AAAAMMJJHH'])
    df_RR_per_day.set_index('AAAAMMJJHH', inplace=True)
    df_RR.set_index('AAAAMMJJHH', inplace=True)

    rockfall_x_days = []
    volume_x_days = []

    #boucle pour vérifier les rockfalls dans les jours suivants
    for date in df_RR_per_day.index:
        end_date_range = date + pd.Timedelta(days=x_days)
        rockfalls_in_x_days = df_sismo_[
            (df_sismo_['AAAAMMJJHH'].dt.date >= date.date()) &
            (df_sismo_['AAAAMMJJHH'].dt.date < end_date_range.date()) &
            (df_sismo_['type'] == 'R')
        ]
        if not rockfalls_in_x_days.empty:
            rockfall_x_days.append(1)
            volume_x_days.append(rockfalls_in_x_days['A(nm/s)'].sum())
        else:
            rockfall_x_days.append(0)
            volume_x_days.append(0)

    df_RR_per_day['rockfall'] = rockfall_x_days
    df_RR_per_day['volume'] = volume_x_days

    #creation colonnes pour les précipitations des 14 jours précédents
    for i in range(1, jour_pluie+1):
        df_RR_per_day[f'{col_name}_day_{i}'] = df_RR_per_day[col_name].shift(i)

    df_RR_per_day['rockfall_target'] = df_RR_per_day['rockfall'].shift(0).astype(int)
    df_RR_per_day['Amp_target'] = df_RR_per_day['volume'].shift(0)
    df_RR_per_day = df_RR_per_day.dropna()

    #colonnes Rockfall et volume
    df_final_columns = [col for col in df_RR_per_day.columns if f'{col_name}_day_' in col]
    df_final_columns += ['rockfall_target', 'Amp_target']

    df_hourP_R = df_RR_per_day[df_final_columns].copy()
    df_hourP_R['AAAAMMJJHH'] = df_RR_per_day.index
    df_hourP_R.set_index('AAAAMMJJHH', inplace=True)

    return df_hourP_R

