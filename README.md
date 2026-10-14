# Uncovering-meteorological-triggers-of-rockfall-at-the-Saint-Eynard-cliff-using-temporal-Grad-CAM


## Overview

This repository contains Python scripts for training, evaluating, and comparing machine learning models for rockfall prediction using meteorological time series and different combinations of environmental input variables.

The repository also includes scripts for computing Grad-CAM explanations to investigate how trained models use temporal information when predicting rockfall events.

## Repository Structure

### Python Scripts

| Script | Description |
|---|---|

| `train_vote.py` | Implements model training and/or prediction procedures based on a voting approach. |
| `compare_benchmarking_models.py` | Compares the performance of different machine learning models for rockfall prediction using a benchmarking framework. |
| `Train_Vote_inputs_combinations_hourly.py` | Trains models using hourly time series and different combinations of input variables using the best architecture. |
| `compare_performances_of_input_combinations.py` | Evaluates and compares model performance across different combinations of environmental input variables. |
| `compute_gradcam.py` | Computes Grad-CAM explanations for trained models to analyze the temporal importance of input signals in model predictions. |

### Directories

| Directory | Description |
|---|---|
| `combinations/` | Contains subdirectories and results associated with different combinations of input variables. |
| `Models_Arch/` | Contains model architecture related implementations used in the experiments. |
| `Rain_Temp_all_models/` | Contains model experiment results associated with rainfall and temperature inputs. |

### Other Files

| File | Description |
|---|---|
| `requirements.txt` | Lists the Python dependencies required to run the scripts. |



## Data

- **Météo France data**:available on Zenodo: zenodo.org/records/19632692

- **Rockfall catalogue**: easydata.earth

## Requirements

Install the required Python dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

Run the relevant script from the repository root, after configuring the required data paths, model directories, and experiment settings.

For example:

```bash
python train_vote.py
python compare_benchmarking_models.py
python Train_Vote_inputs_combinations_hourly.py
python compare_performances_of_input_combinations.py
python compute_gradcam.py
```


**Note:** Check the configuration and expected input files in each script before running it.



