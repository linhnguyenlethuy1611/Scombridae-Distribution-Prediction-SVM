# Scombridae-Distribution-Prediction-SVM
Machine Learning project using Support Vector Machines (SVM) to predict the probability of Scombridae fish family occurrence based on environmental and geospatial variables.
## 1. Objective
This project predicts the probability of occurrence of Scombridae species (tuna, mackerels) using environmental variables derived from satellite and oceanographic sources:
- Sea Surface Temperature (SST)
- Sea Surface Salinity (SSS)
- Chlorophyll-a (Chl-a)
- Bathymetry / Depth

A supervised machine learning pipeline is developed using:
- SVM (Support Vector Machine)
- Random Forest
- XGBoost
- Logistic Regression
- Decision Tree

The best model is selected using AUC (Area Under ROC Curve) and exported to generate global probability maps using NetCDF datasets.
This project follows a clean ML architecture with reproducible scripts for preprocessing, training, and evaluation.

## 2. Project structure

Scombridae-Distribution-Prediction-SVM/
│
├── data/
│   ├── raw/                # Original CSV + NetCDF
│   ├── processed/           # Cleaned datasets
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training_SVM.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_spatial_mapping.ipynb
│
├── src/
│   ├── preprocess.py        # Clean CSV → processed dataset
│   ├── train_model.py       # Train ML models and save best model
│   ├── evaluate.py          # Apply model on NetCDF & generate maps
│   └── utils.py
│
├── outputs/
│   ├── model/               # saved scaler.pkl + best_model.pkl
│   ├── figures/             # probability & occurrence maps
│
├── requirements.txt
└── README.md

## 3. Method
- Data collection and merging (species + environment)
- Feature engineering (SST, Chl-a, salinity, lat, lon, etc.)
- Machine Learning Model Training + cross-validation
- Evaluation with ROC-AUC and other metrics
- Spatial prediction mapped to a regular grid
  ## 4. Data processing
### 4.1. Data preprocessing
Source CSV contains environmental data and species presence/absence (occurrenceStatus).
The preprocessing pipeline:
- Removes NaN from SST, SSS, and Chl-a
- Aligns features: depth, SST, SSS, Chl
- Saves cleaned dataset:data/processed/fish_data_clean.csv
Run: python src/preprocess.py

### 4.2. Machine Learning Model Training

| Model               | Accuracy  | AUC       |
| ------------------- | --------- | --------- |
| Logistic Regression | 0.990     | 0.982     |
| Decision Tree       | 0.998     | 0.997     |
| Random Forest       | 0.998     | 1.000     |
| XGBoost             | 0.998     | 1.000     |
| **SVM (best)**      | **0.992** | **1.000** |

SVM with probability=True achieved the best performance.
Scripts create:
outputs/model/best_model.pkl
outputs/model/scaler.pkl
Run training: python src/train_model.py
### 4.3. Predicting Habitat Probability via NetCDF
NetCDF datasets:
- SSS_02_2024.nc
- SST_02_2024.nc
- Chl_02_2024_reprojected.nc

Evaluation steps:
- Load + reshape NetCDF grids
- Flip & clean masked areas
- Scale using saved scaler
- Predict occurrence + probability
- Save global probability map (PNG)

Run: python src/evaluate.py

Output figures are saved in: outputs/figures/probability_map.png

<img width="2280" height="854" alt="prob_map" src="https://github.com/user-attachments/assets/74f68fb8-3bae-4f9c-985c-9daea023c642" />

### 4.4. Skills Demonstrated

✔ Machine Learning
Classification (SVM, RF, XGBoost, Logistic Regression)
Model evaluation (AUC, accuracy)
Probability mapping

✔ Data Engineering
Pipeline design (src/ structure)
Data cleaning, scaling, transformation
Joblib model persistence

✔ Scientific Computing
NetCDF processing via netCDF4
Satellite/oceanographic datasets
Global grid prediction

✔ Visualization
Probability maps (matplotlib)
Geospatial representation

