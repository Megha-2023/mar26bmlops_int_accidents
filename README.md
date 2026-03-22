# mar26bmlops_int_accidents
The objective of this project is to try to predict the severity of road accidents in France. Predictions will be based on historical data.

## Objective:

Predict the severity of road accidents in France using historical data. Severity can include whether there were injuries, hospitalizations, or fatalities. The predictions will help identify high-risk zones considering meteorology, geography, and accident patterns.

## Author
Mohammad Reza Nilchiyan 

MLOps Learning Project – March 2026

# Road Accidents in France – MLOps Project (DVC + ML Pipeline)

This project is an end-to-end Machine Learning pipeline for analyzing and predicting road accident patterns in France using structured national accident data.

The project follows **MLOps best practices** using **DVC for pipeline versioning** and reproducibility.

---

# Project Goal

The goal of this project is to:

- Build a reproducible ML pipeline for accident data
- Clean and preprocess raw datasets
- Train a machine learning model (XGBoost)
- Evaluate model performance
- Ensure full pipeline automation using DVC

---

# Dataset Structure

The dataset is based on official French road accident data and includes:

- `caracteristiques/` → accident characteristics
- `lieux/` → location information
- `usagers/` → users involved in accidents
- `vehicules/` → vehicle information

Final merged dataset: data/accidents_full.csv


target variable will usually come from usagers.grav (severity).

# train/test split idea
idea is Instead of random split we should use time split.
Train : 2010 — 2016
Test  : 2017 — 2018

## Why?
1. Column structures changed in 2019 
2. Avoid data leakage 
3. Simulates real future prediction

Because in real life:
model learns from past - predicts future

* This is a very good ML practice.
Also I, reduce the dataset + features safely.(Feature Engineering solution)


mar26bmlops_int_accidents/
│
├── data/
│   ├   data/ │     
│   ├── caracteristiques/
│   ├── usagers/
│   ├── lieux/
│   ├── vehicules/                   
│   └── accidents_full.csv     merged dataset
│   └── preprocessed/
│               ├── X_train.csv
│               ├── X_test.csv
│               ├── y_train.csv
│               └── y_test.csv
│
├── src/
│   ├
│   └── make_dataset.py       (clean + split + feature engineering)
│   └── train_model.py
│   │    
│   └── evaluate_model.py      (metrics only) (directly inside src)
│
├── models/
│         └── model.pkl                  (later)
├── dvc.yaml                               (later)
├── README.md
├── requirements.txt



## ML Pipeline (DVC)

The project pipeline is fully managed using DVC:

### 1. Prepare Stage
- Loads raw datasets
- Cleans and merges data
- Creates train/test split

Output: data/preprocessed/


---


### 2. Train Stage
- Trains an XGBoost classifier
- Saves trained model

Output: models/xgb_model.pkl


---

### 3. Evaluate Stage
- Evaluates model performance
- Prints metrics (Accuracy, F1-score, Classification Report)

---

### Model Performance

Current baseline results:

- Accuracy: ~0.61
- F1-score: ~0.59

# Note: Class imbalance affects minority class prediction performance.

---

#  Key MLOps Concepts Used

- Data Version Control (DVC)
- Reproducible ML pipelines
- Dependency tracking
- Automated execution dvc repro
- Separation of data, code, and models



