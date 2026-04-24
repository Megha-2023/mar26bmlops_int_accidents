# Road Accident Prediction in France
This project is an end-to-end Machine Learning pipeline for analyzing and predicting road accident patterns in France using structured national accident data.

## Objective:

The objective of this project is to try to predict the severity of road accidents in France. Predictions will be based on historical data from 2005 to 2020.
Severity can include whether there were injuries, hospitalizations, or fatalities. The predictions will help identify high-risk zones considering meteorology, geography, and accident patterns.


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
Train : 2010 — 2015
Test  : 2016

## Why?
1. Column structures changed in 2019
2. Avoid data leakage
1. Column structures changed in 2019
2. Avoid data leakage
3. Simulates real future prediction

Because in real life:
model learns from past - predicts future

### Model Performance

Current baseline results:

- Accuracy: ~0.61
- F1-score: ~0.59

**Note**: Class imbalance affects minority class prediction performance.

---

#  Key MLOps Concepts Used

- Data Version Control (DVC)
- Reproducible ML pipelines
- Dependency tracking
- Automated execution dvc repro
- Separation of data, code, and models
