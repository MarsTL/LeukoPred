#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error, r2_score
from regression_tree import train_model, predict, calculate


# Load data
features_df = pd.read_csv("leukemia_rnaseq_and_drug_response.csv")

# Drug to analyze
drug = ["Doxorubicin", "Navitoclax"]

for drug in drug:
    print(f"\nRunning regression tree for: {drug} ************")  

    # Drop rows with missing AUC values for this drug
    valid_targets = features_df.dropna(subset=[drug])

    # Extract features and target
    # Columns 1 to 943 = gene features
    # changed from 944 to 943 as the previos iteration was incorrectly grabbing the erlotinib column
    X = valid_targets.iloc[:, 1:943].values  
    y = valid_targets[drug].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train  regression tree
    model = train_model(X_train, y_train, max_depth=5, min_leaf_size=5, min_std=0.01, alpha=0.01)

    # Predict
    y_pred = predict(model, X_test)
    calculate(y_test, y_pred)
