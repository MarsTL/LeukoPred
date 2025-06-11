#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from naive_bayes_classifier import run_classifier

# Load data
features_df = pd.read_csv("leukemia_features_discretized.csv")
targets_df = pd.read_csv("leukemia_targets_discrete.csv")

# drug to analyze
drug = "Doxorubicin"

# Filter non-null drug targets
valid_targets = targets_df.dropna(subset=[drug])
merged_df = pd.merge(valid_targets[['Sample', drug]], features_df, on='Sample')

# Initialize 
X = merged_df.drop(columns=['Sample', drug, 'Navitoclax_Present', 'Doxorubicin_Present'], errors='ignore')
y = merged_df[drug]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier   
train_data = X_train.copy()
train_data['target'] = y_train

# Predictions
predictions = run_classifier(train_data, 'target', X_test, y_test)
predictions.index = y_test.index  # Align index for evaluation

# rmse and r2
rmse = np.sqrt(mean_squared_error(y_test.astype(int), predictions.astype(int)))
r2 = r2_score(y_test.astype(int), predictions.astype(int))

print(f"\nRMSE: {rmse:.4f}")
print(f"Test R²: {r2:.4f}")
print(f"Drug/s tested: {drug}")

