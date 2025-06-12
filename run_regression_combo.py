#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error, r2_score
from regression_tree import train_model, predict, calculate


# Load data
features_df = pd.read_csv("leukemia_rnaseq_and_drug_response.csv")
disc_feat_df = pd.read_csv("leukemia_features_combo_discretize.csv")

# Drug to analyze
#drug = ["Doxorubicin", "Navitoclax"]

# Define drug columns
single_drugs = ["Doxorubicin", "Navitoclax"]
combo_drug = "Doxorubicin:navitoclax"


print(f"\nRunning regression tree for: {combo_drug} ************")  

# Drop rows with missing AUC values for this drug
valid_targets = features_df.dropna(subset=[combo_drug])

# Extract features and target
# Columns 1 to 943 = gene features
test_features = valid_targets.iloc[:, 0:943]
# grab binarized drug presence columns
test_features = pd.merge(test_features, disc_feat_df[["Sample", "Navitoclax_Present_Combo",
                                                      "Doxorubicin_Present_Combo"]], on="Sample")
test_features = test_features.rename(columns = {"Navitoclax_Present_Combo" : "Navitoclax_Treated",
                                                "Doxorubicin_Present_Combo" : "Doxorubicin_Treated"})
target_auc = valid_targets[combo_drug]
target_auc = target_auc.rename("AUC")
X_test = test_features.iloc[:, 1:945].values
y_test = target_auc.values


training_targets = features_df[["Sample", single_drugs[0], single_drugs[1]]]
training_targets = training_targets[training_targets[single_drugs[0]].notna() & features_df[single_drugs[1]].isna() |
                                    training_targets[single_drugs[0]].isna() & features_df[single_drugs[1]].notna()]
training_targets["AUC"] = training_targets[single_drugs[0]].fillna(training_targets[single_drugs[1]])
merged_df = pd.merge(features_df.iloc[:, 0:943], disc_feat_df[["Sample", "Navitoclax_Present_Single",
                                                      "Doxorubicin_Present_Single"]], on="Sample")
merged_df = pd.merge(merged_df, training_targets[["Sample", "AUC"]], on="Sample")
merged_df = merged_df.rename(columns = {"Navitoclax_Present_Single" : "Navitoclax_Treated",
                                                "Doxorubicin_Present_Single" : "Doxorubicin_Treated"})
X_train = merged_df.iloc[:, 1:945].values
y_train = merged_df["AUC"].values

# Train  regression tree
model = train_model(X_train, y_train, max_depth=5, min_leaf_size=5, min_std=0.01, alpha=0.01)

# Predict
y_pred = predict(model, X_test)
calculate(y_test, y_pred)

