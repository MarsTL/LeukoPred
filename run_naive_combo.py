#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train on single and test on combo

import pandas as pd
import numpy as np
#from sklearn.metrics import mean_squared_error, r2_score
# removing this as these metrics don't seem typical for evaluating categorical data

from naive_bayes_classifier import run_classifier

# Load data
features_df = pd.read_csv("leukemia_features_combo_discretize.csv")
targets_df = pd.read_csv("leukemia_targets_discrete.csv")

# Define drug columns
single_drugs = ["Doxorubicin", "Navitoclax"]
combo_drug = "Doxorubicin:navitoclax"

# Filter where only single agent response is present
filtered_df = features_df[features_df["Doxorubicin_Present_Single"] != features_df["Navitoclax_Present_Single"]]
#valid_targets = targets_df.dropna(subset=[drug])
#merged_df = pd.merge(valid_targets[['Sample', drug]], features_df, on='Sample')
merged_df = pd.merge(filtered_df, targets_df, on = "Sample")
merged_df["AUC"] = merged_df[single_drugs[0]].fillna(merged_df[single_drugs[1]])
# Need to make sure to rename these in the test and train so the model recognizes it as the same features
merged_df["Doxorubicin_Treated"] = merged_df["Doxorubicin_Present_Single"]
merged_df["Navitoclax_Treated"] = merged_df["Navitoclax_Present_Single"]

y_train = merged_df["AUC"]
X_train = merged_df.drop(columns = ["Sample", single_drugs[0],
                                    single_drugs[1], combo_drug,
                                    "Doxorubicin_Present_Single",
                                    "Doxorubicin_Present_Combo",
                                    "Navitoclax_Present_Single",
                                    "Navitoclax_Present_Combo", "AUC"])

valid_targets = targets_df.dropna(subset=[combo_drug])
merged_test = pd.merge(valid_targets[["Sample", combo_drug]], features_df, on="Sample")
merged_test["AUC"] = merged_test[combo_drug]
# Need to make sure to rename these in the test and train so the model recognizes it as the same features
merged_test["Doxorubicin_Treated"] = merged_test["Doxorubicin_Present_Combo"]
merged_test["Navitoclax_Treated"] = merged_test["Navitoclax_Present_Combo"]

y_test = merged_test["AUC"]
X_test = merged_test.drop(columns = ["Sample", combo_drug, "AUC",
                                    "Doxorubicin_Present_Single",
                                    "Doxorubicin_Present_Combo",
                                    "Navitoclax_Present_Single",
                                    "Navitoclax_Present_Combo"])

# Train classifier   
train_data = X_train.copy()
train_data['target'] = y_train

# Predictions
predictions = run_classifier(train_data, "target", X_test, y_test)
predictions.index = y_test.index  # Align index for evaluation

print(f"Drug/s tested: {combo_drug}")


