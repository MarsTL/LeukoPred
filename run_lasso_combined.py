
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("leukemia_rnaseq_and_drug_response.csv")

features_df = pd.read_csv("leukemia_rnaseq_and_drug_response.csv")
disc_feat_df = pd.read_csv("leukemia_features_combo_discretize.csv")
single_drugs = ["Doxorubicin", "Navitoclax"]

training_targets = features_df[["Sample", single_drugs[0], single_drugs[1]]]
training_targets = training_targets[training_targets[single_drugs[0]].notna() & features_df[single_drugs[1]].isna() |
                                    training_targets[single_drugs[0]].isna() & features_df[single_drugs[1]].notna()]
training_targets["AUC"] = training_targets[single_drugs[0]].fillna(training_targets[single_drugs[1]])
merged_df = pd.merge(features_df.iloc[:, 0:943], disc_feat_df[["Sample", "Navitoclax_Present_Single",
                                                      "Doxorubicin_Present_Single"]], on="Sample")
merged_df = pd.merge(merged_df, training_targets[["Sample", "AUC"]], on="Sample")
merged_df = merged_df.rename(columns = {"Navitoclax_Present_Single" : "Navitoclax_Treated",
                                               "Doxorubicin_Present_Single" : "Doxorubicin_Treated"})


# Gene columns
feat_cols = df.columns[1:945]
#drugs = ["Doxorubicin", "Navitoclax"]

# Imputer and scaler setup
imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()

# Store selected features
selected_features = {}

#for drug in drugs:
 #   mask = df[drug].notnull()
X_raw = merged_df.iloc[:, 1:945]
y = merged_df["AUC"]

    # Impute and scale
    #X_imputed = imputer.fit_transform(X_raw)
X_scaled = scaler.fit_transform(X_raw)

    # Apply Lasso
lasso = Lasso(alpha=0.001, max_iter=10000)
lasso.fit(X_scaled, y)

    # Select non-zero features
selected = np.array(feat_cols)[lasso.coef_ != 0]
selected_features[drug] = selected

print(f" Combo: Lasso selected {len(selected)} features from {len(feat_cols)} features.")

    # Save to CSV
pd.Series(selected).to_csv(f"lasso_selected_features_combo.csv", index=False)
