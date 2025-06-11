
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error, r2_score
from regression_tree import build_tree, predict

df = pd.read_csv("leukemia_rnaseq_and_drug_response.csv")
gene_cols = df.columns[1:944]

def run_pipeline(drug_name):
    df_valid = df[df[drug_name].notnull()]
    X_raw = df_valid[gene_cols]
    y = df_valid[drug_name]

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    lasso = Lasso(alpha=0.05, max_iter=10000)
    lasso.fit(X_scaled, y)
    selected_idx = np.where(lasso.coef_ != 0)[0]
    selected_features = X_raw.columns[selected_idx]
    X_selected = X_scaled[:, selected_idx]

    split = int(0.8 * len(X_selected))
    X_train, X_test = X_selected[:split], X_selected[split:]
    y_train, y_test = y.values[:split], y.values[split:]

    model = build_tree(X_train, y_train, max_depth=5, min_samples_leaf=5, min_std=0.15)
    y_pred = predict(model, X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "Drug": drug_name,
        "Training_Samples": len(y_train),
        "Test_Samples": len(y_test),
        "Original_Features": len(gene_cols),
        "Selected_Features": len(selected_features),
        "Max_Depth": 5,
        "Min_Samples_Leaf": 5,
        "Min_Std_Deviation": 0.15,
        "Test_RMSE": rmse,
        "Test_R2": r2
    }

navitoclax_summary = run_pipeline("Navitoclax")
doxorubicin_summary = run_pipeline("Doxorubicin")

summary_df = pd.DataFrame([
    ["Drug", navitoclax_summary["Drug"], doxorubicin_summary["Drug"]],
    ["Training Samples", navitoclax_summary["Training_Samples"], doxorubicin_summary["Training_Samples"]],
    ["Test Samples", navitoclax_summary["Test_Samples"], doxorubicin_summary["Test_Samples"]],
    ["Original Features", navitoclax_summary["Original_Features"], doxorubicin_summary["Original_Features"]],
    ["Selected Features", navitoclax_summary["Selected_Features"], doxorubicin_summary["Selected_Features"]],
    ["Max Tree Depth", navitoclax_summary["Max_Depth"], doxorubicin_summary["Max_Depth"]],
    ["Min Samples per Leaf", navitoclax_summary["Min_Samples_Leaf"], doxorubicin_summary["Min_Samples_Leaf"]],
    ["AUC Std Threshold", navitoclax_summary["Min_Std_Deviation"], doxorubicin_summary["Min_Std_Deviation"]],
    ["Test RMSE", navitoclax_summary["Test_RMSE"], doxorubicin_summary["Test_RMSE"]],
    ["Test R² Score", navitoclax_summary["Test_R2"], doxorubicin_summary["Test_R2"]],
], columns=["Metric", "Navitoclax", "Doxorubicin"])

print(summary_df.to_string(index=False))
