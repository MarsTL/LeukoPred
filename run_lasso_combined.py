
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("leukemia_rnaseq_and_drug_response.csv")

# Gene columns
gene_cols = df.columns[1:944]
drugs = ["Doxorubicin", "Navitoclax"]

# Imputer and scaler setup
imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()

# Store selected features
selected_features = {}

for drug in drugs:
    mask = df[drug].notnull()
    X_raw = df.loc[mask, gene_cols]
    y = df.loc[mask, drug]

    # Impute and scale
    X_imputed = imputer.fit_transform(X_raw)
    X_scaled = scaler.fit_transform(X_imputed)

    # Apply Lasso
    lasso = Lasso(alpha=0.001, max_iter=10000)
    lasso.fit(X_scaled, y)

    # Select non-zero features
    selected = np.array(gene_cols)[lasso.coef_ != 0]
    selected_features[drug] = selected

    print(f" {drug}: Lasso selected {len(selected)} features from {len(gene_cols)} genes.")

    # Save to CSV
    pd.Series(selected).to_csv(f"lasso_selected_features_{drug.lower()}.csv", index=False)
