
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("leukemia_rnaseq_and_drug_response.csv")

# Gene and target setup
gene_cols = df.columns[1:944]
drug = "Doxorubicin"
mask = df[drug].notnull()

# Gene expressions and AUC values
X_raw = df.loc[mask, gene_cols]
y = df.loc[mask, drug]

# Impute missing values with column mean
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X_raw)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Fit Lasso model
lasso = Lasso(alpha=0.001, max_iter=10000)
lasso.fit(X_scaled, y)

# Extract selected features
selected = np.array(gene_cols)[lasso.coef_ != 0]
print(f" Lasso selected {len(selected)} features from {len(gene_cols)} genes.")

# Save selected feature names
pd.Series(selected).to_csv("lasso_selected_features.csv", index=False)
