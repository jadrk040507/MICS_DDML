"""
Water Treatment Predictors Analysis
LASSO-Logistic Regression and XGBoost Variable Importance
Uses config.py for data path
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

from config import DATA_FILE

np.random.seed(42)

print('Loading data...')
df = pd.read_stata(DATA_FILE)
print(f'Sample size: {len(df):,} households')

# =============================================================================
# Convert categorical variables to numeric
# =============================================================================

# RiskSource: categorical strings -> numeric (0, 1, 2)
df['RiskSource_num'] = df['RiskSource'].map({
    'No risk': 0, 
    'Moderate to high risk': 1, 
    'Very high risk': 2
}).astype(float)

# urban: categorical strings -> binary (0=Rural, 1=Urban)
df['urban_num'] = (df['urban'] == 'Urban').astype(float)

# HHSEX: categorical strings -> binary (1=Male, 0=Female)
df['HHSEX_num'] = (df['HHSEX'] == '1. Male').astype(float)

# WQ15_g: recode to numeric (0=Nothing, 1=Boil, 2=Chlorine, 3=Strain, 4=Other)
df['WQ15_g_num'] = df['WQ15_g'].map({
    'Treat: Nothing': 0,
    'Treat: Boil': 1,
    'Treat: Chlorine/Aquatabs/PUR': 2,
    'Treat: Strain/Settle': 3,
    'Treat: Other': 4
}).astype(float)

# =============================================================================
# Define variables
# =============================================================================

treatment_any = 'water_treatment'
treatment_multi = 'WQ15_g_num'
source_risk = 'RiskSource_num'
wealth_dummies = ['windex5_1', 'windex5_2', 'windex5_3', 'windex5_4', 'windex5_5']
edu_var = 'water_carrier_edu'
urban = 'urban_num'
sanitation = 'Sanitation_ladder'
age = 'mean_child_age'
sex = 'HHSEX_num'
water_source_dummies = ['PipedWater', 'WellandSpringWater', 'RainandSurfaceWater', 'PurchasedWater']
region_var = 'Region'
country_var = 'Country'
rainy = 'rainy_season'

# =============================================================================
# Create dummy variables for categorical predictors
# =============================================================================

# Convert to strings for dummy creation
df[region_var] = df[region_var].astype(str)
df[country_var] = df[country_var].astype(str)
df[sanitation] = df[sanitation].astype(str)
df[edu_var] = df[edu_var].astype(str)

# Create dummies
region_dum = pd.get_dummies(df[region_var], prefix='region', drop_first=True)
country_dum = pd.get_dummies(df[country_var], prefix='country', drop_first=True)
sanitation_dum = pd.get_dummies(df[sanitation], prefix='sanit', drop_first=True)
edu_dum = pd.get_dummies(df[edu_var], prefix='edu', drop_first=True)

# Add dummies to dataframe
for col in region_dum.columns:
    df[col] = region_dum[col].astype(float)
for col in country_dum.columns:
    df[col] = country_dum[col].astype(float)
for col in sanitation_dum.columns:
    df[col] = sanitation_dum[col].astype(float)
for col in edu_dum.columns:
    df[col] = edu_dum[col].astype(float)

# =============================================================================
# Build predictor list
# =============================================================================

predictors = (
    [source_risk, urban, age, sex, rainy] +
    wealth_dummies +
    water_source_dummies +
    list(region_dum.columns) +
    list(country_dum.columns) +
    list(sanitation_dum.columns) +
    list(edu_dum.columns)
)

# =============================================================================
# Create analysis sample
# =============================================================================

df_analysis = df[predictors + [treatment_any, treatment_multi]].copy().dropna()
print(f'Analysis sample: {len(df_analysis):,} households')

# Ensure all predictors are float
for col in predictors:
    df_analysis[col] = df_analysis[col].astype(float)

X = df_analysis[predictors].values
y = df_analysis[treatment_any].values

# =============================================================================
# PART 1: LASSO-Logistic Regression
# =============================================================================

print('\n' + '='*80)
print('PART 1: LASSO-Logistic Regression')
print('='*80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print('Fitting LASSO...')
lasso_cv = LogisticRegressionCV(
    Cs=10, cv=5, penalty='l1', solver='saga',
    max_iter=10000, scoring='roc_auc',
    random_state=42, n_jobs=-1
)
lasso_cv.fit(X_scaled, y)

selected_mask = lasso_cv.coef_[0] != 0
selected_vars = [predictors[i] for i in range(len(predictors)) if selected_mask[i]]
print(f'Selected: {len(selected_vars)}/{len(predictors)}')

coef_df = pd.DataFrame({
    'variable': predictors,
    'coefficient': lasso_cv.coef_[0],
    'selected': selected_mask
})
coef_df = coef_df[coef_df['selected']].sort_values('coefficient', key=abs, ascending=False)

pred_probs = lasso_cv.predict_proba(X_scaled)[:, 1]
mean_prob = pred_probs.mean()
coef_df['AME'] = coef_df['coefficient'] * mean_prob * (1 - mean_prob)

print('Bootstrap SEs...')
n_bootstrap = 100
ame_bootstrap = []
for i in range(n_bootstrap):
    idx = np.random.choice(len(X_scaled), size=len(X_scaled), replace=True)
    lasso_boot = LogisticRegressionCV(
        Cs=10, cv=5, penalty='l1', solver='saga',
        max_iter=10000, random_state=42, n_jobs=-1
    )
    lasso_boot.fit(X_scaled[idx], y[idx])
    pred_boot = lasso_boot.predict_proba(X_scaled)[:, 1]
    ame_boot = lasso_boot.coef_[0] * pred_boot.mean() * (1 - pred_boot.mean())
    ame_bootstrap.append(ame_boot)

coef_df['SE'] = [np.std(ame_bootstrap, axis=0)[i] for i in range(len(predictors)) if selected_mask[i]]
coef_df['p_value'] = 2 * (1 - pd.Series(np.abs(coef_df['AME']) / coef_df['SE']).map(
    lambda x: 0.5 * (1 + np.math.erf(x / np.sqrt(2)))
))

auc_score = roc_auc_score(y, pred_probs)
print(f'AUC: {auc_score:.4f}')

coef_df['AME_pct'] = coef_df['AME'] * 100
coef_df['SE_pct'] = coef_df['SE'] * 100

# =============================================================================
# Generate LASSO LaTeX table
# =============================================================================

def fmt_latex(row):
    sig = '***' if row['p_value'] < 0.01 else '**' if row['p_value'] < 0.05 else '*' if row['p_value'] < 0.1 else ''
    var_name = row['variable'].replace('_', ' ').title()
    return f"{var_name} & {row['AME_pct']:.2f}{sig} & ({row['SE_pct']:.2f}) \\\\"

lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{LASSO-Logistic: Predictors of Water Treatment}")
lines.append(r"\label{tab:lasso_any}")
lines.append(r"\begin{tabular}{lcc}")
lines.append(r"\hline\hline")
lines.append("Variable & AME (pp) & (SE) \\\\")
lines.append(r"\hline")
for _, row in coef_df.iterrows():
    lines.append(fmt_latex(row))
lines.append(r"\hline")
lines.append(f"Observations & {len(df_analysis)} & \\\\")
lines.append(f"AUC & {auc_score:.3f} & \\\\")
lines.append(f"Selected & {len(selected_vars)}/{len(predictors)} & \\\\")
lines.append(r"\hline\hline")
lines.append(r"\multicolumn{3}{l}{\footnotesize Notes: LASSO with 10-fold CV. * p<0.1, ** p<0.05, *** p<0.01} \\\\")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

with open("Table/LASSO_AnyTreatment.tex", "w") as f:
    f.write("\n".join(lines))
print("Saved: Table/LASSO_AnyTreatment.tex")

# =============================================================================
# PART 2: XGBoost
# =============================================================================

print('\n' + '='*80)
print('PART 2: XGBoost')
print('='*80)

xgb_model = xgb.XGBClassifier(
    max_depth=4, learning_rate=0.1, n_estimators=100,
    random_state=42, n_jobs=-1, eval_metric='logloss'
)
xgb_model.fit(X_scaled, y)

importance_df = pd.DataFrame({
    'variable': predictors,
    'gain': xgb_model.feature_importances_
})
importance_df = importance_df.sort_values('gain', ascending=False)
importance_df['gain_pct'] = importance_df['gain'] / importance_df['gain'].sum() * 100

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X_scaled, y, cv=cv, scoring='roc_auc')
print(f'XGBoost AUC: {cv_scores.mean():.4f}')

lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{XGBoost Variable Importance}")
lines.append(r"\label{tab:xgb_importance}")
lines.append(r"\begin{tabular}{lrr}")
lines.append(r"\hline\hline")
lines.append("Variable & Gain & Gain (\\%) \\\\")
lines.append(r"\hline")
for _, row in importance_df.head(15).iterrows():
    lines.append(f"{row['variable'].replace('_', ' ').title()} & {row['gain']:.4f} & {row['gain_pct']:.1f}\\% \\\\")
lines.append(r"\hline")
lines.append(f"Observations & {len(df_analysis)} & \\\\")
lines.append(f"CV AUC & {cv_scores.mean():.3f} & \\\\")
lines.append(r"\hline\hline")
lines.append(r"\multicolumn{3}{l}{\footnotesize Notes: XGBoost max\_depth=4. Gain = loss reduction.} \\\\")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

with open("Table/XGBoost_VariableImportance.tex", "w") as f:
    f.write("\n".join(lines))
print("Saved: Table/XGBoost_VariableImportance.tex")

# =============================================================================
# PART 3: OLS Comparison
# =============================================================================

print('\n' + '='*80)
print('PART 3: OLS')
print('='*80)

X_ols = df_analysis[selected_vars]
X_ols_const = sm.add_constant(X_ols)
ols_results = sm.OLS(y, X_ols_const).fit(cov_type='HC3')

lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{OLS: Predictors of Water Treatment}")
lines.append(r"\label{tab:ols_treatment}")
lines.append(r"\begin{tabular}{lcc}")
lines.append(r"\hline\hline")
lines.append("Variable & Coef & (SE) \\\\")
lines.append(r"\hline")
for var in selected_vars:
    coef, se, pval = ols_results.params[var], ols_results.bse[var], ols_results.pvalues[var]
    sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
    lines.append(f"{var.replace('_', ' ').title()} & {coef*100:.2f}{sig} & ({se*100:.2f}) \\\\")
lines.append(r"\hline")
lines.append(f"Constant & {ols_results.params['const']*100:.2f}*** & ({ols_results.bse['const']*100:.2f}) \\\\")
lines.append(f"Observations & {len(df_analysis)} & \\\\")
lines.append(f"R-squared & {ols_results.rsquared:.3f} & \\\\")
lines.append(r"\hline\hline")
lines.append(r"\multicolumn{3}{l}{\footnotesize Notes: OLS with robust SE. * p<0.1, ** p<0.05, *** p<0.01} \\\\")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

with open("Table/OLS_Treatment.tex", "w") as f:
    f.write("\n".join(lines))
print("Saved: Table/OLS_Treatment.tex")

# =============================================================================
# Summary
# =============================================================================

print('\n' + '='*80)
print('DONE')
print('='*80)
print(f'LASSO AUC: {auc_score:.4f}')
print(f'XGBoost AUC: {cv_scores.mean():.4f}')
print(f"Top predictor: {importance_df.iloc[0]['variable']} ({importance_df.iloc[0]['gain_pct']:.1f}%)")
