# MICS Water Treatment and Risk

## Motivation
This project studies whether household water treatment reduces microbiological
risk in drinking water using household-level data from the Multiple Indicator
Cluster Surveys (MICS). The goal is to estimate causal effects of reported
treatment on binary outcomes capturing high and very high contamination risk.

## Data and Variables
- Treatment: $D_i$ indicates whether the household reports treating its
  drinking water.
- Outcomes: $Y_i$ are binary indicators of elevated microbiological risk
  (high and very high contamination).
- Covariates: $X_i$ includes pre-treatment household, source-water, and
  contextual controls: socioeconomic status (wealth and expenditure rankings),
  education of household head, urban-rural status, country indicators, primary
  drinking water source categories, reported water treatment practices, and
  source-level risk classifications. Categorical variables use ordinal or
  one-hot encodings to preserve ordering where applicable and allow flexible
  nonlinear adjustment.

## Methodology (PLM and IRM)
We use a Double/Debiased ML (DDML) framework grounded in the
Frisch–Waugh–Lovell (FWL) residualization principle to estimate low-dimensional
causal parameters in the presence of high-dimensional confounders.

### Partially Linear Model (PLM)
Let $Y_i$ denote the outcome, $D_i$ the treatment, and
$X_i = (X_{i1}, \ldots, X_{ip})$ the covariates. The PLM is:
$$
\begin{align*}
    Y_i &= D_i \theta_0 + g_0(X_i) + \zeta_i, \quad \mathbb{E}[\zeta_i \vert D_i, X_i] = 0, \\
    D_i &= m_0(X_i) + v_i, \quad \mathbb{E}[v_i \vert X_i] = 0.
\end{align*}
$$

The parameter $\theta_0$ is the average treatment effect (ATE) under a constant
marginal effect assumption.

Estimation follows FWL residualization:
1. Estimate $g_0(X_i)$ and $m_0(X_i)$ using flexible ML methods.
2. Compute residuals:
   $\tilde{Y}_i = Y_i - \hat{g}(X_i)$ and
   $\tilde{D}_i = D_i - \hat{m}(X_i)$.
3. Regress $\tilde{Y}_i$ on $\tilde{D}_i$ via OLS to estimate $\theta_0$.

### Interactive Regression Model (IRM)
We also use the IRM to allow heterogeneous treatment effects:
$$
\begin{align*}
    Y_i &= g_0(D_i, X_i) + u_i, \quad \mathbb{E}[u_i \vert D_i, X_i] = 0, \\
    D_i &= m_0(X_i) + v_i, \quad \mathbb{E}[v_i \vert X_i] = 0.
\end{align*}
$$
which permits the treatment effect to vary with covariates. Under this model we
can recover both the ATE ($\theta_0 = \mathbb{E}[g_0(1,X_i) - g_0(0,X_i)]$) and the average treatment effect on the treated (ATT) (($\theta_0 = \mathbb{E}[g_0(1,X_i) - g_0(0,X_i) \vert D_i = 1]$)),
while the PLM identifies only the ATE.

## Nuisance Estimation and Cross-Fitting
Nuisance functions $g_0(\cdot)$ and $m_0(\cdot)$ are estimated with XGBoost to
capture nonlinear relationships without restrictive parametric assumptions.
Cross-fitting is used to avoid regularization bias and deliver valid inference:
nuisance models are trained on auxiliary folds and evaluated on held-out data,
ensuring Neyman orthogonality of the estimating equation.

## Overfitting Control
We tune XGBoost hyperparameters to control complexity, focusing on:
- maximum tree depth,
- number of boosting iterations,
- learning rate, and
- minimum leaf weight.

While XGBoost supports additional $L1$ and $L2$ penalties, the DDML framework
does not require explicit regularization in the final stage because
orthogonalization and cross-fitting prevent nuisance regularization bias from
contaminating estimation of the causal parameters.
