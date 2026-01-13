# MICS Water Treatment and Risk

## Motivation

This project studies whether household water treatment reduces microbiological
risk in drinking water using household-level data from the Multiple Indicator
Cluster Surveys (MICS). The goal is to estimate causal effects of reported
treatment on binary outcomes capturing high and very high contamination risk.

## Data and Variables

* **Treatment:**
  $D_i$ indicates whether the household reports treating its drinking water.
* **Outcomes:**
  $Y_i$ are binary indicators of elevated microbiological risk (high and very
  high contamination).
* **Covariates:**
  $X_i$ includes pre-treatment household, source-water, and contextual controls:
  socioeconomic status (wealth and expenditure rankings), education of household
  head, urban–rural status, country indicators, primary drinking water source
  categories, reported water treatment practices, and source-level risk
  classifications.

Categorical variables use ordinal or one-hot encodings to preserve ordering
where applicable and allow flexible nonlinear adjustment.

## Methodology (PLM and IRM)

We use a Double/Debiased Machine Learning (DDML) framework grounded in the
Frisch–Waugh–Lovell (FWL) residualization principle to estimate low-dimensional
causal parameters in the presence of high-dimensional confounders.

---

## Partially Linear Model (PLM)

Let $Y_i$ denote the outcome, $D_i$ the treatment, and
$X_i = (X_{i1}, \ldots, X_{ip})$ the covariates. The PLM is:

$$
Y_i = D_i \theta_0 + g_0(X_i) + \zeta_i, \quad \mathbb{E}[\zeta_i \mid D_i, X_i] = 0
$$

$$
D_i = m_0(X_i) + v_i, \quad \mathbb{E}[v_i \mid X_i] = 0
$$

The parameter $\theta_0$ represents the **average treatment effect (ATE)** under
a constant marginal effect assumption.

### Estimation (FWL Residualization)

1. Estimate $g_0(X_i)$ and $m_0(X_i)$ using flexible machine learning methods.
2. Compute residuals:

   * $\tilde{Y}_i = Y_i - \hat{g}(X_i)$
   * $\tilde{D}_i = D_i - \hat{m}(X_i)$
3. Regress $\tilde{Y}_i$ on $\tilde{D}_i$ via OLS to estimate $\theta_0$.

---

## Interactive Regression Model (IRM)

To allow for heterogeneous treatment effects, we also estimate an IRM:

$$
Y_i = g_0(D_i, X_i) + u_i, \quad \mathbb{E}[u_i \mid D_i, X_i] = 0
$$

$$
D_i = m_0(X_i) + v_i, \quad \mathbb{E}[v_i \mid X_i] = 0
$$

This specification permits the treatment effect to vary with covariates.

Under the IRM, we can identify:

* **Average Treatment Effect (ATE):**
  $$
  \theta_0 = \mathbb{E}\big[g_0(1, X_i) - g_0(0, X_i)\big]
  $$

* **Average Treatment Effect on the Treated (ATT):**
  $$
  \theta_0 = \mathbb{E}\big[g_0(1, X_i) - g_0(0, X_i) \mid D_i = 1\big]
  $$

In contrast, the PLM identifies only the ATE.

---

## Nuisance Estimation and Cross-Fitting

Nuisance functions $g_0(\cdot)$ and $m_0(\cdot)$ are estimated using XGBoost to
capture nonlinear relationships without restrictive parametric assumptions.

Cross-fitting is employed to avoid regularization bias and deliver valid
inference. Nuisance models are trained on auxiliary folds and evaluated on
held-out data, ensuring Neyman orthogonality of the estimating equations.

---

## Overfitting Control

We tune XGBoost hyperparameters to control model complexity, focusing on:

* maximum tree depth,
* number of boosting iterations,
* learning rate, and
* minimum leaf weight.

While XGBoost supports additional $L_1$ and $L_2$ penalties, the DDML framework
does not require explicit regularization in the final stage. Orthogonalization
and cross-fitting prevent nuisance regularization bias from contaminating
estimation of the causal parameters.