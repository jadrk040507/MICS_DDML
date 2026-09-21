"""MICS DoubleML analysis written as a fully commented do-file.

Read this file by following the numbered sections. The first sections state
the analysis choices. Shared prediction and inference machinery lives in
``ddml_engine.py``. The ``main()`` function at the end shows the complete
order of the ATE analysis.

Files produced:
    Output/ATE/checkpoints/*.pkl   fitted IRM and APOS models
    Output/ATE/results_*.pkl       pandas result tables
    Output/ATE/table_*.tex         publication tables for LaTeX

To run a quick check with 5% of the sample:
    Change SAMPLED to True, then run python3 ate.py

To run the complete analysis:
    Keep SAMPLED set to False, then run python3 ate.py
"""

from pathlib import Path
import gc
import json
import warnings

import doubleml as dml
import joblib
import numpy as np
import pandas as pd
from joblib import parallel_backend
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from ddml_engine import (
    ConvexClassifier,
    ConvexRegressor,
    build_clustered_sensitivity_framework,
    cluster_robust_framework_inference,
    collect_convex_weights,
    convex_weights,
    format_coefficient,
    score_array_with_named_dimensions,
    sensitivity_params,
    sum_rows_within_psu,
)
from provenance import build_checkpoint_provenance


# Hide repeated convergence messages from penalized regression learners.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# =============================================================================
# FILE MAP — READ THIS FIRST
# =============================================================================
# This script keeps the ATE-specific workflow together. Shared statistical
# machinery lives in ddml_engine.py. Search for "SECTION" to navigate here.
#
#   SECTION 1  Choices, folds, repetitions, paths, and analysis list
#   SECTION 2  Controls, complete-case sample, and model-ready data
#   SECTION 3  Boundary with the shared Super Learner engine
#   SECTION 4  Which candidate learners enter the Super Learner
#   SECTION 5  Reusable checkpoints, folds, fitting, and inference machinery
#   SECTION 6  The four estimations run for each outcome
#   SECTION 7  Main results and LaTeX publication tables
#   SECTION 8  Sensitivity analysis and its step-by-step checkpoints
#   SECTION 9  GATE heterogeneity analysis by E. coli decile and risk group
#   SECTION 10 Complete run order and manifest
#
# If you only want to change or run the analysis, start with SECTIONS 1 and 10.
# SECTIONS 3–5 connect the ATE workflow to shared technical machinery.
#
# CLUSTERED versus UNCLUSTERED, in one glance:
#   - Both use the same prespecified controls from SECTION 2.
#   - Clustered specifications keep each PSU together when creating folds and
#     calculate PSU-cluster-robust standard errors.
#   - Unclustered specifications create ordinary observation-level folds and
#     use the ordinary DoubleML standard errors.
# =============================================================================


# =============================================================================
# SECTION 1 OF 10 — ANALYSIS CHOICES AND PATHS
# Edit here: run size, fold counts, treatment levels, files, and outcomes.
# =============================================================================

SEED = 42
# Change this single value to True for a quick run with 5% of the data.
# Keep it False for the complete analysis.
SAMPLED = False
SAMPLE_FRAC = 0.05
SAMPLE_SEED = SEED

FOLDS = 2 if SAMPLED else 5
REPETITIONS = 1 if SAMPLED else 3
INNER_FOLDS = 2 if SAMPLED else 3

# Parallel work is kept limited and explicit. APOS can fit several treatment
# levels at once. Random Forest and XGBoost can also parallelize each fit.
# Running both levels too widely at the same time can exhaust RAM.
APOS_WORKERS = 1
LEARNER_JOBS = 1 if SAMPLED else 2

TREATMENT_LEVELS = (0, 1, 2, 3, 98)
REPORTED_LEVELS = (0, 1, 2, 3)

PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "Data" / "3. Final"
OUTPUT_DIR = PROJECT / "Output" / "ATE"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
TABLE_DIR = OUTPUT_DIR
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoint filenames include an automatic provenance fingerprint. Existing
# unversioned files remain on disk but are never mistaken for current models.
CHECKPOINT_SCHEMA_VERSION = 2


# Each item contains:
#   1. a short sample name;
#   2. the data file;
#   3. whether this is the under-five sample;
#   4. the outcome variable.
ANALYSIS_SPECS = [
    ("HH", DATA_DIR / "MASTER_MICS_FINAL.dta", False, "SomeRiskHome"),
    ("HH", DATA_DIR / "MASTER_MICS_FINAL.dta", False, "VeryHighRiskHome"),
    ("U5", DATA_DIR / "MASTER_MICS_FINAL_U5.dta", True, "diarrhea"),
]

# This additional analysis is run after the complete main analysis.
# country_cat is stored as numeric Stata codes in the model data.
SELECTED_COUNTRIES = {
    "Dominican Republic": 9,
    "Guyana": 15,
    "Honduras": 16,
    "Malawi": 21,
}


# =============================================================================
# SECTION 2 OF 10 — VARIABLES AND DATA PREPARATION
# Purpose: define controls once and construct the exact rows/columns modeled.
# Key guarantee: clustered and unclustered models use the same substantive
# controls; a PSU code used internally for grouped fitting is never a control.
# =============================================================================

# -----------------------------------------------------------------------------
# 2A. Prespecified controls
# -----------------------------------------------------------------------------

COMMON_CONTROLS = [
    "windex5",
    "urban",
    "WS1_g",
    "wq27_decile",
    "Any_U5",
    "Girls_less_than15",
    "Boys_15or_less",
    "Toilet",
]


def controls_for_sample(child):
    """List the prespecified controls for a household or child sample.

    Parameters
    ----------
    child : bool
        Add child age and sex controls when ``True``.

    Returns
    -------
    list[str]
        Control-variable names in their modeling order.
    """

    controls = list(COMMON_CONTROLS)
    if child:
        controls += ["age", "male"]
    return controls


# -----------------------------------------------------------------------------
# 2B. Complete-case sample rule
# Every downstream specification calls this rule so row selection is explicit.
# -----------------------------------------------------------------------------

def complete_case_sample(
    data,
    outcome,
    treatment,
    child=False,
    cluster=True,
    allowed_levels=None,
    extra_columns=(),
):
    """Select the complete-case observations for one specification.

    This rule is defined once so the main estimation and the heterogeneity
    analysis use exactly the same rows in exactly the same order.

    Parameters
    ----------
    data : pandas.DataFrame
        Loaded MICS observations.
    outcome : str
        Outcome column to require.
    treatment : str
        Treatment column to require.
    child : bool, default=False
        Include child-specific controls.
    cluster : bool, default=True
        Require a nonmissing PSU identifier.
    allowed_levels : iterable or None
        Treatment values retained; ``None`` keeps every observed value.

    extra_columns : iterable[str]
        Metadata retained after filtering; missing metadata never drops model rows.

    Returns
    -------
    pandas.DataFrame
        Filtered rows with a reset 0, ..., N-1 index.
    """

    controls = controls_for_sample(child)

    required = [outcome, treatment, *controls]
    required.append("country_cat")
    if cluster:
        required.append("Cluster_var")

    frame = data[required].copy()
    frame = frame.dropna()
    if allowed_levels is not None:
        frame = frame[frame[treatment].isin(allowed_levels)].copy()
    for column in extra_columns:
        frame[column] = data.loc[frame.index, column]
    return frame.reset_index(drop=True)


# -----------------------------------------------------------------------------
# 2C. Model-ready frame
# Categorical controls are encoded here; outcome and treatment remain named.
# -----------------------------------------------------------------------------

def make_frame(
    data,
    outcome,
    treatment,
    child=False,
    cluster=True,
    allowed_levels=None,
):
    """Build the analysis frame and encoded controls used by DoubleML.

    Parameters
    ----------
    data, outcome, treatment, child, cluster, allowed_levels
        Same inputs as ``complete_case_sample``.

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        Model-ready frame and names of its encoded control columns.
    """

    controls = controls_for_sample(child)
    frame = complete_case_sample(
        data=data,
        outcome=outcome,
        treatment=treatment,
        child=child,
        cluster=cluster,
        allowed_levels=allowed_levels,
    )

    categorical = ["windex5", "WS1_g", "wq27_decile", "Toilet"]
    categorical.append("country_cat")
    numeric = [v for v in controls if v not in categorical]

    controls_frame = frame[numeric + categorical].copy()
    x = pd.get_dummies(
        controls_frame,
        columns=categorical,
        drop_first=True,
        dtype=float,
    )
    x = x.astype(float).reset_index(drop=True)
    # This column only identifies clusters inside the Super Learner. The two
    # convex learner classes remove it before fitting each model. Therefore,
    # Cluster_var is never used as an explanatory variable.
    if cluster:
        x["_cluster_model_code"] = pd.factorize(
            frame["Cluster_var"], sort=True
        )[0].astype(float)

    columns = [outcome, treatment]
    if cluster:
        columns.append("Cluster_var")
    model_frame = pd.concat([frame[columns], x], axis=1)
    return model_frame, list(x.columns)


# =============================================================================
# SECTION 3 OF 10 — HOW THE SUPER LEARNER COMBINES PREDICTIONS
# Purpose: estimate nonnegative learner weights that add to one, separately for
# outcome regression and treatment classification. Normally do not edit here.
# =============================================================================

# -----------------------------------------------------------------------------
# 3A. Shared prediction engine
# -----------------------------------------------------------------------------

# The shared implementation lives in ddml_engine.py and is imported above.
# Keeping the estimand scripts free of prediction-engine details makes the
# ATE/ATT distinction visible without duplicating the Super Learner.


# =============================================================================
# SECTION 4 OF 10 — CANDIDATE LEARNERS USED BY THE SUPER LEARNER
# Edit here only when intentionally changing the nuisance-learning library.
# Full runs use all listed learners; quick 5% runs use the first two only.
# =============================================================================

# -----------------------------------------------------------------------------
# 4A. Learner libraries used in the complete analysis
# -----------------------------------------------------------------------------

REGRESSORS = [
    ("ols", LinearRegression()),
    (
        "lasso",
        Pipeline([
            ("scale", StandardScaler()),
            ("model", LassoCV(cv=3, max_iter=10000, random_state=SEED)),
        ]),
    ),
    (
        "elastic_net",
        Pipeline([
            ("scale", StandardScaler()),
            (
                "model",
                ElasticNetCV(
                    cv=3,
                    l1_ratio=(0.25, 0.5, 0.75),
                    max_iter=10000,
                    random_state=SEED,
                ),
            ),
        ]),
    ),
    (
        "random_forest",
        RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=LEARNER_JOBS,
        ),
    ),
    (
        "xgboost",
        XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=SEED,
            n_jobs=LEARNER_JOBS,
            eval_metric="rmse",
            verbosity=0,
        ),
    ),
]

CLASSIFIERS = [
    ("logit", LogisticRegression(max_iter=2000)),
    (
        "lasso",
        LogisticRegressionCV(
            cv=3,
            l1_ratios=(1,),
            solver="liblinear",
            max_iter=5000,
            scoring="neg_log_loss",
            random_state=SEED,
            use_legacy_attributes=False,
        ),
    ),
    (
        "elastic_net",
        LogisticRegressionCV(
            cv=3,
            solver="saga",
            l1_ratios=(0.5,),
            max_iter=10000,
            scoring="neg_log_loss",
            random_state=SEED,
            use_legacy_attributes=False,
        ),
    ),
    (
        "random_forest",
        RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=LEARNER_JOBS,
        ),
    ),
    (
        "xgboost",
        XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=SEED,
            n_jobs=LEARNER_JOBS,
            eval_metric="logloss",
            verbosity=0,
        ),
    ),
]

# -----------------------------------------------------------------------------
# 4B. Reduced learner library used only for a quick workflow check
# -----------------------------------------------------------------------------

# The sampled run checks the workflow; it is not a model comparison. Two
# learners are enough to test the convex combination in a few minutes.
if SAMPLED:
    REGRESSORS = REGRESSORS[:2]
    CLASSIFIERS = CLASSIFIERS[:2]


def checkpoint_provenance():
    """Describe every input that determines checkpoint compatibility."""

    files = {
        "analysis_script": Path(__file__),
        "shared_engine": Path(__file__).with_name("ddml_engine.py"),
        "environment_lock": PROJECT / "uv.lock",
    }
    for dataset, data_path, _, _ in ANALYSIS_SPECS:
        files[f"data_{dataset}"] = data_path
    settings = {
        "estimand": "ATE",
        "seed": SEED,
        "sampled": SAMPLED,
        "sample_fraction": SAMPLE_FRAC if SAMPLED else None,
        "folds": FOLDS,
        "repetitions": REPETITIONS,
        "inner_folds": INNER_FOLDS,
        "treatment_levels": list(TREATMENT_LEVELS),
        "reported_levels": list(REPORTED_LEVELS),
        "outcome_learners": [name for name, _ in REGRESSORS],
        "treatment_learners": [name for name, _ in CLASSIFIERS],
    }
    return build_checkpoint_provenance(
        CHECKPOINT_SCHEMA_VERSION,
        files,
        settings,
    )


# =============================================================================
# SECTION 5 OF 10 — REUSABLE ANALYSIS BUILDING BLOCKS
# Purpose: checkpoints, clustered inference, GATE projection, model fitting,
# and folds used by the readable workflow in SECTION 6. Normally do not edit.
# =============================================================================

# -----------------------------------------------------------------------------
# 5A. Checkpoint files and resume logic
# One place controls full versus sample filenames and prevents overwriting.
# -----------------------------------------------------------------------------

class CheckpointStore:
    """Read and write checkpoints for either a full or quick-sample run.

    This class is the single place that knows how checkpoint filenames are
    constructed, how fitted models are made smaller before saving, and how
    load/save activity is reported to the person running the script.

    Parameters
    ----------
    quick_sample : bool
        If ``True``, add ``_sample05`` to every checkpoint name. This prevents
        a quick diagnostic run from loading or overwriting full-run models.
    """

    def __init__(self, quick_sample, fingerprint=None):
        """Remember whether this store belongs to a full or sample run.

        Parameters
        ----------
        quick_sample : bool
            Select sample-tagged filenames when ``True``.
        """

        self.quick_sample = bool(quick_sample)
        if fingerprint is None:
            fingerprint, provenance = checkpoint_provenance()
        else:
            provenance = None
        self.fingerprint = str(fingerprint)
        self.provenance = provenance

    def path(self, name):
        """Return the filesystem path for a logical checkpoint name.

        Parameters
        ----------
        name : str
            Human-readable model or analysis name without ``.pkl``.

        Returns
        -------
        pathlib.Path
            Full checkpoint path, including version and sample tags.
        """

        sample_tag = (
            f"_sample{int(SAMPLE_FRAC * 100):02d}"
            if self.quick_sample
            else ""
        )
        provenance_tag = f"_{self.fingerprint[:12]}"
        return CHECKPOINT_DIR / f"{name}{provenance_tag}{sample_tag}.pkl"

    def exists(self, name):
        """Check whether a named checkpoint is already on disk.

        Parameters
        ----------
        name : str
            Logical checkpoint name.

        Returns
        -------
        bool
            ``True`` when the matching full/sample file exists.
        """

        return self.path(name).exists()

    def load(self, name):
        """Load one checkpoint and announce the reused filename.

        Parameters
        ----------
        name : str
            Logical checkpoint name.

        Returns
        -------
        object
            Deserialized model or sensitivity rows.
        """

        path = self.path(name)
        print(f"Loading checkpoint: {path.name}", flush=True)
        return joblib.load(path)

    def save(self, name, value, fitted_model=False):
        """Save one checkpoint and return the same value.

        Parameters
        ----------
        name : str
            Logical checkpoint name without a suffix or extension.
        value : object
            Python object to serialize with joblib.
        fitted_model : bool, default=False
            Set to ``True`` for fitted IRM/APOS checkpoints. IRM learner
            weights are retained while bulky fitted nuisance models are
            removed. Plain sensitivity rows are saved unchanged.

        Returns
        -------
        object
            The unmodified input ``value``, allowing save calls in workflows.
        """

        if fitted_model:
            # Clustered APOS is stored as {"model": ..., "cluster_se": ...};
            # other model checkpoints contain the DoubleML model directly.
            model = value["model"] if isinstance(value, dict) else value
            if not isinstance(model, dml.DoubleMLAPOS):
                # Preserve the interpretable Super Learner weights before
                # releasing fitted nuisance learners that make IRM very large.
                model.convex_weights = collect_convex_weights(model)
                model._models = None

        path = self.path(name)
        joblib.dump(value, path, compress=3)
        print(f"Saved checkpoint: {path.name}", flush=True)
        return value


class OutcomeCheckpointBundle:
    """Expose saved models and lightweight table frames through one mapping.

    Parameters
    ----------
    paths : dict[str, pathlib.Path]
        Checkpoint paths for ``irm_cluster``, ``irm_no_cluster``,
        ``apos_cluster``, and ``apos_no_cluster``.
    table_frames : dict[str, pandas.DataFrame]
        Small in-memory frames used for sample statistics and PSU identifiers.

    Notes
    -----
    Accessing ``bundle[key]`` loads that model only for the current task. This
    avoids keeping several large DoubleML models in memory simultaneously.
    """

    def __init__(self, paths, table_frames):
        """Store checkpoint paths and already-small table data.

        Parameters
        ----------
        paths, table_frames : dict
            Model paths and in-memory descriptive frames documented above.
        """

        self.paths = paths
        self.table_frames = table_frames

    def __getitem__(self, key):
        """Return a table frame immediately or load the requested model.

        Parameters
        ----------
        key : str
            Model or table-frame name.

        Returns
        -------
        object
            DataFrame or fitted DoubleML model associated with ``key``.
        """

        if key in self.table_frames:
            return self.table_frames[key]

        saved_object = joblib.load(self.paths[key])
        if key == "apos_cluster":
            return saved_object["model"]
        return saved_object


# -----------------------------------------------------------------------------
# 5B. Result filenames and saved Super Learner weights
# These helpers organize outputs; they do not estimate coefficients or SEs.
# -----------------------------------------------------------------------------

def result_pickle_path(name, quick_sample, file_suffix=""):
    """Build a result-pickle path without touching the filesystem.

    Parameters
    ----------
    name : str
        Base result name, such as ``results_apos``.
    quick_sample : bool
        Add ``_sample05`` when ``True``.
    file_suffix : str, default=""
        Optional analysis suffix, such as ``_selected_countries``.

    Returns
    -------
    pathlib.Path
        Output path. Full and sample results can never share this path.
    """

    sample_tag = f"_sample{int(SAMPLE_FRAC * 100):02d}" if quick_sample else ""
    return OUTPUT_DIR / f"{name}{file_suffix}{sample_tag}.pkl"




# -----------------------------------------------------------------------------
# 5C. PSU-clustered sandwich SEs and repeated-cross-fitting inference
# Statistical order: observation scores -> PSU sums -> SE for each repetition
# -> one reported SE/p-value/confidence interval across repetitions.
# -----------------------------------------------------------------------------







# -----------------------------------------------------------------------------
# 5D. Clustered score framework used by the sensitivity analysis
# This repackages already estimated scores by PSU; it does not refit the model.
# -----------------------------------------------------------------------------





# -----------------------------------------------------------------------------
# 5E. GATE projection from an already fitted treatment contrast
# The fitted model is reused for projections onto prespecified groups.
# -----------------------------------------------------------------------------

def estimate_gate_from_contrast(
    contrast,
    treatment_level,
    effect_index,
    group_values,
    cluster_ids,
    group_labels,
    level=0.95,
    n_rep_boot=500,
):
    """Project a fitted orthogonal signal onto prespecified GATE groups.

    Parameters
    ----------
    contrast : DoubleMLFramework
        Fitted IRM effect or APOS treatment contrast.
    treatment_level : object
        Treatment label copied into the returned rows.
    effect_index : int
        Contrast column projected from the J effects in ``contrast``.
    group_values : array-like
        Numeric group code for every observation.
    cluster_ids : array-like or None
        PSU identifiers for clustered covariance; ``None`` requests HC0.
    group_labels : dict[str, str]
        Mapping from numeric group strings to readable labels.
    level : float, default=0.95
        Pointwise and joint confidence level.
    n_rep_boot : int, default=500
        Gaussian draws used for joint intervals.

    Returns
    -------
    pandas.DataFrame
        One row per observed group with coefficient, SE, p-value, intervals,
        sample counts, and median BLP R-squared.
    """

    # Use the same N observations x J effects x R repetitions layout as the
    # clustered-inference section above.
    scaled_influence_scores = score_array_with_named_dimensions(
        contrast.scaled_psi
    )
    # DoubleML stores scaled_psi = theta - orthogonal_signal for these linear
    # scores. GATEs must project the orthogonal signal itself, as the native
    # IRM gate() method does, rather than project the centered influence term.
    all_thetas = np.asarray(contrast.all_thetas, dtype=float)
    # all_thetas is J x R. np.newaxis inserts an observation dimension, making
    # it 1 x J x R; NumPy then broadcasts each theta across all N observations
    # when subtracting the N x J x R influence-score array.
    orthogonal_signals = (
        all_thetas[np.newaxis, :, :] - scaled_influence_scores
    )

    values = pd.Series(group_values).reset_index(drop=True)
    valid_observation = values.notna().to_numpy()
    values = values.loc[valid_observation].reset_index(drop=True)
    orthogonal_signals = orthogonal_signals[valid_observation, :, :]

    # Normalize numeric group codes to strings before creating indicators.
    numeric_values = pd.to_numeric(values)
    values = numeric_values.astype(int).astype("string")

    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)[valid_observation]

    group_indicators = pd.get_dummies(
        values.astype("string"),
        prefix="group",
        dtype=float,
    )
    observed_groups = group_indicators.sum(axis=0) > 0
    group_indicators = group_indicators.loc[:, observed_groups]

    blp = dml.DoubleMLBLP(
        orthogonal_signals[:, effect_index, :],
        basis=group_indicators,
        is_gate=True,
    )
    if cluster_ids is None:
        blp.fit(cov_type="HC0")
    else:
        blp.fit(
            cov_type="cluster",
            cov_kwds={"groups": cluster_ids},
        )

    pointwise_interval = blp.confint(joint=False, level=float(level))
    joint_interval = blp.confint(
        joint=True,
        level=float(level),
        n_rep_boot=int(n_rep_boot),
    )

    labels = group_labels
    r_squared_by_repetition = [
        float(fitted_model.rsquared)
        for fitted_model in blp._blp_model
    ]
    median_r_squared = float(np.median(r_squared_by_repetition))

    rows = []
    for indicator_name in group_indicators.columns:
        raw_value = indicator_name.removeprefix("group_")
        readable_label = labels[raw_value]

        group_mask = values.astype("string").eq(raw_value).to_numpy()
        if cluster_ids is None:
            number_of_group_clusters = np.nan
        else:
            number_of_group_clusters = pd.Series(
                cluster_ids[group_mask]
            ).nunique()

        result = blp.summary.loc[indicator_name]
        rows.append({
            "treatment_level": treatment_level,
            "group_value": raw_value,
            "group_label": readable_label,
            "n": int(group_indicators[indicator_name].sum()),
            "n_psu": (
                int(number_of_group_clusters)
                if pd.notna(number_of_group_clusters)
                else np.nan
            ),
            "r2": median_r_squared,
            "coef": float(result["coef"]),
            "se": float(result["std err"]),
            "pval": float(result["P>|t|"]),
            "ci_lower": float(pointwise_interval.loc[indicator_name].iloc[0]),
            "ci_upper": float(pointwise_interval.loc[indicator_name].iloc[-1]),
            "ci_lower_joint": float(
                joint_interval.loc[indicator_name].iloc[0]
            ),
            "ci_upper_joint": float(
                joint_interval.loc[indicator_name].iloc[-1]
            ),
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 5F. Basic DoubleML fitting functions
# IRM handles binary any-treatment; APOS handles multiple treatment levels.
# -----------------------------------------------------------------------------

def fit_irm(frame, x_columns, outcome, treatment, clustered):
    """Fit one binary-treatment IRM specification.

    Parameters
    ----------
    frame : pandas.DataFrame
        Model-ready observations.
    x_columns : list[str]
        Encoded controls used by the nuisance learners.
    outcome, treatment : str
        Outcome and binary-treatment column names.
    clustered : bool
        Use PSU-grouped folds and native clustered inference when ``True``.

    Returns
    -------
    DoubleMLIRM
        Fitted IRM with explicit reproducible sample splitting.
    """

    data = dml.DoubleMLData(
        frame,
        y_col=outcome,
        d_cols=treatment,
        x_cols=x_columns,
        cluster_cols="Cluster_var" if clustered else None,
    )
    model = dml.DoubleMLIRM(
        data,
        ConvexRegressor(
            REGRESSORS,
            random_state=SEED,
            group_column=-1 if clustered else None,
            inner_folds=INNER_FOLDS,
        ),
        ConvexClassifier(
            CLASSIFIERS,
            random_state=SEED,
            group_column=-1 if clustered else None,
            inner_folds=INNER_FOLDS,
        ),
        n_folds=FOLDS,
        n_rep=REPETITIONS,
        draw_sample_splitting=False,
    )
    if clustered:
        splits = make_cluster_splits(frame, treatment)
        cluster_splits = make_cluster_split_metadata(frame, splits)
        model.set_sample_splitting(splits, cluster_splits)
    else:
        model.set_sample_splitting(make_iid_splits(frame, treatment))
    return model.fit(
        n_jobs_cv=1,
        store_predictions=False,
        store_models=True,
    )


def fit_apos(frame, x_columns, outcome):
    """Fit APOS with ordinary observation-level folds.

    Parameters
    ----------
    frame : pandas.DataFrame
        Model-ready observations.
    x_columns : list[str]
        Encoded nuisance-model controls.
    outcome : str
        Outcome column name.

    Returns
    -------
    DoubleMLAPOS
        Fitted potential-outcome models for levels 0, 1, 2, 3, and 98.
    """

    data = dml.DoubleMLData(
        frame,
        y_col=outcome,
        d_cols="WQ15_g",
        x_cols=x_columns,
    )
    model = dml.DoubleMLAPOS(
        data,
        ConvexRegressor(
            REGRESSORS,
            random_state=SEED,
            group_column=None,
            inner_folds=INNER_FOLDS,
        ),
        ConvexClassifier(
            CLASSIFIERS,
            random_state=SEED,
            group_column=None,
            inner_folds=INNER_FOLDS,
        ),
        treatment_levels=list(TREATMENT_LEVELS),
        n_folds=FOLDS,
        n_rep=REPETITIONS,
        draw_sample_splitting=False,
    )
    model.set_sample_splitting(make_iid_splits(frame, "WQ15_g"))
    with parallel_backend("threading"):
        return model.fit(
            n_jobs_models=APOS_WORKERS,
            n_jobs_cv=1,
            store_predictions=False,
            store_models=False,
        )


# -----------------------------------------------------------------------------
# 5G. Build and validate cross-fitting folds
# Audit here when checking PSU isolation, FOLDS, or REPETITIONS.
# -----------------------------------------------------------------------------

def _validate_splits(splits, target, groups=None):
    """Reject invalid external or inner sample splits before estimation.

    Parameters
    ----------
    splits : list
        Repetitions containing ``(train_indices, test_indices)`` folds.
    target : array-like
        Treatment values used to check support.
    groups : array-like or None
        PSU identifiers used to check train/test isolation.

    Returns
    -------
    None
        Raises ``ValueError`` when coverage, support, or isolation fails.
    """

    expected_levels = np.unique(target)
    expected_rows = np.arange(len(target))
    for repetition in splits:
        tested_rows = np.concatenate([test for _, test in repetition])
        if not np.array_equal(np.sort(tested_rows), expected_rows):
            raise ValueError("Test folds do not partition the analysis sample.")
        for train, test in repetition:
            if not np.array_equal(np.unique(target[train]), expected_levels):
                raise ValueError("A training fold is missing a treatment level.")
            if not np.array_equal(np.unique(target[test]), expected_levels):
                raise ValueError("A test fold is missing a treatment level.")
            if groups is not None:
                overlap = np.intersect1d(groups[train], groups[test])
                if overlap.size:
                    raise ValueError("A sampling cluster appears in train and test.")


def make_iid_splits(frame, treatment):
    """Build reproducible observation-level cross-fitting folds.

    Parameters
    ----------
    frame : pandas.DataFrame
        Analysis observations.
    treatment : str
        Column whose treatment levels are stratified.

    Returns
    -------
    list
        R repetitions of F train/test index pairs.
    """

    target = frame[treatment].to_numpy()
    all_repetitions = []
    for repetition in range(REPETITIONS):
        splitter = StratifiedKFold(
            n_splits=FOLDS,
            shuffle=True,
            random_state=SEED + repetition,
        )
        splits = list(splitter.split(np.zeros(len(frame)), target))
        all_repetitions.append(splits)

    _validate_splits(all_repetitions, target)
    return all_repetitions


def make_cluster_splits(frame, treatment):
    """Build reproducible folds that keep every PSU together.

    Parameters
    ----------
    frame : pandas.DataFrame
        Analysis observations containing ``Cluster_var``.
    treatment : str
        Column whose treatment levels are balanced across folds.

    Returns
    -------
    list
        R repetitions of F train/test index pairs with no PSU overlap.
    """

    groups = frame["Cluster_var"].to_numpy()
    target = frame[treatment].to_numpy()
    all_repetitions = []

    for repetition in range(REPETITIONS):
        splitter = StratifiedGroupKFold(
            n_splits=FOLDS,
            shuffle=True,
            random_state=SEED + repetition,
        )
        splits = list(splitter.split(np.zeros(len(frame)), target, groups))
        all_repetitions.append(splits)

    _validate_splits(all_repetitions, target, groups=groups)
    return all_repetitions


def make_cluster_split_metadata(frame, splits):
    """Translate observation folds into DoubleML's PSU-fold metadata.

    Parameters
    ----------
    frame : pandas.DataFrame
        Analysis frame containing ``Cluster_var``.
    splits : list
        Observation-index folds from ``make_cluster_splits``.

    Returns
    -------
    list
        Matching train/test PSU-label arrays for every fold and repetition.
    """

    groups = frame["Cluster_var"].to_numpy()
    return [
        [
            ([np.unique(groups[train])], [np.unique(groups[test])])
            for train, test in repetition
        ]
        for repetition in splits
    ]


# -----------------------------------------------------------------------------
# 5H. Fit clustered APOS and standardize result summaries
# Clustered inference is inserted into the APOS summary before tables are made.
# -----------------------------------------------------------------------------

def fit_apos_clustered(frame, x_columns, outcome, splits):
    """Fit APOS using folds that keep each PSU together.

    Clustered inference is calculated later from the saved model's contrast,
    in ``cluster_robust_framework_inference``. Keeping fitting and inference
    as visible consecutive steps makes the statistical workflow easier to
    audit. The dictionary return shape remains compatible with existing APOS
    checkpoints, which may also contain an older unused ``cluster_se`` key.

    Parameters
    ----------
    frame : pandas.DataFrame
        Model-ready observations including the temporary PSU fold column.
    x_columns : list[str]
        Encoded nuisance-model controls.
    outcome : str
        Outcome column name.
    splits : list
        Prevalidated PSU-grouped train/test folds.

    Returns
    -------
    dict
        ``{"model": fitted_doubleml_apos}``, matching checkpoint format.
    """

    data = dml.DoubleMLData(
        frame,
        y_col=outcome,
        d_cols="WQ15_g",
        x_cols=x_columns,
    )
    model = dml.DoubleMLAPOS(
        data,
        ConvexRegressor(
            REGRESSORS,
            random_state=SEED,
            group_column=-1,
            inner_folds=INNER_FOLDS,
        ),
        ConvexClassifier(
            CLASSIFIERS,
            random_state=SEED,
            group_column=-1,
            inner_folds=INNER_FOLDS,
        ),
        treatment_levels=list(TREATMENT_LEVELS),
        n_folds=FOLDS,
        n_rep=REPETITIONS,
        draw_sample_splitting=False,
    )
    model.set_sample_splitting(splits)
    with parallel_backend("threading"):
        model = model.fit(
            n_jobs_models=APOS_WORKERS,
            n_jobs_cv=1,
            store_predictions=False,
            store_models=False,
        )
    return {"model": model}


def _add_metadata(summary, dataset, outcome, method, specification, n, clusters):
    """Attach analysis labels and sample counts to a model summary.

    Parameters
    ----------
    summary : pandas.DataFrame
        DoubleML coefficient summary.
    dataset, outcome, method, specification : str
        Labels identifying the analysis column.
    n : int
        Number of observations.
    clusters : int or None
        Number of PSUs for clustered specifications.

    Returns
    -------
    pandas.DataFrame
        Copy of the summary with identifying metadata columns.
    """

    table = summary.reset_index()
    table.insert(0, "dataset", dataset)
    table.insert(1, "outcome", outcome)
    table.insert(2, "method", method)
    table.insert(3, "specification", specification)
    table["n"] = n
    table["clusters"] = clusters
    return table


def summary_with_clustered_inference(summary, clustered_inference):
    """Replace ordinary APOS inference with repeated clustered inference.

    This copy is saved in ``results_apos.pkl``. The result file and LaTeX
    tables therefore report the same standard errors, t statistics, p-values,
    and confidence intervals.

    Parameters
    ----------
    summary : pandas.DataFrame
        Ordinary DoubleML contrast summary providing coefficients and rows.
    clustered_inference : dict[str, numpy.ndarray]
        Output from ``cluster_robust_framework_inference``.

    Returns
    -------
    pandas.DataFrame
        Summary with clustered SEs, p-values, and confidence intervals.
    """

    adjusted = summary.copy()
    standard_errors = np.asarray(clustered_inference["se"], dtype=float)

    coefficients = adjusted["coef"].to_numpy(dtype=float)
    t_statistics = coefficients / standard_errors

    adjusted["std err"] = standard_errors
    adjusted["t"] = t_statistics
    adjusted["P>|t|"] = np.asarray(
        clustered_inference["pval"],
        dtype=float,
    )
    adjusted["2.5 %"] = np.asarray(
        clustered_inference["ci_lower"],
        dtype=float,
    )
    adjusted["97.5 %"] = np.asarray(
        clustered_inference["ci_upper"],
        dtype=float,
    )
    return adjusted


# =============================================================================
# SECTION 6 OF 10 — LOAD DATA AND ESTIMATE IRM/APOS
# This is the main economist-facing estimation workflow. For every outcome it
# visibly runs: IRM clustered, IRM unclustered, APOS clustered, APOS unclustered.
# =============================================================================

# -----------------------------------------------------------------------------
# 6A. Load one outcome and optionally draw the reproducible 5% sample
# -----------------------------------------------------------------------------

def load_analysis_data(
    path,
    outcome,
    child,
    country_codes,
    quick_sample,
):
    """Load and optionally sample the columns required for one outcome.

    Parameters
    ----------
    path : pathlib.Path
        Stata data file.
    outcome : str
        Outcome column to load.
    child : bool
        Include child age and sex columns.
    country_codes : iterable or None
        Country codes retained before sampling; ``None`` keeps all countries.
    quick_sample : bool
        Draw the reproducible 5% diagnostic sample when ``True``.

    Returns
    -------
    pandas.DataFrame
        Loaded observations with only columns required downstream.
    """

    columns = set(COMMON_CONTROLS)
    columns.update({
        outcome,
        "water_treatment",
        "WQ15_g",
        "country_cat",
        "Cluster_var",
        "RiskSource",
    })
    if child:
        columns.update({"age", "male"})

    # convert_categoricals=False preserves numeric Stata codes for treatment
    # levels. Loading selected columns keeps unrelated survey variables out of
    # memory; only one outcome's data frame is alive at a time.
    data = pd.read_stata(
        path,
        columns=sorted(columns),
        convert_categoricals=False,
    )
    if country_codes is not None:
        data = data[data["country_cat"].isin(country_codes)].copy()
        print(
            f"Selected countries: {len(data):,} observations before sampling"
        )
    if quick_sample:
        data = data.sample(
            frac=SAMPLE_FRAC,
            random_state=SAMPLE_SEED,
        ).reset_index(drop=True)
        print(
            f"Quick sample: {len(data):,} observations "
            f"({SAMPLE_FRAC:.0%} of loaded data)"
        )
    return data


# -----------------------------------------------------------------------------
# 6B. Estimate or load all four specifications for one outcome
# Each lettered block has its own checkpoint and releases memory after use.
# -----------------------------------------------------------------------------

def estimate_one_outcome(
    dataset,
    data_path,
    child,
    outcome,
    country_codes,
    checkpoint_prefix,
    quick_sample,
):
    """Fit the four specifications for one outcome.

    The order is deliberately explicit:
        A. IRM with clustered folds.
        B. IRM with ordinary folds.
        C. APOS with clustered folds.
        D. APOS with ordinary folds.

    Parameters
    ----------
    dataset : str
        Short sample label, ``HH`` or ``U5``.
    data_path : pathlib.Path
        Stata input file.
    child : bool
        Select household or under-five controls.
    outcome : str
        Outcome estimated in all four specifications.
    country_codes : iterable or None
        Optional selected-country restriction.
    checkpoint_prefix : str
        Prefix distinguishing main and selected-country checkpoints.
    quick_sample : bool
        Use sample-tagged checkpoints and 5% data when ``True``.

    Returns
    -------
    dict
        Result tables, Super Learner weights, and an on-demand model bundle.
    """

    checkpoints = CheckpointStore(quick_sample)
    print(f"\nPreparing {dataset} — {outcome}", flush=True)
    print("Step 1 of 4: IRM with clustered folds", flush=True)
    data = load_analysis_data(
        data_path,
        outcome,
        child,
        country_codes,
        quick_sample,
    )

    irm_result_tables = []
    apos_result_tables = []
    weight_rows = []

    # --------------------------------------------------------
    # A. IRM with clustered folds and clustered standard errors
    # --------------------------------------------------------
    irm_cluster_frame, irm_cluster_x = make_frame(
        data=data,
        outcome=outcome,
        treatment="water_treatment",
        child=child,
        cluster=True,
        allowed_levels=(0, 1),
    )
    irm_cluster_name = (
        f"{checkpoint_prefix}{dataset}_{outcome}_IRM_clustered"
    )

    if checkpoints.exists(irm_cluster_name):
        irm_cluster_model = checkpoints.load(irm_cluster_name)
    else:
        print(f"Estimating: {irm_cluster_name}", flush=True)
        irm_cluster_model = fit_irm(
            frame=irm_cluster_frame,
            x_columns=irm_cluster_x,
            outcome=outcome,
            treatment="water_treatment",
            clustered=True,
        )
        checkpoints.save(
            irm_cluster_name,
            irm_cluster_model,
            fitted_model=True,
        )

    number_of_clusters = irm_cluster_frame["Cluster_var"].nunique()
    irm_result_tables.append(_add_metadata(
        summary=irm_cluster_model.summary,
        dataset=dataset,
        outcome=outcome,
        method="IRM",
        specification="clustered",
        n=len(irm_cluster_frame),
        clusters=number_of_clusters,
    ))
    for nuisance, learner_weights in irm_cluster_model.convex_weights.items():
        for learner, weight in learner_weights.items():
            weight_rows.append({
                "dataset": dataset,
                "outcome": outcome,
                "model": irm_cluster_name,
                "nuisance": nuisance,
                "learner": learner,
                "weight": float(weight),
            })
    irm_table_frame_cluster = irm_cluster_frame[
        [outcome, "water_treatment", "Cluster_var"]
    ].copy()

    irm_cluster_model = None
    irm_cluster_frame = None
    irm_cluster_x = None
    gc.collect()

    # --------------------------------------------------------
    # B. IRM with ordinary observation-level folds
    # --------------------------------------------------------
    print("Step 2 of 4: IRM with ordinary folds", flush=True)
    irm_iid_frame, irm_iid_x = make_frame(
        data=data,
        outcome=outcome,
        treatment="water_treatment",
        child=child,
        cluster=False,
        allowed_levels=(0, 1),
    )
    irm_iid_name = f"{checkpoint_prefix}{dataset}_{outcome}_IRM_iid"

    if checkpoints.exists(irm_iid_name):
        irm_iid_model = checkpoints.load(irm_iid_name)
    else:
        print(f"Estimating: {irm_iid_name}", flush=True)
        irm_iid_model = fit_irm(
            frame=irm_iid_frame,
            x_columns=irm_iid_x,
            outcome=outcome,
            treatment="water_treatment",
            clustered=False,
        )
        checkpoints.save(
            irm_iid_name,
            irm_iid_model,
            fitted_model=True,
        )

    irm_result_tables.append(_add_metadata(
        summary=irm_iid_model.summary,
        dataset=dataset,
        outcome=outcome,
        method="IRM",
        specification="iid",
        n=len(irm_iid_frame),
        clusters=None,
    ))
    for nuisance, learner_weights in irm_iid_model.convex_weights.items():
        for learner, weight in learner_weights.items():
            weight_rows.append({
                "dataset": dataset,
                "outcome": outcome,
                "model": irm_iid_name,
                "nuisance": nuisance,
                "learner": learner,
                "weight": float(weight),
            })
    irm_table_frame_iid = irm_iid_frame[
        [outcome, "water_treatment"]
    ].copy()

    irm_iid_model = None
    irm_iid_frame = None
    irm_iid_x = None
    gc.collect()

    # --------------------------------------------------------
    # C. APOS with clustered folds and clustered standard errors
    # --------------------------------------------------------
    # DoubleMLAPOS does not accept cluster_cols in the project version. Each
    # cluster is therefore kept inside one fold, and one-way cluster-robust
    # standard errors are calculated after fitting the model.
    print("Step 3 of 4: APOS with clustered folds", flush=True)
    apos_cluster_frame, apos_cluster_x = make_frame(
        data=data,
        outcome=outcome,
        treatment="WQ15_g",
        child=child,
        cluster=True,
        allowed_levels=TREATMENT_LEVELS,
    )
    apos_cluster_splits = make_cluster_splits(
        apos_cluster_frame,
        treatment="WQ15_g",
    )
    apos_cluster_name = f"{checkpoint_prefix}{dataset}_{outcome}_APOS"
    apos_cluster_n = len(apos_cluster_frame)
    apos_cluster_count = apos_cluster_frame["Cluster_var"].nunique()
    apos_table_frame_cluster = apos_cluster_frame[
        [outcome, "WQ15_g", "Cluster_var"]
    ].copy()

    apos_cluster_exists = checkpoints.exists(apos_cluster_name)
    if apos_cluster_exists:
        # Release the large data frame before opening an APOS checkpoint.
        data = None
        apos_cluster_frame = None
        apos_cluster_x = None
        apos_cluster_splits = None
        gc.collect()
        apos_saved_object = checkpoints.load(apos_cluster_name)
    else:
        print(f"Estimating: {apos_cluster_name}", flush=True)
        apos_saved_object = fit_apos_clustered(
            frame=apos_cluster_frame,
            x_columns=apos_cluster_x,
            outcome=outcome,
            splits=apos_cluster_splits,
        )
        checkpoints.save(
            apos_cluster_name,
            apos_saved_object,
            fitted_model=True,
        )

    apos_cluster_model = apos_saved_object["model"]
    apos_cluster_contrast = apos_cluster_model.causal_contrast(
        reference_levels=[0]
    )
    apos_cluster_inference = cluster_robust_framework_inference(
        apos_cluster_contrast,
        apos_table_frame_cluster["Cluster_var"].to_numpy(),
    )
    apos_cluster_summary = apos_cluster_contrast.summary
    apos_cluster_summary = summary_with_clustered_inference(
        apos_cluster_summary,
        apos_cluster_inference,
    )
    apos_result_tables.append(_add_metadata(
        summary=apos_cluster_summary,
        dataset=dataset,
        outcome=outcome,
        method="APOS",
        specification="clustered_folds",
        n=apos_cluster_n,
        clusters=apos_cluster_count,
    ))

    data = None
    apos_cluster_frame = None
    apos_cluster_x = None
    apos_cluster_splits = None
    apos_saved_object = None
    apos_cluster_model = None
    apos_cluster_summary = None
    gc.collect()

    # --------------------------------------------------------
    # D. APOS with ordinary observation-level folds
    # --------------------------------------------------------
    print("Step 4 of 4: APOS with ordinary folds", flush=True)
    data = load_analysis_data(
        data_path,
        outcome,
        child,
        country_codes,
        quick_sample,
    )
    apos_iid_frame, apos_iid_x = make_frame(
        data=data,
        outcome=outcome,
        treatment="WQ15_g",
        child=child,
        cluster=False,
        allowed_levels=TREATMENT_LEVELS,
    )
    apos_iid_name = f"{checkpoint_prefix}{dataset}_{outcome}_APOS_iid"
    apos_iid_n = len(apos_iid_frame)
    apos_table_frame_iid = apos_iid_frame[[outcome, "WQ15_g"]].copy()

    apos_iid_exists = checkpoints.exists(apos_iid_name)
    if apos_iid_exists:
        data = None
        apos_iid_frame = None
        apos_iid_x = None
        gc.collect()
        apos_iid_model = checkpoints.load(apos_iid_name)
    else:
        print(f"Estimating: {apos_iid_name}", flush=True)
        apos_iid_model = fit_apos(
            frame=apos_iid_frame,
            x_columns=apos_iid_x,
            outcome=outcome,
        )
        checkpoints.save(
            apos_iid_name,
            apos_iid_model,
            fitted_model=True,
        )

    apos_iid_summary = apos_iid_model.causal_contrast(
        reference_levels=[0]
    ).summary
    apos_result_tables.append(_add_metadata(
        summary=apos_iid_summary,
        dataset=dataset,
        outcome=outcome,
        method="APOS",
        specification="iid",
        n=apos_iid_n,
        clusters=None,
    ))

    data = None
    apos_iid_frame = None
    apos_iid_x = None
    apos_iid_model = None
    apos_iid_summary = None
    gc.collect()

    # This bundle stores paths, not open models. Each later stage loads only
    # the checkpoint needed at that moment.
    checkpoint_paths = {
        "irm_cluster": checkpoints.path(irm_cluster_name),
        "irm_no_cluster": checkpoints.path(irm_iid_name),
        "apos_cluster": checkpoints.path(apos_cluster_name),
        "apos_no_cluster": checkpoints.path(apos_iid_name),
    }
    table_frames = {
        "irm_frame_cluster": irm_table_frame_cluster,
        "irm_frame_no_cluster": irm_table_frame_iid,
        "apos_frame_cluster": apos_table_frame_cluster,
        "apos_frame_no_cluster": apos_table_frame_iid,
    }
    bundle = OutcomeCheckpointBundle(checkpoint_paths, table_frames)

    print(
        f"Completed {dataset} — {outcome}: "
        "IRM clustered, IRM iid, APOS clustered, APOS iid",
        flush=True,
    )
    return {
        "irm_results": irm_result_tables,
        "apos_results": apos_result_tables,
        "weights": weight_rows,
        "bundle": bundle,
    }


# -----------------------------------------------------------------------------
# 6C. Repeat the four-specification workflow for all three outcomes
# -----------------------------------------------------------------------------

def estimate_all_models(country_codes, checkpoint_prefix, quick_sample):
    """Estimate or load all specifications for all three outcomes.

    Parameters
    ----------
    country_codes : iterable or None
        Optional country restriction.
    checkpoint_prefix : str
        Prefix separating main and selected-country checkpoints.
    quick_sample : bool
        Use 5% data and sample-tagged files when ``True``.

    Returns
    -------
    tuple
        On-demand outcome bundles, IRM summaries, APOS summaries, and learner
        weight rows.
    """

    all_irm_results = []
    all_apos_results = []
    all_weight_rows = []
    estimates = {}

    for dataset, data_path, child, outcome in ANALYSIS_SPECS:
        result = estimate_one_outcome(
            dataset=dataset,
            data_path=data_path,
            child=child,
            outcome=outcome,
            country_codes=country_codes,
            checkpoint_prefix=checkpoint_prefix,
            quick_sample=quick_sample,
        )
        all_irm_results.extend(result["irm_results"])
        all_apos_results.extend(result["apos_results"])
        all_weight_rows.extend(result["weights"])
        estimates[(dataset, outcome)] = result["bundle"]

    print(
        f"All {len(estimates)} outcomes completed; "
        "12 model specifications are available.",
        flush=True,
    )
    return estimates, all_irm_results, all_apos_results, all_weight_rows


# =============================================================================
# SECTION 7 OF 10 — MAIN RESULTS AND PUBLICATION TABLES
# Purpose: turn completed model summaries into result pickles and LaTeX tables.
# Important: table stars use the final aggregated p-value, not pval_rep.
# =============================================================================

# -----------------------------------------------------------------------------
# 7A. Human-readable labels and coefficient formatting
# -----------------------------------------------------------------------------

OUTCOME_LABELS = {
    "SomeRiskHome": r"Some Risk Home (E.coli $>0$ CFU)",
    "VeryHighRiskHome": r"Very High Risk Home (E.coli $>100$ CFU)",
    "diarrhea": "Diarrhea (under-5)",
}
TREATMENT_LABELS = {
    0: "No treatment",
    1: "Boiling",
    2: "Chlorination/tablets",
    3: "Straining/settling",
    98: "Other treatment",
}




# -----------------------------------------------------------------------------
# 7B. Main IRM/APOS regression tables
# -----------------------------------------------------------------------------

def write_publication_table(
    output_dir,
    estimates,
    outcome_order,
    filename,
    caption,
    label,
    report_levels,
    folds,
    repetitions,
    specifications=("clustered",),
):
    """Write the main IRM/APOS table with outcomes arranged in columns.

    Parameters
    ----------
    output_dir : path-like
        Destination folder.
    estimates : dict
        On-demand outcome bundles from ``estimate_all_models``.
    outcome_order : list[str]
        Left-to-right outcome order.
    filename, caption, label : str
        LaTeX filename and table metadata.
    report_levels : iterable
        Treatment levels represented in the table.
    folds, repetitions : int
        Cross-fitting choices printed in the notes.
    specifications : tuple[str, ...]
        ``clustered`` and/or ``unclustered`` columns.

    Returns
    -------
    pathlib.Path
        Path of the written ``.tex`` file.
    """

    short_outcome_labels = {
        "SomeRiskHome": "Some risk",
        "VeryHighRiskHome": "Very high risk",
        "diarrhea": "Diarrhea (U5)",
    }
    short_specification_labels = {
        "clustered": "Clustered",
        "unclustered": "Unclustered",
    }

    def sample_statistics(frame, outcome, treatment):
        """Summarize sample size, PSUs, outcome mean, and treatment shares."""

        treatment_shares = frame[treatment].value_counts(
            normalize=True
        ).reindex(TREATMENT_LEVELS, fill_value=0.0)
        no_treatment = frame[treatment].eq(0)
        cluster_count = None
        if "Cluster_var" in frame.columns:
            cluster_count = frame["Cluster_var"].nunique()
        treatment_mean = None
        if treatment == "water_treatment":
            treatment_mean = float(frame[treatment].mean())
        return {
            "n": len(frame),
            "clusters": cluster_count,
            "outcome_mean": float(frame.loc[no_treatment, outcome].mean()),
            "treatment_mean": treatment_mean,
            "treatment_shares": treatment_shares,
        }

    def result_cell(summary, row_number, standard_error_override=None):
        """Format one coefficient/SE pair from a selected summary row."""

        result = summary.iloc[row_number]
        standard_error = float(result["std err"])
        if standard_error_override is not None:
            standard_error = float(standard_error_override)
        coefficient = format_coefficient(
            float(result["coef"]),
            standard_error,
            p_value=float(result["P>|t|"]),
        )
        return coefficient, f"({standard_error:.3f})"

    # Build one lightweight column at a time. Models are released immediately
    # after their summaries have been copied.
    columns = []
    for outcome in outcome_order:
        dataset = "U5" if outcome == "diarrhea" else "HH"
        bundle = estimates[(dataset, outcome)]

        for specification in specifications:
            if specification == "clustered":
                irm_key = "irm_cluster"
                apos_key = "apos_cluster"
                irm_frame_key = "irm_frame_cluster"
                apos_frame_key = "apos_frame_cluster"
            else:
                irm_key = "irm_no_cluster"
                apos_key = "apos_no_cluster"
                irm_frame_key = "irm_frame_no_cluster"
                apos_frame_key = "apos_frame_no_cluster"

            irm_model = bundle[irm_key]
            irm_summary = irm_model.summary.copy()
            irm_model = None
            gc.collect()

            apos_model = bundle[apos_key]
            apos_contrast = apos_model.causal_contrast(
                reference_levels=[0]
            )
            apos_summary = apos_contrast.summary.copy()
            if specification == "clustered":
                apos_inference = cluster_robust_framework_inference(
                    apos_contrast,
                    bundle[apos_frame_key]["Cluster_var"].to_numpy(),
                )
                apos_summary = summary_with_clustered_inference(
                    apos_summary,
                    apos_inference,
                )
            apos_summary = apos_summary.iloc[
                :len(report_levels) - 1
            ].copy()
            apos_model = None
            apos_contrast = None
            gc.collect()

            columns.append({
                "outcome": outcome,
                "specification": specification,
                "irm_summary": irm_summary,
                "apos_summary": apos_summary,
                "irm_sample": sample_statistics(
                    bundle[irm_frame_key],
                    outcome,
                    "water_treatment",
                ),
                "apos_sample": sample_statistics(
                    bundle[apos_frame_key],
                    outcome,
                    "WQ15_g",
                ),
            })

    number_of_columns = len(columns)
    lines = [
        r"% Requires: \usepackage{booktabs, graphicx, adjustbox}",
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\begin{adjustbox}{max width=\linewidth}",
        rf"\begin{{tabular}}{{l{'c' * number_of_columns}}}",
        r"\hline\hline",
    ]

    if len(specifications) == 1:
        outcome_header = [
            short_outcome_labels[column["outcome"]]
            for column in columns
        ]
        lines.append(" & " + " & ".join(outcome_header) + r" \\")
    else:
        outcome_header = [""]
        for outcome in outcome_order:
            outcome_header.append(
                rf"\multicolumn{{{len(specifications)}}}{{c}}"
                rf"{{{short_outcome_labels[outcome]}}}"
            )
        lines.append(" & ".join(outcome_header) + r" \\")
        fold_header = [
            short_specification_labels[column["specification"]]
            for column in columns
        ]
        lines.append(" & " + " & ".join(fold_header) + r" \\")

    lines.append(r"\hline")

    def table_row(row_label, values):
        """Join one readable label and its cells into a LaTeX table row."""

        return row_label + " & " + " & ".join(values) + r" \\"

    def show_once_per_outcome(values):
        """Blank duplicate descriptive values in two-specification tables."""

        if len(specifications) == 1:
            return values
        displayed_outcomes = set()
        displayed_values = []
        for column, value in zip(columns, values):
            outcome = column["outcome"]
            if outcome in displayed_outcomes:
                displayed_values.append("")
            else:
                displayed_values.append(value)
                displayed_outcomes.add(outcome)
        return displayed_values

    # IRM coefficient and descriptive statistics.
    lines.append(
        rf"\multicolumn{{{number_of_columns + 1}}}{{l}}{{\textit{{IRM}}}} \\"
    )
    irm_cells = [
        result_cell(column["irm_summary"], 0)
        for column in columns
    ]
    lines.append(table_row(
        "Any treatment",
        [coefficient for coefficient, _ in irm_cells],
    ))
    lines.append(table_row(
        "",
        [standard_error for _, standard_error in irm_cells],
    ))
    lines.append(r"\addlinespace[6pt]")
    lines.append(
        rf"\multicolumn{{{number_of_columns + 1}}}{{l}}"
        r"{\textit{Descriptive statistics}} \\"
    )
    outcome_means = [
        f"{100 * column['irm_sample']['outcome_mean']:.1f}\\%"
        for column in columns
    ]
    treatment_means = [
        f"{100 * column['irm_sample']['treatment_mean']:.1f}\\%"
        for column in columns
    ]
    lines.append(table_row(
        "Y mean, no treatment (\\%)",
        show_once_per_outcome(outcome_means),
    ))
    lines.append(table_row(
        "Treated (\\%)",
        show_once_per_outcome(treatment_means),
    ))

    # APOS coefficients and treatment shares.
    lines.append(r"\midrule")
    lines.append(
        rf"\multicolumn{{{number_of_columns + 1}}}{{l}}{{\textit{{APOS}}}} \\"
    )
    reported_treatments = [
        "Boiling",
        "Chlorination/tablets",
        "Straining/settling",
    ]
    for row_number, treatment_label in enumerate(reported_treatments):
        apos_cells = []
        for column in columns:
            apos_cells.append(result_cell(
                column["apos_summary"],
                row_number,
            ))
        lines.append(table_row(
            treatment_label,
            [coefficient for coefficient, _ in apos_cells],
        ))
        lines.append(table_row(
            "",
            [standard_error for _, standard_error in apos_cells],
        ))

    lines.append(r"\addlinespace[6pt]")
    for treatment_level in TREATMENT_LEVELS:
        shares = [
            f"{100 * column['apos_sample']['treatment_shares'].loc[treatment_level]:.1f}\\%"
            for column in columns
        ]
        lines.append(table_row(
            TREATMENT_LABELS[treatment_level],
            show_once_per_outcome(shares),
        ))

    # Sample size.
    lines.append(r"\midrule")
    lines.append(
        rf"\multicolumn{{{number_of_columns + 1}}}{{l}}{{\textit{{Sample}}}} \\"
    )
    observations = [
        f"{column['irm_sample']['n']:,}"
        for column in columns
    ]
    clusters = []
    for column in columns:
        cluster_count = column["irm_sample"]["clusters"]
        clusters.append(
            "---" if cluster_count is None else f"{cluster_count:,}"
        )
    lines.append(table_row(
        "Observations",
        show_once_per_outcome(observations),
    ))
    lines.append(table_row(
        "PSUs",
        show_once_per_outcome(clusters),
    ))

    lines.extend([
        r"\hline\hline",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\par\vspace{3pt}",
        r"\begin{minipage}{\linewidth}\scriptsize \textit{Notes:} "
        r"Cells report coefficients with significance stars and standard "
        r"errors in parentheses. Clustered specifications use cluster-level "
        r"sample splitting; ordinary specifications use observation-level "
        r"folds. APOS effects compare treatment levels 1--3 with level 0. "
        rf"Cross-fitting uses {folds} folds and {repetitions} repetitions. "
        r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$."
        r"\end{minipage}",
        r"\end{table}",
    ])

    output_path = Path(output_dir) / filename
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# -----------------------------------------------------------------------------
# 7C. Super Learner weight tables
# -----------------------------------------------------------------------------

def write_super_learner_weights_tables(
    output_dir,
    outcome_order,
    weights,
    checkpoint_prefix,
    file_suffix,
    caption_suffix,
):
    """Write one compact Super Learner weight table for each fold type.

    Parameters
    ----------
    output_dir : path-like
        Destination folder.
    outcome_order : list[str]
        Outcome display order.
    weights : pandas.DataFrame
        Averaged learner weights.
    checkpoint_prefix : str
        Main or selected-country model prefix.
    file_suffix, caption_suffix : str
        Optional selected-country output labels.

    Returns
    -------
    None
        Writes one table per fold specification.
    """

    outcome_learner_names = [name for name, _ in REGRESSORS]
    treatment_learner_names = [name for name, _ in CLASSIFIERS]
    short_outcome_labels = {
        "SomeRiskHome": "Any detectable E. coli at home",
        "VeryHighRiskHome": "Very high E. coli at home (>100 CFU/100 mL)",
        "diarrhea": "Diarrhea among children under five",
    }

    for specification in ("clustered", "unclustered"):
        lines = [
            r"% Requires: \usepackage{booktabs}",
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{Super Learner weights: {specification} folds"
            rf"{caption_suffix}}}",
            rf"\label{{tab:super-learner-weights-{specification}"
            rf"{file_suffix.replace('_', '-')}}}",
            r"\small",
            r"\begin{tabular}{lrlr}",
            r"\toprule",
            r"Learner $g(X)$ & Weight & Learner $m(X)$ & Weight \\",
            r"\midrule",
        ]

        for panel_number, outcome in enumerate(outcome_order):
            dataset = "U5" if outcome == "diarrhea" else "HH"
            model_suffix = "clustered" if specification == "clustered" else "iid"
            model_name = (
                f"{checkpoint_prefix}{dataset}_{outcome}_IRM_{model_suffix}"
            )
            selected = weights[weights["model"].eq(model_name)]

            outcome_weights = selected[
                selected["nuisance"].astype(str).str.startswith("ml_g")
            ].groupby("learner")["weight"].mean()
            treatment_weights = selected[
                selected["nuisance"].eq("ml_m")
            ].groupby("learner")["weight"].mean()

            panel_letter = "ABC"[panel_number]
            panel_label = short_outcome_labels[outcome]
            lines.append(
                rf"\multicolumn{{4}}{{l}}{{\textit{{Panel "
                rf"{panel_letter}: {panel_label}}}}} \\"
            )

            paired_learners = zip(
                outcome_learner_names,
                treatment_learner_names,
            )
            for outcome_learner, treatment_learner in paired_learners:
                outcome_value = f"{outcome_weights[outcome_learner]:.3f}"
                treatment_value = f"{treatment_weights[treatment_learner]:.3f}"

                lines.append(
                    f"{outcome_learner.replace('_', r'\_')} & {outcome_value} & "
                    f"{treatment_learner.replace('_', r'\_')} & "
                    f"{treatment_value} "
                    + r"\\"
                )

            if panel_number < len(outcome_order) - 1:
                lines.append(r"\midrule")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\vspace{3pt}",
            r"\begin{minipage}{0.9\linewidth}\footnotesize Weights are "
            r"averaged across outer folds and repetitions. The $g(X)$ learner "
            r"predicts the outcome; the $m(X)$ learner predicts treatment."
            r"\end{minipage}",
            r"\end{table}",
        ])
        output_path = (
            Path(output_dir)
            / f"table_super_learner_weights_{specification}{file_suffix}.tex"
        )
        output_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# 7D. Save all main result pickles and build all main tables
# -----------------------------------------------------------------------------

def save_main_results_and_tables(
    estimates,
    irm_result_tables,
    apos_result_tables,
    weight_rows,
    checkpoint_prefix,
    file_suffix,
    caption_suffix,
    quick_sample,
):
    """Save model summaries and generate all treatment-effect tables.

    Parameters
    ----------
    estimates : dict
        On-demand outcome bundles.
    irm_result_tables, apos_result_tables : list[pandas.DataFrame]
        Per-outcome model summaries.
    weight_rows : list[dict]
        Averaged Super Learner weights.
    checkpoint_prefix : str
        Main or selected-country model prefix.
    file_suffix, caption_suffix : str
        Optional selected-country output labels.
    quick_sample : bool
        Write result pickles with ``_sample05`` when ``True``.

    Returns
    -------
    None
        Writes result pickles and LaTeX tables.
    """

    irm_results = pd.concat(irm_result_tables, ignore_index=True)
    apos_results = pd.concat(apos_result_tables, ignore_index=True)
    pd.to_pickle(
        irm_results,
        result_pickle_path("results_irm", quick_sample, file_suffix),
    )
    pd.to_pickle(
        apos_results,
        result_pickle_path("results_apos", quick_sample, file_suffix),
    )

    if weight_rows:
        weights = pd.DataFrame(weight_rows)
        pd.to_pickle(
            weights,
            result_pickle_path(
                "results_convex_weights",
                quick_sample,
                file_suffix,
            ),
        )
    else:
        weights = pd.DataFrame()

    outcomes = ["SomeRiskHome", "VeryHighRiskHome", "diarrhea"]

    # Main table: only the preferred clustered specification.
    write_publication_table(
        TABLE_DIR,
        estimates,
        outcomes,
        f"table_water_treatment_main{file_suffix}.tex",
        f"Stacked water-treatment effects{caption_suffix}",
        f"tab:water-treatment-main{file_suffix.replace('_', '-')}",
        REPORTED_LEVELS,
        FOLDS,
        REPETITIONS,
        specifications=("clustered",),
    )

    # Appendix: compare clustered and ordinary folds.
    write_publication_table(
        TABLE_DIR,
        estimates,
        outcomes,
        f"table_water_treatment_appendix{file_suffix}.tex",
        "Stacked water-treatment effects: clustered and ordinary folds"
        f"{caption_suffix}",
        f"tab:water-treatment-appendix{file_suffix.replace('_', '-')}",
        REPORTED_LEVELS,
        FOLDS,
        REPETITIONS,
        specifications=("clustered", "unclustered"),
    )

    write_super_learner_weights_tables(
        TABLE_DIR,
        outcomes,
        weights,
        checkpoint_prefix=checkpoint_prefix,
        file_suffix=file_suffix,
        caption_suffix=caption_suffix,
    )


# =============================================================================
# SECTION 8 OF 10 — SENSITIVITY ANALYSIS
# Purpose: benchmark omitted-confounding strength and report RV/RV-alpha.
# Runtime note: every method/specification benchmark may refit nuisance models,
# so each completed block is checkpointed immediately and can be resumed.
# =============================================================================

# -----------------------------------------------------------------------------
# 8A. LaTeX sensitivity table
# -----------------------------------------------------------------------------

def write_sensitivity_summary_table(
    results,
    output_dir,
    filename,
    specifications,
    label,
):
    """Write a table for benchmark strength, RV, and RV-alpha.

    Parameters
    ----------
    results : pandas.DataFrame
        Completed sensitivity rows.
    output_dir : path-like
        Destination folder.
    filename : str
        LaTeX filename.
    specifications : tuple[str, ...]
        Fold specifications included as columns.
    label : str
        LaTeX cross-reference label.

    Returns
    -------
    pathlib.Path
        Path of the written table.
    """

    outcomes = ["SomeRiskHome", "VeryHighRiskHome", "diarrhea"]
    outcome_labels = {
        "SomeRiskHome": r"\shortstack{Some risk\\ at home}",
        "VeryHighRiskHome": r"\shortstack{Very high risk\\ at home}",
        "diarrhea": "Diarrhea",
    }
    specification_labels = {
        "clustered_folds": "Panel A: Clustered folds",
        "unclustered": "Panel B: Unclustered folds",
    }
    reported_effects = [
        ("IRM", "Any Treatment", "Any treatment"),
        ("APOS", "1", "Boiling"),
        ("APOS", "2", "Chlorination/tablets"),
        ("APOS", "3", "Straining/settling"),
    ]
    metrics = [
        ("cf_y", r"$cf_y$"),
        ("cf_d", r"$cf_d$"),
        ("rv", "RV"),
        ("rva", r"RV$_\alpha$"),
    ]

    number_of_result_columns = len(metrics) * len(outcomes)
    total_columns = 1 + number_of_result_columns
    if tuple(specifications) == ("clustered_folds",):
        caption = (
            "Sensitivity of DoubleML estimates to unobserved confounding"
        )
    else:
        caption = (
            "Sensitivity of DoubleML estimates: clustered and ordinary folds"
        )

    lines = [
        r"% Requires: \usepackage{booktabs, pdflscape, adjustbox}",
        r"\begin{landscape}",
        r"\begin{table}[p]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize\setlength{\tabcolsep}{5pt}",
        r"\begin{adjustbox}{max width=\linewidth, center}",
        rf"\begin{{tabular}}{{l{'c' * number_of_result_columns}}}",
        r"\toprule",
    ]

    outcome_header = [""]
    for outcome in outcomes:
        outcome_header.append(
            rf"\multicolumn{{{len(metrics)}}}{{c}}"
            rf"{{{outcome_labels[outcome]}}}"
        )
    lines.append(" & ".join(outcome_header) + r" \\")

    midrules = []
    for outcome_number in range(len(outcomes)):
        first_column = 2 + len(metrics) * outcome_number
        last_column = 1 + len(metrics) * (outcome_number + 1)
        midrules.append(
            rf"\cmidrule(lr){{{first_column}-{last_column}}}"
        )
    lines.append(" ".join(midrules))
    metric_labels = [label_text for _, label_text in metrics]
    lines.append(
        " & ".join(
            ["Estimand / treatment"]
            + metric_labels * len(outcomes)
        )
        + r" \\"
    )
    lines.append(r"\midrule")

    for specification_number, specification in enumerate(specifications):
        lines.append(
            rf"\multicolumn{{{total_columns}}}{{l}}"
            rf"{{\textit{{{specification_labels[specification]}}}}} \\"
        )

        previous_method = None
        for method, treatment, treatment_label in reported_effects:
            if method != previous_method:
                lines.append(
                    rf"\multicolumn{{{total_columns}}}{{l}}"
                    rf"{{\quad\textit{{{method}}}}} \\"
                )
                previous_method = method

            row = [rf"\qquad {treatment_label}"]
            for outcome in outcomes:
                selected = results[
                    results["specification"].eq(specification)
                    & results["method"].eq(method)
                    & results["treatment"].astype(str).eq(treatment)
                    & results["outcome"].eq(outcome)
                ]
                result = selected.iloc[0]
                for metric, _ in metrics:
                    row.append(f"{100 * float(result[metric]):.1f}\\%")
            lines.append(" & ".join(row) + r" \\")

        if specification_number < len(specifications) - 1:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\par\vspace{3pt}",
        r"\begin{minipage}{\linewidth}\footnotesize \textit{Notes:} The "
        r"benchmark omits the observed source-water E. coli decile block. "
        r"$cf_y$ and $cf_d$ measure the predictive strength of that omitted "
        r"block in the outcome and treatment equations. RV is the strength "
        r"of unobserved confounding needed to reduce the estimate to zero. "
        r"RV$_\alpha$ is the strength needed for the 95\% confidence interval "
        r"to include zero. All four measures are percentages."
        r"\end{minipage}",
        r"\end{table}",
        r"\end{landscape}",
    ])

    output_path = Path(output_dir) / filename
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# -----------------------------------------------------------------------------
# 8B. Find the source-water E. coli controls used for benchmarking
# -----------------------------------------------------------------------------

def source_ecoli_columns(model):
    """Identify encoded E. coli-decile controls for benchmarking.

    Parameters
    ----------
    model : fitted DoubleML model
        Model exposing its encoded control names.

    Returns
    -------
    list[str]
        Columns beginning with ``wq27_decile_``.
    """

    # make_frame() converts the decile into indicator columns such as
    # wq27_decile_2. The first category is the omitted reference category.
    return [
        column
        for column in model._dml_data.x_cols
        if column.startswith("wq27_decile_")
    ]


# -----------------------------------------------------------------------------
# 8C. Run one checkpointable sensitivity block for one fitted method
# -----------------------------------------------------------------------------

def sensitivity_for_method(
    bundle,
    dataset,
    outcome,
    specification,
    clustered,
    method,
):
    """Calculate one independently checkpointable sensitivity block.

    Parameters
    ----------
    bundle : OutcomeCheckpointBundle
        Outcome models and small table frames.
    dataset, outcome, specification : str
        Labels attached to returned rows.
    clustered : bool
        Use PSU-level sensitivity scores when ``True``.
    method : {"IRM", "APOS"}
        Model family calculated in this block.

    Returns
    -------
    list[dict]
        One IRM row or three APOS treatment-contrast rows.
    """

    if method == "IRM":
        irm_key = "irm_cluster" if clustered else "irm_no_cluster"
        irm = bundle[irm_key]
        irm_rv, irm_rva = sensitivity_params(irm)
        irm_benchmark = irm.sensitivity_benchmark(
            benchmarking_set=source_ecoli_columns(irm),
            fit_args={
                "n_jobs_cv": LEARNER_JOBS,
                "store_predictions": False,
                "store_models": False,
            },
        )
        rows = [{
            "dataset": dataset,
            "outcome": outcome,
            "specification": specification,
            "method": "IRM",
            "treatment": "Any Treatment",
            "rv": float(irm_rv[0]),
            "rva": float(irm_rva[0]),
            "cf_y": float(irm_benchmark.iloc[0]["cf_y"]),
            "cf_d": float(irm_benchmark.iloc[0]["cf_d"]),
        }]
        irm = None
        irm_benchmark = None
        gc.collect()
        return rows

    if method != "APOS":
        raise ValueError(f"Unknown sensitivity method: {method}")

    apos_key = "apos_cluster" if clustered else "apos_no_cluster"
    apos_model = bundle[apos_key]
    apos_contrast = apos_model.causal_contrast(reference_levels=[0])

    if clustered:
        apos_rv, apos_rva = sensitivity_params(
            apos_contrast,
            cluster_ids=bundle["apos_frame_cluster"][
                "Cluster_var"
            ].to_numpy(),
        )
    else:
        apos_rv, apos_rva = sensitivity_params(apos_contrast)

    # DoubleML modifies internal arrays during benchmarking. Threads prevent
    # joblib from turning those arrays into read-only memory-mapped files.
    with parallel_backend("threading"):
        apos_benchmark = apos_model.sensitivity_benchmark(
            benchmarking_set=source_ecoli_columns(apos_model),
            fit_args={
                "n_jobs_models": APOS_WORKERS,
                "n_jobs_cv": LEARNER_JOBS,
                "store_predictions": False,
                "store_models": False,
            },
        )

    rows = []
    for effect_index, treatment_level in enumerate((1, 2, 3)):
        rows.append({
            "dataset": dataset,
            "outcome": outcome,
            "specification": specification,
            "method": "APOS",
            "treatment": str(treatment_level),
            "rv": float(apos_rv[effect_index]),
            "rva": float(apos_rva[effect_index]),
            "cf_y": float(apos_benchmark.loc[treatment_level, "cf_y"]),
            "cf_d": float(apos_benchmark.loc[treatment_level, "cf_d"]),
        })

    apos_model = None
    apos_contrast = None
    apos_benchmark = None
    gc.collect()
    return rows


# -----------------------------------------------------------------------------
# 8D. Coordinate every sensitivity block and rebuild the final tables
# -----------------------------------------------------------------------------

def run_sensitivity_analysis(estimates, quick_sample):
    """Run or resume all sensitivity blocks and write their outputs.

    Parameters
    ----------
    estimates : dict
        On-demand outcome bundles from the main analysis.
    quick_sample : bool
        Select full or ``_sample05`` sensitivity checkpoints/results.

    Returns
    -------
    None
        Saves each completed block immediately, then writes result and table
        files after every block is available.
    """

    checkpoints = CheckpointStore(quick_sample)
    print("\nSensitivity analysis", flush=True)
    sensitivity_rows = []
    for (dataset, outcome), bundle in estimates.items():
        print(f"\nPreparing {dataset} — {outcome}", flush=True)
        blocks = [
            ("clustered_folds", True, "IRM"),
            ("clustered_folds", True, "APOS"),
            ("unclustered", False, "IRM"),
            ("unclustered", False, "APOS"),
        ]
        for block_number, (specification, clustered, method) in enumerate(
            blocks,
            start=1,
        ):
            checkpoint_name = (
                f"sensitivity_{dataset}_{outcome}_"
                f"{specification}_{method}"
            )
            print(
                f"Step {block_number} of {len(blocks)}: {method}, "
                f"{specification}",
                flush=True,
            )
            if checkpoints.exists(checkpoint_name):
                block_rows = checkpoints.load(checkpoint_name)
            else:
                block_rows = sensitivity_for_method(
                    bundle=bundle,
                    dataset=dataset,
                    outcome=outcome,
                    specification=specification,
                    clustered=clustered,
                    method=method,
                )
                checkpoints.save(checkpoint_name, block_rows)
            sensitivity_rows.extend(block_rows)
        print(f"Completed sensitivity: {dataset} — {outcome}", flush=True)

    sensitivity_results = pd.DataFrame(sensitivity_rows)
    pd.to_pickle(
        sensitivity_results,
        result_pickle_path("results_sensitivity", quick_sample),
    )

    write_sensitivity_summary_table(
        sensitivity_results,
        TABLE_DIR,
        filename="table_sensitivity_main.tex",
        specifications=("clustered_folds",),
        label="tab:sensitivity-main",
    )
    write_sensitivity_summary_table(
        sensitivity_results,
        TABLE_DIR,
        filename="table_sensitivity_appendix.tex",
        specifications=("clustered_folds", "unclustered"),
        label="tab:sensitivity-appendix",
    )
    print(
        f"All {len(estimates)} sensitivity analyses completed.",
        flush=True,
    )


# =============================================================================
# SECTION 9 OF 10 — HETEROGENEITY BY E. COLI DECILE AND RISK (GATE)
# Purpose: project the fitted orthogonal signal onto initial-contamination
# deciles and risk categories and compare clustered with unclustered specifications.
# =============================================================================

# -----------------------------------------------------------------------------
# 9A. Build household and under-five GATE LaTeX tables
# -----------------------------------------------------------------------------

SOURCE_ECOLI_TOP_CODE = 101


def source_ecoli_ranges_from_master_data():
    """Calculate each GATE group's exact WQ27 range from the source data.

    The ranges are calculated separately for the household and under-five
    master files. This prevents table labels from depending on hand-entered
    cutoffs or on which observations happen to enter a quick 5% run.

    Parameters
    ----------
    None

    Returns
    -------
    dict[tuple[str, str], str]
        LaTeX-ready range label keyed by ``(dataset, decile_code)``.
    """

    ranges = {}
    data_sources = {
        dataset: data_path
        for dataset, data_path, _, _ in ANALYSIS_SPECS
    }

    for dataset, data_path in data_sources.items():
        source_data = pd.read_stata(
            data_path,
            columns=["WQ27", "wq27_decile"],
            convert_categoricals=False,
        ).dropna(subset=["WQ27", "wq27_decile"])

        observed_ranges = source_data.groupby("wq27_decile")["WQ27"].agg(
            ["min", "max"]
        )
        for decile_code, observed in observed_ranges.iterrows():
            minimum = int(observed["min"])
            maximum = int(observed["max"])

            # WQ27=101 is not an observed count of 101. The MICS value label
            # defines it as the top code "more than 100" CFU/100 mL.
            maximum_label = (
                r"$>100$"
                if maximum == SOURCE_ECOLI_TOP_CODE
                else f"{maximum:,}"
            )
            range_label = (
                f"{minimum:,}"
                if minimum == maximum
                else f"{minimum:,}--{maximum_label}"
            )
            ranges[(dataset, str(int(decile_code)))] = range_label

    return ranges


def create_heterogeneity_comparison_tables(
    results,
    output_dir,
    filename_prefix,
    specifications,
    include_blp_r2,
):
    """Write GATE tables for household outcomes and under-five diarrhea.

    Parameters
    ----------
    results : pandas.DataFrame
        Long-form GATE estimates.
    output_dir : path-like
        Destination folder.
    filename_prefix : str
        Prefix used for household and diarrhea filenames.
    specifications : tuple[str, ...]
        Fold specifications included as columns.
    include_blp_r2 : bool
        Include BLP goodness-of-fit rows when ``True``.

    Returns
    -------
    list[pathlib.Path]
        Paths of the household and diarrhea tables.
    """

    selected_results = results[
        results["specification"].isin(specifications)
    ].copy()

    table_definitions = [
        ("ecoli", ["SomeRiskHome", "VeryHighRiskHome"]),
        ("diarrhea", ["diarrhea"]),
    ]
    reported_effects = [
        ("IRM stacked", "Any Treatment"),
        ("APOS stacked", "Boiling"),
        ("APOS stacked", "Chlorination/tablets"),
        ("APOS stacked", "Straining/settling"),
    ]
    specification_labels = {
        "clustered_folds": "Clustered folds",
        "unclustered": "Ordinary folds",
    }

    output_paths = []
    for table_name, outcomes in table_definitions:
        table_results = selected_results[
            selected_results["outcome"].isin(outcomes)
        ]

        number_of_effect_columns = (
            len(specifications) * len(reported_effects)
        )
        # The first two columns identify the GATE group and show the underlying
        # source-water E. coli values. Treatment effects begin after them.
        total_columns = 2 + number_of_effect_columns
        lines = [
            r"% Requires: \usepackage{booktabs, pdflscape, adjustbox}",
            r"\begin{landscape}",
            r"\begin{table}[p]",
            r"\centering",
            rf"\caption{{Heterogeneity of stacked GATE effects: {table_name}}}",
            rf"\label{{tab:{filename_prefix}-{table_name}}}",
            r"\scriptsize\setlength{\tabcolsep}{4pt}",
            r"\begin{adjustbox}{max width=\linewidth, center}",
            rf"\begin{{tabular}}{{ll{'c' * number_of_effect_columns}}}",
            r"\toprule",
        ]

        if len(specifications) == 1:
            effect_header = [
                "Heterogeneity group",
                r"\shortstack{Source E. coli range\\(CFU/100 mL)}",
            ]
            effect_header.extend(
                treatment for _, treatment in reported_effects
            )
            lines.append(" & ".join(effect_header) + r" \\")
        else:
            effect_header = [
                "Heterogeneity group",
                r"\shortstack{Source E. coli range\\(CFU/100 mL)}",
            ]
            for _, treatment in reported_effects:
                effect_header.append(
                    rf"\multicolumn{{{len(specifications)}}}{{c}}"
                    rf"{{{treatment}}}"
                )
            lines.append(" & ".join(effect_header) + r" \\")

            fold_header = ["", ""]
            for _ in reported_effects:
                for specification in specifications:
                    fold_header.append(
                        specification_labels[specification]
                    )
            lines.append(" & ".join(fold_header) + r" \\")

        lines.append(r"\midrule")

        for outcome_number, outcome in enumerate(outcomes):
            outcome_results = table_results[
                table_results["outcome"].eq(outcome)
            ]

            group_names = outcome_results["group"].drop_duplicates().tolist()
            for group_number, group_name in enumerate(group_names):
                group_results = outcome_results[
                    outcome_results["group"].eq(group_name)
                ]
                heterogeneity_label = group_results[
                    "heterogeneity_label"
                ].iloc[0]
                panel_label = (
                    f"{OUTCOME_LABELS[outcome]}: {heterogeneity_label}"
                )
                if len(outcomes) > 1:
                    panel_letter = chr(65 + outcome_number * len(group_names) + group_number)
                    panel_label = f"Panel {panel_letter}: {panel_label}"
                lines.append(
                    rf"\multicolumn{{{total_columns}}}{{l}}"
                    rf"{{{panel_label}}} \\"
                )

                group_values = sorted(
                    group_results["group_value"].drop_duplicates(),
                    key=int,
                )
                for group_value in group_values:
                    one_group = group_results[
                        group_results["group_value"].eq(group_value)
                    ]

                    coefficient_cells = []
                    standard_error_cells = []
                    for method, treatment in reported_effects:
                        for specification in specifications:
                            selected = one_group[
                                one_group["method"].eq(method)
                                & one_group["treatment_label"].eq(treatment)
                                & one_group["specification"].eq(specification)
                            ]
                            result = selected.iloc[0]
                            standard_error = float(result["se"])
                            coefficient_cells.append(format_coefficient(
                                float(result["coef"]),
                                standard_error,
                                p_value=float(result["pval"]),
                            ))
                            standard_error_cells.append(
                                f"({standard_error:.3f})"
                            )

                    group_label = str(one_group["group_label"].iloc[0])
                    ecoli_range = str(
                        one_group["source_ecoli_range"].iloc[0]
                    )
                    lines.append(
                        " & ".join(
                            [group_label, ecoli_range] + coefficient_cells
                        )
                        + r" \\"
                    )
                    lines.append(
                        " & ".join(["", ""] + standard_error_cells)
                        + r" \\"
                    )

                lines.append(
                    rf"\multicolumn{{{total_columns}}}{{l}}"
                    r"{\textit{Sample}} \\"
                )
                for statistic in ("Observations", "PSUs"):
                    statistic_cells = [rf"\quad {statistic}", ""]
                    for method, treatment in reported_effects:
                        for specification in specifications:
                            method_results = group_results[
                                group_results["method"].eq(method)
                                & group_results["treatment_label"].eq(
                                    treatment
                                )
                                & group_results["specification"].eq(
                                    specification
                                )
                            ]

                            if statistic == "Observations":
                                statistic_value = (
                                    f"{int(method_results['sample_n'].iloc[0]):,}"
                                )
                            else:
                                if specification == "clustered_folds":
                                    statistic_value = (
                                        f"{int(method_results['sample_n_psu'].iloc[0]):,}"
                                    )
                                else:
                                    statistic_value = "---"
                            statistic_cells.append(statistic_value)
                    lines.append(" & ".join(statistic_cells) + r" \\")

                if include_blp_r2:
                    lines.append(
                        rf"\multicolumn{{{total_columns}}}{{l}}"
                        r"{\textit{BLP $R^{2}$}} \\"
                    )
                    for method, treatment in reported_effects:
                        r_squared_cells = [rf"\quad {treatment}", ""]
                        for candidate_method, candidate_treatment in reported_effects:
                            if (
                                candidate_method == method
                                and candidate_treatment == treatment
                            ):
                                for specification in specifications:
                                    selected = group_results[
                                        group_results["method"].eq(method)
                                        & group_results["treatment_label"].eq(
                                            treatment
                                        )
                                        & group_results["specification"].eq(
                                            specification
                                        )
                                    ]
                                    r_squared_cells.append(
                                        f"{float(selected['r2'].iloc[0]):.3f}"
                                    )
                            else:
                                r_squared_cells.extend(
                                    [""] * len(specifications)
                                )
                        lines.append(
                            " & ".join(r_squared_cells) + r" \\"
                        )

                if group_number < len(group_names) - 1:
                    lines.append(r"\addlinespace[4pt]")

            if outcome_number < len(outcomes) - 1:
                lines.append(r"\midrule")

        if tuple(specifications) == ("clustered_folds",):
            fold_note = (
                "Clustered folds use cluster-level sample splitting."
            )
        else:
            fold_note = (
                "Clustered folds use cluster-level sample splitting; "
                "ordinary folds use observation-level splitting."
            )
        r_squared_note = ""
        if include_blp_r2:
            r_squared_note = (
                " BLP $R^2$ is descriptive; it is not a causal-model $R^2$."
            )
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\par\vspace{3pt}",
            r"\begin{minipage}{\linewidth}\footnotesize \textit{Notes:} "
            r"Coefficient rows report GATE estimates with significance "
            r"stars; the following rows report standard errors. "
            + fold_note
            + r_squared_note
            + " Source-water E. coli ranges are measured in CFU/100 mL; "
            + "the upper group includes the top-coded value above 100."
            + " The heterogeneity analysis is exploratory."
            + r"\end{minipage}",
            r"\end{table}",
            r"\end{landscape}",
        ])

        output_path = (
            Path(output_dir) / f"{filename_prefix}_{table_name}.tex"
        )
        output_path.write_text("\n".join(lines), encoding="utf-8")
        output_paths.append(output_path)

    return output_paths


GATE_GROUP_LABELS = {
    str(decile): f"Decile {decile}"
    for decile in range(1, 11)
}
GATE_GROUPS = {
    "source_ecoli": ("wq27_decile", "Initial source-water E. coli decile", GATE_GROUP_LABELS),
    "source_risk": ("RiskSource", "Initial source-water E. coli risk", {
        "0": "No Risk Source",
        "1": "Some Risk Source",
        "2": "Very High Risk Source",
    }),
}
# Mutually exclusive RiskSource categories; Some Risk excludes very high risk.
SOURCE_RISK_RANGES = {"0": "0", "1": "1--100", "2": r"$>100$"}


APOS_TREATMENT_LABELS = {
    1: "Boiling",
    2: "Chlorination/tablets",
    3: "Straining/settling",
}


# -----------------------------------------------------------------------------
# 9B. Add readable outcome, treatment, and decile labels
# -----------------------------------------------------------------------------

def add_gate_labels(
    gate_table,
    dataset,
    outcome,
    method,
    specification,
    treatment_label,
    sample_n,
    sample_n_psu,
    group="source_ecoli",
):
    """Add common analysis labels to one GATE result frame.

    Parameters
    ----------
    gate_table : pandas.DataFrame
        Group-level estimates returned by ``estimate_gate_from_contrast``.
    dataset, outcome, method, specification, treatment_label : str
        Labels identifying the estimand and fold specification.
    sample_n : int
        Number of observations in the fitted sample.
    sample_n_psu : int or None
        Number of PSUs for clustered specifications.

    Returns
    -------
    pandas.DataFrame
        The same frame with identifying columns inserted at the front.
    """

    gate_table.insert(0, "dataset", dataset)
    gate_table.insert(1, "outcome", outcome)
    gate_table.insert(2, "method", method)
    gate_table.insert(3, "specification", specification)
    gate_table.insert(4, "group", group)
    gate_table.insert(
        5,
        "heterogeneity_label",
        GATE_GROUPS[group][1],
    )
    gate_table.insert(6, "treatment_label", treatment_label)
    gate_table.insert(7, "sample_n", sample_n)
    gate_table.insert(8, "sample_n_psu", sample_n_psu)
    return gate_table


# -----------------------------------------------------------------------------
# 9C. Estimate GATEs for one method/specification from saved models
# -----------------------------------------------------------------------------

def gate_for_specification(bundle, data, dataset, outcome, child, specification, clustered):
    """Estimate both decile and risk GATEs using the same fitted models."""
    rows = []
    for group in GATE_GROUPS:
        rows.extend(gate_for_group(
            bundle, data, dataset, outcome, child, specification, clustered, group,
        ))
    return rows


def gate_for_group(
    bundle,
    data,
    dataset,
    outcome,
    child,
    specification,
    clustered,
    group,
):
    """Calculate IRM and APOS GATEs for one fold specification.

    Parameters
    ----------
    bundle : OutcomeCheckpointBundle
        Saved models and table frames for one outcome.
    data : pandas.DataFrame
        Loaded observations used to recover E. coli decile and risk groups.
    dataset, outcome, specification : str
        Labels attached to returned rows.
    child : bool
        Use child-specific controls and complete-case rules.
    clustered : bool
        Use PSU-clustered GATE covariance when ``True``; otherwise HC0.

    Returns
    -------
    list[pandas.DataFrame]
        One IRM GATE frame and three APOS treatment GATE frames.
    """

    group_column, _, group_labels = GATE_GROUPS[group]
    rows = []

    # A. GATE for the binary IRM treatment.
    irm_sample = complete_case_sample(
        data=data,
        outcome=outcome,
        treatment="water_treatment",
        child=child,
        cluster=clustered,
        allowed_levels=(0, 1),
        extra_columns=(group_column,),
    )
    irm_groups = irm_sample[group_column]
    irm_cluster_ids = None
    irm_sample_n_psu = None
    if clustered:
        irm_cluster_ids = irm_sample["Cluster_var"].to_numpy()
        irm_sample_n_psu = irm_sample["Cluster_var"].nunique()

    irm_key = "irm_cluster" if clustered else "irm_no_cluster"
    irm = bundle[irm_key]
    irm_gate = estimate_gate_from_contrast(
        contrast=irm.framework,
        treatment_level="Any Treatment",
        effect_index=0,
        group_values=irm_groups,
        cluster_ids=irm_cluster_ids,
        group_labels=group_labels,
    )
    rows.append(add_gate_labels(
        gate_table=irm_gate,
        group=group,
        dataset=dataset,
        outcome=outcome,
        method="IRM stacked",
        specification=specification,
        treatment_label="Any Treatment",
        sample_n=len(irm_sample),
        sample_n_psu=irm_sample_n_psu,
    ))
    irm = None
    irm_gate = None
    irm_sample = None
    gc.collect()

    # B. GATE for each APOS contrast against treatment level zero.
    apos_sample = complete_case_sample(
        data=data,
        outcome=outcome,
        treatment="WQ15_g",
        child=child,
        cluster=clustered,
        allowed_levels=TREATMENT_LEVELS,
        extra_columns=(group_column,),
    )
    apos_groups = apos_sample[group_column]
    apos_cluster_ids = None
    apos_sample_n_psu = None
    if clustered:
        apos_cluster_ids = apos_sample["Cluster_var"].to_numpy()
        apos_sample_n_psu = apos_sample["Cluster_var"].nunique()

    apos_key = "apos_cluster" if clustered else "apos_no_cluster"
    apos_model = bundle[apos_key]
    apos_contrast = apos_model.causal_contrast(reference_levels=[0])

    for effect_index, treatment_level in enumerate((1, 2, 3)):
        apos_gate = estimate_gate_from_contrast(
            contrast=apos_contrast,
            treatment_level=treatment_level,
            effect_index=effect_index,
            group_values=apos_groups,
            cluster_ids=apos_cluster_ids,
            group_labels=group_labels,
        )
        rows.append(add_gate_labels(
            gate_table=apos_gate,
            group=group,
            dataset=dataset,
            outcome=outcome,
            method="APOS stacked",
            specification=specification,
            treatment_label=APOS_TREATMENT_LABELS[treatment_level],
            sample_n=len(apos_sample),
            sample_n_psu=apos_sample_n_psu,
        ))

    apos_model = None
    apos_contrast = None
    apos_sample = None
    gc.collect()
    return rows


# -----------------------------------------------------------------------------
# 9D. Coordinate all GATE specifications and write the final tables
# -----------------------------------------------------------------------------

def write_gate_tables(gate_results):
    """Preserve combined GATE tables and publish each grouping separately."""
    required = {'source_ecoli', 'source_risk'}
    if set(gate_results['group']) != required:
        raise ValueError('Both source-water deciles and risk groups are required')
    for label, subset in [('', gate_results),
                          ('_deciles', gate_results.loc[gate_results.group.eq('source_ecoli')]),
                          ('_risk_groups', gate_results.loc[gate_results.group.eq('source_risk')])]:
        for section, specifications, include_r2 in [
            ('main', ('clustered_folds',), False),
            ('appendix', ('clustered_folds', 'unclustered'), True),
        ]:
            paths = create_heterogeneity_comparison_tables(
                subset, output_dir=TABLE_DIR,
                filename_prefix=f'table_gate_{section}{label}',
                specifications=specifications, include_blp_r2=include_r2)
            # Explicit captions distinguish the estimand, grouping and outcomes.
            grouping = {'': 'Deciles and Risk Groups', '_deciles': 'Source-Water Deciles',
                        '_risk_groups': 'Three Source-Water Risk Groups'}[label]
            for path in paths:
                text = path.read_text()
                for outcome, title in [('ecoli', 'Water Quality'), ('diarrhea', 'Diarrhea (U5)')]:
                    text = text.replace(
                        f'Heterogeneity of stacked GATE effects: {outcome}',
                        f'GATE ({'ATE'}) by {grouping}: {title}'
                        + (' -- Clustered and Unclustered' if section == 'appendix' else ''))
                path.write_text(text, encoding='utf-8')


def run_gate_analysis(estimates, quick_sample):
    """Run all GATE projections and write their result/table files.

    Parameters
    ----------
    estimates : dict
        On-demand outcome bundles from the main analysis.
    quick_sample : bool
        Reload matching full or 5% data and tag the result pickle accordingly.

    Returns
    -------
    None
        Writes one result pickle and four LaTeX tables.
    """

    print("\nGATE analysis", flush=True)
    gate_rows = []
    for dataset, data_path, child, outcome in ANALYSIS_SPECS:
        print(f"\nPreparing {dataset} — {outcome}", flush=True)
        bundle = estimates[(dataset, outcome)]
        data = load_analysis_data(
            data_path,
            outcome,
            child,
            country_codes=None,
            quick_sample=quick_sample,
        )

        print("Step 1 of 2: GATE with clustered folds", flush=True)
        clustered_rows = gate_for_specification(
            bundle=bundle,
            data=data,
            dataset=dataset,
            outcome=outcome,
            child=child,
            specification="clustered_folds",
            clustered=True,
        )
        gate_rows.extend(clustered_rows)

        print("Step 2 of 2: GATE with ordinary folds", flush=True)
        ordinary_rows = gate_for_specification(
            bundle=bundle,
            data=data,
            dataset=dataset,
            outcome=outcome,
            child=child,
            specification="unclustered",
            clustered=False,
        )
        gate_rows.extend(ordinary_rows)
        print(f"Completed GATE: {dataset} — {outcome}", flush=True)

        data = None
        gc.collect()

    gate_results = pd.concat(gate_rows, ignore_index=True)

    # Attach exact source-water E. coli ranges calculated from WQ27 in each
    # full master file. These are descriptive labels; no model is refitted.
    source_ecoli_ranges = source_ecoli_ranges_from_master_data()
    gate_results["source_ecoli_range"] = [
        SOURCE_RISK_RANGES[str(row.group_value)]
        if row.group == "source_risk"
        else source_ecoli_ranges[(str(row.dataset), str(row.group_value))]
        for row in gate_results.itertuples()
    ]

    pd.to_pickle(
        gate_results,
        result_pickle_path("results_heterogeneity_gates", quick_sample),
    )
    write_gate_tables(gate_results)
    print(
        f"All {len(ANALYSIS_SPECS)} GATE analyses completed.",
        flush=True,
    )


# =============================================================================
# SECTION 10 OF 10 — COMPLETE RUN ORDER AND MANIFEST
# Read main() below for the shortest end-to-end description of the script.
# =============================================================================

# -----------------------------------------------------------------------------
# 10A. Record settings and produced files
# -----------------------------------------------------------------------------

def write_manifest():
    """Record run settings and generated filenames in JSON.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Writes ``Output/ATE/manifest.json``.
    """

    checkpoint_fingerprint, checkpoint_details = checkpoint_provenance()
    manifest = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "checkpoint_provenance": checkpoint_details,
        "seed": SEED,
        "sampled": SAMPLED,
        "sample_frac": SAMPLE_FRAC if SAMPLED else None,
        "folds": FOLDS,
        "repetitions": REPETITIONS,
        "inner_folds": INNER_FOLDS,
        "apos_workers": APOS_WORKERS,
        "learner_jobs": LEARNER_JOBS,
        "learners_outcome": [name for name, _ in REGRESSORS],
        "learners_treatment": [name for name, _ in CLASSIFIERS],
        "treatment_levels": list(TREATMENT_LEVELS),
        "selected_countries": SELECTED_COUNTRIES,
        "checkpoints": sorted(
            path.name
            for path in CHECKPOINT_DIR.glob("*.pkl")
            if f"_{checkpoint_fingerprint[:12]}" in path.stem
        ),
        "tables": sorted(path.name for path in TABLE_DIR.glob("*.tex")),
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


# -----------------------------------------------------------------------------
# 10B. Run the complete analysis in publication order
# -----------------------------------------------------------------------------

def main():
    """Run the complete analysis in its documented publication order.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Loads or fits models, writes checkpoints/results/tables, and records
        the final manifest.
    """

    # Step 1. Fit IRM and APOS for all three outcomes.
    estimates, irm_tables, apos_tables, weight_rows = estimate_all_models(
        country_codes=None,
        checkpoint_prefix="",
        quick_sample=SAMPLED,
    )

    # Step 2. Save results and build the main publication tables.
    save_main_results_and_tables(
        estimates=estimates,
        irm_result_tables=irm_tables,
        apos_result_tables=apos_tables,
        weight_rows=weight_rows,
        checkpoint_prefix="",
        file_suffix="",
        caption_suffix="",
        quick_sample=SAMPLED,
    )

    # Step 3. Assess sensitivity to unobserved confounding.
    run_sensitivity_analysis(estimates, quick_sample=SAMPLED)

    # Step 4. Estimate heterogeneity by initial E. coli decile and risk group.
    run_gate_analysis(estimates, quick_sample=SAMPLED)

    # Step 5. Repeat the main analysis in four selected countries.
    print("\nSelected-country main analysis", flush=True)
    print(
        "Countries: " + ", ".join(SELECTED_COUNTRIES),
        flush=True,
    )
    (
        selected_estimates,
        selected_irm_tables,
        selected_apos_tables,
        selected_weight_rows,
    ) = estimate_all_models(
        country_codes=tuple(SELECTED_COUNTRIES.values()),
        checkpoint_prefix="selected_countries_",
        # A smoke run must not create untagged selected-country checkpoints
        # with the reduced fold/learner configuration. Full runs remain
        # untagged because SAMPLED is False.
        quick_sample=SAMPLED,
    )
    save_main_results_and_tables(
        estimates=selected_estimates,
        irm_result_tables=selected_irm_tables,
        apos_result_tables=selected_apos_tables,
        weight_rows=selected_weight_rows,
        checkpoint_prefix="selected_countries_",
        file_suffix="_selected_countries",
        caption_suffix=(
            ": Dominican Republic, Guyana, Honduras, and Malawi"
        ),
        quick_sample=SAMPLED,
    )
    print("Selected-country main analysis completed.", flush=True)

    # Step 6. Record the options and files produced.
    write_manifest()

    print("\nAnalysis finished.")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print(f"Tables:      {TABLE_DIR}")


if __name__ == "__main__":
    main()
