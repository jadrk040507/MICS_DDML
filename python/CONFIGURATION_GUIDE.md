# DoubleML MICS Analysis - Configuration Guide

## Overview

The code has been reorganized to make it **easy to edit** all analysis parameters from a single file: `config.py`.

## File Structure

```
python/
├── config.py       # ← EDIT THIS FILE to change your analysis
├── data.py         # Data preparation (uses config.py)
├── learners.py     # ML learner definitions
├── models.py       # Model estimation and export
└── run.py          # Main analysis script
```

## How to Edit Your Analysis

### 1. Change Outcomes (Dependent Variables)

In `config.py`, edit the `OUTCOMES` list:

```python
OUTCOMES = [
    {
        "var": "some_risk_home",      # Column name in dataset
        "label": "Some Risk Home",     # Display name
        "desc": "E.coli count 0-100",  # Description
        "type": "binary"               # "binary" or "continuous"
    },
    {
        "var": "diarrhea",
        "label": "Diarrhea",
        "desc": "Child had diarrhea in last 2 weeks",
        "type": "binary"
    },
    # Add more outcomes here...
]
```

**To add a new outcome:** Simply add a new dictionary to the list. The analysis will automatically include it in all models.

### 2. Change Treatments

**Any Treatment (Analysis 1):**
```python
ANY_TREATMENT = {
    "var": "any_treatment",
    "label": "Any Treatment"
}
```

**Specific Treatments (Analysis 2):**
```python
SPECIFIC_TREATMENTS = [
    {"var": "treat_boil", "label": "Boil"},
    {"var": "treat_chlorine", "label": "Chlorine"},
    {"var": "treat_filter", "label": "Filter"},
    {"var": "treat_other", "label": "Other"}
    # Add more treatments here...
]
```

### 3. Change Confounders (Control Variables)

Edit the `BASE_CONFOUNDERS` list in `config.py`:

```python
BASE_CONFOUNDERS = [
    # Wealth quintiles
    "wealth_q1", "wealth_q2", "wealth_q3", "wealth_q4", "wealth_q5",
    
    # Education
    "edu_0", "edu_1", "edu_2", "edu_3", "edu_4", "edu_na",
    
    # Urban/rural
    "urban_bin",
    
    # Sanitation
    "sanitation",
    
    # Household composition
    "num_children",
    
    # Water source (dummies created automatically)
    "water_source",
    
    # Country fixed effects (dummies created automatically)
    "country",
    
    # Source risk level (ALWAYS included as control)
    # 0 = No Risk, 1 = Moderate, 2 = Very High Risk
    "risk_source"  # ← This is automatically included in ALL models
]
```

**Important:** `risk_source` is automatically included in all models as a control variable. This ensures that all estimates control for source water risk level (0=No Risk, 1=Moderate, 2=Very High Risk).

**Outcome-specific confounders:**
```python
DIARRHEA_ADDITIONAL_CONFOUNDERS = [
    # Add extra confounders specific to diarrhea model
    # Example: "child_age", "breastfeeding", etc.
]
```

### 4. Change Subgroup Analysis

```python
# Which variable to use for subgroups
SUBGROUP_VAR = "risk_source"

# Labels for interpretation
SUBGROUP_LABELS = {
    0: "No Risk",
    1: "Moderate Risk", 
    2: "Very High Risk"
}

# Which treatments to analyze by subgroup
SUBGROUP_TREATMENTS = [
    {"var": "treat_boil", "label": "Boil"}
    # Add more treatments for subgroup analysis
]
```

### 5. Change Cross-Validation Settings

```python
N_FOLDS = 2    # Number of CV folds
N_REP = 2      # Number of repetitions
```

### 6. Change ML Learners

```python
LEARNER_NAMES = ["ols", "lasso", "ridge", "enet", "rf", "xgb", "stacked"]
```

To add/remove learners, edit `learners.py` in the `create_learners()` function.

## Key Features

### ✅ Automatic Variable Creation

The `data.py` module automatically creates:
- **Wealth quintile dummies** from `windex5`
- **Education dummies** from `helevel`
- **Water source dummies** from `WS1`
- **Country dummies** from `Country`
- **Risk source** from `RiskSource` (always included)
- **Outcome variables** (e.g., `some_risk_home`, `very_high_risk_home` from `WQ26`)

### ✅ Risk Source Always Controlled

All models automatically include `risk_source` as a control variable:
- **Analysis 1 (Any Treatment):** Controls for risk_source
- **Analysis 2 (Specific Treatments):** Controls for risk_source
- **Analysis 3 (Subgroups):** Stratified by risk_source levels (0, 1, 2)

### ✅ Diarrhea Outcome Included

The diarrhea outcome is now included by default in all analyses:
```python
{
    "var": "diarrhea",
    "label": "Diarrhea",
    "desc": "Child had diarrhea in last 2 weeks (binary)",
    "type": "binary"
}
```

### ✅ Data Validation

The script validates that all required variables exist before running:
```python
if not data.validate_data(dt):
    print("\nERROR: Data validation failed...")
    return
```

## Running the Analysis

```bash
cd python
python run.py
```

The script will:
1. Load and prepare data
2. Validate all variables
3. Create ML learners
4. Run Analysis 1: Any Treatment (all outcomes)
5. Run Analysis 2: Specific Treatments (all outcomes)
6. Run Analysis 3: Subgroups by Risk Source (all outcomes)
7. Export results (pickle, CSV, LaTeX)

## Output Files

All results are saved to `Output/`:
- `results_any_treatment.pkl` - Analysis 1 results
- `results_multi_treatment.pkl` - Analysis 2 results
- `results_subgroups.pkl` - Analysis 3 results
- `results_all.pkl` - Combined results
- `results_all.csv` - Combined results in CSV format
- `tables.tex` - LaTeX table

## Example: Adding a New Outcome

Let's say you want to add "stunting" as an outcome:

1. **Edit `config.py`:**
```python
OUTCOMES = [
    # ... existing outcomes ...
    {
        "var": "stunting",
        "label": "Stunting",
        "desc": "Child is stunted (height-for-age < -2 SD)",
        "type": "binary"
    }
]
```

2. **Make sure `stunting` exists in your dataset** (or add code to create it in `data.py`)

3. **Run the analysis** - it will automatically include stunting in all models!

## Troubleshooting

### "Column not found" errors
- Check that the column name in `config.py` matches exactly with your dataset
- Column names are case-sensitive

### "Too few observations" errors
- Increase `MIN_OBSERVATIONS` in `config.py` (default: 50)
- Or check your data filtering

### Memory issues
- Reduce `N_FOLDS` or `N_REP` in `config.py`
- Use fewer learners
- Reduce tree depth/estimators in `learners.py`

## Questions?

Edit `config.py` - all settings are documented with comments!
