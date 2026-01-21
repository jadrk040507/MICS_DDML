# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DoubleML causal inference analysis estimating the effect of water treatment on E.coli risk using MICS (Multiple Indicator Cluster Surveys) data.

## Commands

```bash
# Install dependencies
uv sync

# Run Jupyter notebooks
jupyter notebook
```

## Notebook Execution Order

1. `01_data.ipynb` - Data preprocessing (creates `mics_clean.csv`)
2. `02_doubleml_analysis.ipynb` - Main DoubleML analysis (PLR & IRM models)
3. `03_robustness_analysis.ipynb` - Robustness checks with 10 ML algorithms

## Variable Definitions

### Outcome Variables (Y)
- `SomeRiskHome`: Binary E.coli risk indicator
- `VeryHighRiskHome`: Binary high E.coli risk indicator

### Treatment Variable (T)
- `water_treatment`: Binary (0/1)

### Basic Controls
- `windex5`: Wealth index (ordinal 0-4: Poorest to Richest)
- `helevel`: Education level (ordinal 0-2: None, Primary, Secondary+)
- `country_cat`: Country (categorical, one-hot encoded)
- `urban`: Urban/Rural (binary: 0=Rural, 1=Urban)
- `WS1_g`: Water source groups (categorical, one-hot encoded)
- `wq27_decile`: Water quality decile (ordinal)

### Extended Controls (Basic + Additional)

**Binary:**
- `Any_U5`, `Girls_less_than15`, `Boys_15or_less` - Household composition
- `improved_latrine`, `Flush`, `Pit_latrine`, `Open_defecation` - Sanitation
- `rainy_season` - Season indicator
- `RainandSurfaceWater`, `PurchasedWater` - Water source types
- `Basic_water_service`, `Limited_water_service`, `Unimproved_water_service` - Service levels
- `ImprovedWaterSource`, `PipedWater`, `WellandSpringWater` - Water infrastructure

**Ordinal:**
- `water_carrier_edu` - Education of water carrier

## Key Patterns

- Use `display()` instead of `print()` for DataFrame outputs in notebooks
- Ordinal variables: encode with integer mapping preserving order
- Categorical variables: one-hot encode with `drop='first'`
- DoubleML requires complete cases (NaN rows dropped)

## Data Files

- `mics.csv` - Raw MICS survey data (785 columns)
- `mics_clean.csv` - Preprocessed data for DoubleML analysis
