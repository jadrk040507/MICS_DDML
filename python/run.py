"""
Main analysis script for the DoubleML MICS Analysis.

This script runs the complete analysis pipeline.
All configuration is in config.py - edit that file to change the analysis.
"""

from pathlib import Path
import warnings
from typing import List, Dict, Any

import numpy as np
import pandas as pd

# Import project modules
import config
import data
import learners
import models

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main() -> None:
    """Run the complete MICS DoubleML analysis."""
    print("\n" + "=" * 60)
    print("MICS DoubleML Analysis")
    print("=" * 60)

    # ==========================================================================
    # LOAD AND PREPARE DATA
    # ==========================================================================
    print("\n[1/5] Loading and preparing data...")
    dt = data.prepare_data(config.DATA_FILE)
    
    # Validate data
    if not data.validate_data(dt):
        print("\nERROR: Data validation failed. Please check config.py and your dataset.")
        return
    
    # Print summary
    data.get_summary(dt)

    # ==========================================================================
    # CREATE LEARNERS (reused across all analyses)
    # ==========================================================================
    print("\n[2/5] Creating machine learning learners...")
    learners_dict = learners.create_learners(learner_type="binary")
    
    # Add stacked ensemble
    stacked_g, stacked_m = learners.create_stacked_ensemble()
    learners_dict['stacked'] = {'g': stacked_g, 'm': stacked_m}
    
    print(f"  Created {len(learners_dict)} learners: {', '.join(learners_dict.keys())}")

    # ==========================================================================
    # ANALYSIS 1: ANY TREATMENT
    # ==========================================================================
    print("\n" + "=" * 60)
    print("[3/5] ANALYSIS 1: ANY WATER TREATMENT")
    print("=" * 60)

    # Filter to observations with valid treatment indicator
    dt_analysis = dt[~dt['any_treatment'].isna()].copy()

    # Run analysis for ALL outcomes
    results_any: List[Dict[str, Any]] = []
    
    for outcome in config.OUTCOMES:
        print(f"\n  Outcome: {outcome['label']}")
        outcome_results = models.run_analysis(
            dt=dt_analysis,
            outcomes=[outcome],
            treatments=[config.ANY_TREATMENT],
            learners=learners_dict,
            include_source_ecoli=config.INCLUDE_SOURCE_ECOLI
        )
        results_any.extend(outcome_results)

    # Save results
    models.export_results(results_any, "results_any_treatment.pkl")
    print(f"\n  ✓ Saved {len(results_any)} results")

    # ==========================================================================
    # ANALYSIS 2: SPECIFIC TREATMENT METHODS
    # ==========================================================================
    print("\n" + "=" * 60)
    print("[4/5] ANALYSIS 2: SPECIFIC TREATMENT METHODS")
    print("=" * 60)

    # Filter to observations with valid treatment indicators
    dt_multi = dt[
        (dt['no_treatment'] == 1) |
        (dt['boil'] == 1) |
        (dt['chlorine'] == 1) |
        (dt['filter'] == 1) |
        (dt['other_treat'] == 1)
    ].copy()
    
    # Create treatment indicators for analysis
    dt_multi['treat_boil'] = dt_multi['boil']
    dt_multi['treat_chlorine'] = dt_multi['chlorine']
    dt_multi['treat_filter'] = dt_multi['filter']
    dt_multi['treat_other'] = dt_multi['other_treat']

    # Run analysis for ALL outcomes and ALL specific treatments
    results_multi: List[Dict[str, Any]] = []
    
    for outcome in config.OUTCOMES:
        print(f"\n  Outcome: {outcome['label']}")
        outcome_results = models.run_analysis(
            dt=dt_multi,
            outcomes=[outcome],
            treatments=config.SPECIFIC_TREATMENTS,
            learners=learners_dict,
            include_source_ecoli=config.INCLUDE_SOURCE_ECOLI
        )
        results_multi.extend(outcome_results)

    # Save results
    models.export_results(results_multi, "results_multi_treatment.pkl")
    print(f"\n  ✓ Saved {len(results_multi)} results")

    # ==========================================================================
    # ANALYSIS 3: SUBGROUPS BY SOURCE RISK
    # ==========================================================================
    print("\n" + "=" * 60)
    print("[5/5] ANALYSIS 3: SUBGROUPS BY SOURCE RISK")
    print("=" * 60)

    # Subgroup analysis for selected treatments
    subgroup_results: List[Dict[str, Any]] = []
    
    print(f"\n  Subgroup variable: {config.SUBGROUP_VAR}")
    print(f"  Subgroup labels: {config.SUBGROUP_LABELS}")
    
    for risk_level in sorted(config.SUBGROUP_LABELS.keys()):
        print(f"\n  --- {config.SUBGROUP_LABELS[risk_level]} (level {risk_level}) ---")
        
        # Filter to subgroup
        dt_sub = dt_multi[dt_multi[config.SUBGROUP_VAR] == risk_level].copy()
        
        # Check minimum observations
        if dt_sub.shape[0] < config.MIN_OBSERVATIONS:
            print(f"    SKIP: Too few observations ({dt_sub.shape[0]:,})")
            continue
        
        print(f"    N = {dt_sub.shape[0]:,} observations")
        
        # Run analysis for ALL outcomes
        for outcome in config.OUTCOMES:
            print(f"\n    Outcome: {outcome['label']}")
            outcome_results = models.run_analysis(
                dt=dt_sub,
                outcomes=[outcome],
                treatments=config.SUBGROUP_TREATMENTS,
                learners=learners_dict,
                include_source_ecoli=config.INCLUDE_SOURCE_ECOLI,
                subgroup_var=config.SUBGROUP_VAR,
                subgroup_val=risk_level
            )
            
            # Tag results with subgroup info
            for r in outcome_results:
                r['subgroup_analysis'] = True
                r['risk_source'] = risk_level
            
            subgroup_results.extend(outcome_results)

    # Save subgroup results
    if subgroup_results:
        models.export_results(subgroup_results, "results_subgroups.pkl")
        print(f"\n  ✓ Saved {len(subgroup_results)} subgroup results")
    else:
        print("\n  No subgroup results to save")

    # ==========================================================================
    # EXPORT ALL RESULTS
    # ==========================================================================
    print("\n" + "=" * 60)
    print("EXPORTING RESULTS")
    print("=" * 60)

    # Combine all results with analysis labels
    all_results: List[Dict[str, Any]] = []
    
    for r in results_any:
        r['analysis'] = 'any_treatment'
        all_results.append(r)
    
    for r in results_multi:
        r['analysis'] = 'multi_treatment'
        all_results.append(r)
    
    for r in subgroup_results:
        r['analysis'] = 'subgroups'
        all_results.append(r)

    # Export LaTeX tables
    models.export_latex(all_results, "tables.tex")

    # Save combined results
    models.export_results(all_results, "results_all.pkl")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {config.OUTPUT_DIR}")
    print(f"Checkpoints saved to: {config.CHECKPOINT_DIR}")

    # Summary of significant effects
    print("\n=== SUMMARY OF SIGNIFICANT EFFECTS ===\n")
    
    # Significant if CI does not contain zero
    sig_results = [
        r for r in all_results 
        if r['ci_lower'] > 0 or r['ci_upper'] < 0
    ]
    
    if sig_results:
        print(f"Found {len(sig_results)} significant effects out of {len(all_results)} total.\n")
        
        # Print top significant effects
        print("Top significant effects:")
        for r in sorted(sig_results, key=lambda x: abs(x['coef']), reverse=True)[:10]:
            sig_marker = "***" if (r['ci_lower'] > 0 and r['coef'] > 0) or (r['ci_upper'] < 0 and r['coef'] < 0) else ""
            print(f"  {r['outcome']} | {r['treatment']} | {r['learner']}: "
                  f"{r['coef']:.4f} ({r['se']:.4f}) [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}] "
                  f"N={r['n']:,} {sig_marker}")
    else:
        print("No significant effects found (all CIs contain zero).")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
