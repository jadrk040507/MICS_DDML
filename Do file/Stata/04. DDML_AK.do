*--------------------------------------------------------------------
* Project: MICS – Impact of Water Treatment
* File:    DDML_Final.do
* Purpose: Estimate causal effect of household water treatment
*          on childhood diarrhea using Double/Debiased ML (DDML)
* Author:  Juan Álvaro
*--------------------------------------------------------------------

/*
Comments:
This version uses two models from the DDML package: PLM and Interactive. For this initial analysis, I (JA) took Akito's control variables from 3.Analysis.do and added a bunch of different learners to see the possible results. I saw that all of this can be recreated using only pystacked with Stata and then residualizing. This is useful since pystacked has elements that allows to graph performance plots.
Right now, the learners are:
- OLS
- Logit
- Lasso
- Ridge
- Elastic Net
- Random Forest
- Gradient Boost

The stacked versions are shortstacked for performance.

Next steps:
Run all of this with Ecoli presence (either WQ26 or VeryHighRisk)
*/

version 17
clear all
set more off
set seed 12345
set linesize 135

*====================================================================
* 1. PATHS & ENVIRONMENT
*====================================================================
* Adjust paths automatically based on username
else if c(username) == "akitokamei" {		
	global Dropbox "/Users/akitokamei/Library/CloudStorage/Dropbox/"
	global Overleaf "${Dropbox}Apps/Overleaf/"	

}

else if c(username) == "jadrk" {		
	global Dropbox "C:/Users/jadrk/Dropbox/"
	* global Overleaf "----"			
}

global Data_Final "${Dropbox}MICS_DDML/Data/3. Final/"
global Tables     "${Overleaf}MICS_DDML/Table/"
global Figures    "${Overleaf}MICS_DDML/Figure/"


cap program drop start_from_final
program define   start_from_final

*====================================================================
* 2. LOAD DATA & (OPTIONAL) TEST SAMPLE
*====================================================================
use "${Data_Final}MASTER_MICS_U5_DDML.dta", clear
use "${Data_Final}MASTER_MICS_FINAL.dta", clear

*====================================================================
* 3. DATA CLEANING & RECODING
*====================================================================
* Education: collapse into 3 categories
recode helevel (2/4 = 2)
label define helevell ///
    0 "No education" ///
    1 "Primary" ///
    2 "Secondary or higher", modify
label values helevel helevell


* IMPORTANT:
* ddml interactive requires NO missing values in Y, D, or X
drop if SomeRiskHome==.
drop if water_treatment==.

* Check the need for WQ11
foreach v in urban windex_ur helevel WQ11 {
    drop if missing(`v')
}

* NOTE: Sampling is for debugging/testing only
* sample 5
drop if RiskSource==0

end

													*====================================================================
													* 0. SomeRiskHome: Modereate & High risk (VeryHighRiskHome)
													*====================================================================

start_from_final
*====================================================================
* 1.1. VARIABLE DEFINITIONS
*====================================================================
* Outcome, Treatment, Control
local Reps 5

global Y SomeRiskHome
global D water_treatment
global X i.windex_ur i.helevel WQ27 i.country_cat i.urban i.WS1_g 
* i.HH51
reg   $Y $D $X

*====================================================================
* 1.2. DDML – PARTIAL LINEAR MODEL (Ecoli)
*====================================================================
ddml init partial, ///
    kfolds(10) ///
    reps(`Reps') ///
    mname(m_ecoli)

*--------------------------------------------------------------------
* E[Y | X]  — Outcome model (classification)
*--------------------------------------------------------------------
ddml E[Y|X], mname(m_ecoli) learner(Y_ols):   reg   $Y $X
*ddml E[Y|X], mname(m_ecoli) learner(Y_logit): logit $Y $X
ddml E[Y|X], mname(m_ecoli) learner(Y_lasso): pystacked $Y $X, method(lassocv)  type(reg)
ddml E[Y|X], mname(m_ecoli) learner(Y_ridge): pystacked $Y $X, method(ridgecv)  type(reg)
ddml E[Y|X], mname(m_ecoli) learner(Y_enet):  pystacked $Y $X, method(elasticcv) type(reg)
ddml E[Y|X], mname(m_ecoli) learner(Y_rf):    pystacked $Y $X, method(rf)        type(reg) njobs(-1)
ddml E[Y|X], mname(m_ecoli) learner(Y_gb):    pystacked $Y $X, method(gradboost) type(reg) njobs(-1)
ddml E[Y|X], mname(m_ecoli) learner(Y_nnet):  pystacked $Y $X, method(nnet)      type(reg)

* Stacked ensemble
ddml E[Y|X], mname(m_ecoli) learner(Y_stack): ///
    pystacked $Y $X ///
        || method(lassocv) ///
        || method(ridgecv) ///
        || method(elasticcv) ///
        || method(rf) ///
        || method(gradboost) ///
		|| method(nnet), ///
    type(reg) njobs(-1)

*--------------------------------------------------------------------
* E[D|X]
*--------------------------------------------------------------------
ddml E[D|X], mname(m_ecoli) learner(D_ols):   reg   $D $X
ddml E[D|X], mname(m_ecoli) learner(D_logit): logit $D $X
ddml E[D|X], mname(m_ecoli) learner(D_lasso): pystacked $D $X, method(lassocv)  type(class)
ddml E[D|X], mname(m_ecoli) learner(D_ridge): pystacked $D $X, method(ridgecv)  type(class)
ddml E[D|X], mname(m_ecoli) learner(D_enet):  pystacked $D $X, method(elasticcv) type(class)
ddml E[D|X], mname(m_ecoli) learner(D_rf):    pystacked $D $X, method(rf)        type(class) njobs(-1)
ddml E[D|X], mname(m_ecoli) learner(D_gb):    pystacked $D $X, method(gradboost) type(class) njobs(-1)
ddml E[D|X], mname(m_ecoli) learner(D_nnet):  pystacked $D $X, method(nnet)      type(class)

* Stacked ensemble
ddml E[D|X], mname(m_ecoli) learner(D_stack): ///
    pystacked $D $X ///
        || method(logit) ///
        || method(lassocv) ///
        || method(ridgecv) ///
        || method(elasticcv) ///
        || method(rf) ///
        || method(gradboost) ///
		|| method(nnet), ///
    type(class) njobs(-1)

*--------------------------------------------------------------------
* Cross-fitting & estimation
*--------------------------------------------------------------------
ddml crossfit, mname(m_ecoli) shortstack
eststo: ddml estimate, mname(m_ecoli) robust allcombos

************************************************************
* 4) Export comparison table (OLS vs DDML LASSO/RF)
************************************************************
esttab  using "${Tables}ddml_partial_SomeEcoli_bi.tex", replace ///
    title("Diarrhea: OLS vs. DDML (LASSO / RF) and BEST — Partial Effect of \$D\$ on \$Y\$") ///
    booktabs label compress nonotes mtitle("OLS" "LASSO" "Random Forest" "Best") ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    keep($D) ///
    stats(N nreps D_water_treatment_ss_mse Y_SomeRiskHome_ss_mse, fmt(%9.0fc %9.0fc %9.4fc %9.2fc ) labels(`"Observations"' `"Reps"'))	
eststo clear

start_from_final
*====================================================================
* 1.1. VARIABLE DEFINITIONS
*====================================================================
* Outcome, Treatment, Control
local Reps 5

global Y SomeRiskHome
global D water_treatment
global X i.windex_ur i.helevel WQ27 i.country_cat i.urban i.WS1_g 
reg   $Y $D $X

*====================================================================
* 9. DDML – INTERACTIVE MODEL (Ecoli)
*====================================================================
ddml init interactive, ///
	kfolds(10) /// 
	reps(`Reps') ///
	mname(m_interactive)

*--------------------------------------------------------------------
* E[Y | X, D] — Outcome models by treatment status
*--------------------------------------------------------------------
ddml E[Y|X,D], mname(m_interactive) learner(Y_ols):   reg   $Y $X
*ddml E[Y|X,D], mname(m_interactive) learner(Y_logit): logit $Y $X
ddml E[Y|X,D], mname(m_interactive) learner(Y_lasso): pystacked $Y $X, method(lassocv)  type(reg)
ddml E[Y|X,D], mname(m_interactive) learner(Y_ridge): pystacked $Y $X, method(ridgecv)  type(reg)
ddml E[Y|X,D], mname(m_interactive) learner(Y_enet):  pystacked $Y $X, method(elasticcv) type(reg)
ddml E[Y|X,D], mname(m_interactive) learner(Y_rf):    pystacked $Y $X, method(rf)        type(reg) njobs(-1)
ddml E[Y|X,D], mname(m_interactive) learner(Y_gb):    pystacked $Y $X, method(gradboost) type(class) njobs(-1)
ddml E[Y|X,D], mname(m_interactive) learner(Y_nnet):  pystacked $Y $X, method(nnet)      type(class)

* Stacked ensemble
ddml E[Y|X,D], mname(m_interactive) learner(Y_stack): ///
    pystacked $Y $X ///
        || method(logit) ///
        || method(lassocv) ///
        || method(ridgecv) ///
        || method(elasticcv) ///
        || method(rf) ///
        || method(gradboost) ///
		|| method(nnet), ///
    type(class) njobs(-1)

*--------------------------------------------------------------------
* E[D|X]
*--------------------------------------------------------------------
ddml E[D|X], mname(m_interactive) learner(D_ols):   reg   $D $X
ddml E[D|X], mname(m_interactive) learner(D_logit): logit $D $X
ddml E[D|X], mname(m_interactive) learner(D_lasso): pystacked $D $X, method(lassocv)  type(class)
ddml E[D|X], mname(m_interactive) learner(D_ridge): pystacked $D $X, method(ridgecv)  type(class)
ddml E[D|X], mname(m_interactive) learner(D_enet):  pystacked $D $X, method(elasticcv) type(class)
ddml E[D|X], mname(m_interactive) learner(D_rf):    pystacked $D $X, method(rf)        type(class) njobs(-1)
ddml E[D|X], mname(m_interactive) learner(D_gb):    pystacked $D $X, method(gradboost) type(class) njobs(-1)
ddml E[D|X], mname(m_interactive) learner(D_nnet):  pystacked $D $X, method(nnet)      type(class)

* Stacked ensemble
ddml E[D|X], mname(m_interactive) learner(D_stack): ///
    pystacked $D $X ///
        || method(logit) ///
        || method(lassocv) ///
        || method(ridgecv) ///
        || method(elasticcv) ///
        || method(rf) ///
        || method(gradboost) ///
		|| method(nnet), ///
    type(class) njobs(-1)

ddml crossfit, mname(m_interactive) shortstack
ddml estimate, mname(m_interactive) robust allcombos
************************************************************
* 4) Export comparison table (OLS vs DDML LASSO/RF)
************************************************************

eststo:  ddml extract, mname(m_interactive) vname($Y1) show(ssweights)
* ddml extract, mname(m_interactive) vname($D)  show(ssweights)
esttab  using "${Tables}ddml_interactive_SomeEcoli_bi.tex", replace ///
    title("Diarrhea: OLS vs. DDML (LASSO / RF) and BEST — Partial Effect of \$D\$ on \$Y\$") ///
    booktabs label compress nonotes mtitle("OLS" "LASSO" "Random Forest" "Best") ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    keep($D) ///
    stats(N nreps D_water_treatment_ss_mse Y_SomeRiskHome_ss1_mse Y_SomeRiskHome_ss0_mse, ///
	      fmt(%9.0fc %9.0fc %9.5fc %9.2fc  %9.2fc) labels(`"Observations"' `"Reps"'))	
eststo clear

													*====================================================================
													* 0. High risk (VeryHighRiskHome)
													*====================================================================

start_from_final
*====================================================================
* 1.1. VARIABLE DEFINITIONS
*====================================================================
* Outcome, Treatment, Control
local Reps 5

global Y VeryHighRiskHome
global D water_treatment
global X i.windex_ur i.helevel WQ27 i.country_cat i.urban i.WS1_g 
* i.HH51

reg   $Y $D $X

*====================================================================
* 1.2. DDML – PARTIAL LINEAR MODEL (Ecoli)
*====================================================================
ddml init partial, ///
    kfolds(10) ///
    reps(`Reps') ///
    mname(m_ecoli)

*--------------------------------------------------------------------
* E[Y | X]  — Outcome model (classification)
*--------------------------------------------------------------------
ddml E[Y|X], mname(m_ecoli) learner(Y_ols):   reg   $Y $X
*ddml E[Y|X], mname(m_ecoli) learner(Y_logit): logit $Y $X
ddml E[Y|X], mname(m_ecoli) learner(Y_lasso): pystacked $Y $X, method(lassocv)  type(reg)
ddml E[Y|X], mname(m_ecoli) learner(Y_ridge): pystacked $Y $X, method(ridgecv)  type(reg)
ddml E[Y|X], mname(m_ecoli) learner(Y_enet):  pystacked $Y $X, method(elasticcv) type(reg)
ddml E[Y|X], mname(m_ecoli) learner(Y_rf):    pystacked $Y $X, method(rf)        type(reg) njobs(-1)
ddml E[Y|X], mname(m_ecoli) learner(Y_gb):    pystacked $Y $X, method(gradboost) type(reg) njobs(-1)
ddml E[Y|X], mname(m_ecoli) learner(Y_nnet):  pystacked $Y $X, method(nnet)      type(reg)

* Stacked ensemble
ddml E[Y|X], mname(m_ecoli) learner(Y_stack): ///
    pystacked $Y $X ///
        || method(lassocv) ///
        || method(ridgecv) ///
        || method(elasticcv) ///
        || method(rf) ///
        || method(gradboost) ///
		|| method(nnet), ///
    type(reg) njobs(-1)

*--------------------------------------------------------------------
* E[D|X]
*--------------------------------------------------------------------
ddml E[D|X], mname(m_ecoli) learner(D_ols):   reg   $D $X
ddml E[D|X], mname(m_ecoli) learner(D_logit): logit $D $X
ddml E[D|X], mname(m_ecoli) learner(D_lasso): pystacked $D $X, method(lassocv)  type(class)
ddml E[D|X], mname(m_ecoli) learner(D_ridge): pystacked $D $X, method(ridgecv)  type(class)
ddml E[D|X], mname(m_ecoli) learner(D_enet):  pystacked $D $X, method(elasticcv) type(class)
ddml E[D|X], mname(m_ecoli) learner(D_rf):    pystacked $D $X, method(rf)        type(class) njobs(-1)
ddml E[D|X], mname(m_ecoli) learner(D_gb):    pystacked $D $X, method(gradboost) type(class) njobs(-1)
ddml E[D|X], mname(m_ecoli) learner(D_nnet):  pystacked $D $X, method(nnet)      type(class)

* Stacked ensemble
ddml E[D|X], mname(m_ecoli) learner(D_stack): ///
    pystacked $D $X ///
        || method(logit) ///
        || method(lassocv) ///
        || method(ridgecv) ///
        || method(elasticcv) ///
        || method(rf) ///
        || method(gradboost) ///
		|| method(nnet), ///
    type(class) njobs(-1)

*--------------------------------------------------------------------
* Cross-fitting & estimation
*--------------------------------------------------------------------
ddml crossfit, mname(m_ecoli) shortstack
eststo: ddml estimate, mname(m_ecoli) robust allcombos

************************************************************
* 4) Export comparison table (OLS vs DDML LASSO/RF)
************************************************************
esttab  using "${Tables}ddml_partial_Ecoli_bi.tex", replace ///
    title("Diarrhea: OLS vs. DDML (LASSO / RF) and BEST — Partial Effect of \$D\$ on \$Y\$") ///
    booktabs label compress nonotes mtitle("OLS" "LASSO" "Random Forest" "Best") ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    keep($D) ///
    stats(N rep D_water_treatment_ss_mse Y_VeryHighRiskHome_ss_mse, fmt(%9.0fc %9.0fc %9.4fc %9.2fc ) labels(`"Observations"' `"Reps"'))	
eststo clear

																		start_from_final
*====================================================================
* 1.1. VARIABLE DEFINITIONS
*====================================================================
* Outcome, Treatment, Control
global Y VeryHighRiskHome
global D water_treatment
global X i.windex_ur i.helevel WQ27 i.country_cat i.urban i.WS1_g 
reg   $Y $D $X

*====================================================================
* 9. DDML – INTERACTIVE MODEL (Ecoli)
*====================================================================
ddml init interactive, ///
	kfolds(10) /// 
	reps(`Reps') ///
	mname(m_interactive)

*--------------------------------------------------------------------
* E[Y | X, D] — Outcome models by treatment status
*--------------------------------------------------------------------
ddml E[Y|X,D], mname(m_interactive) learner(Y_ols):   reg   $Y $X
*ddml E[Y|X,D], mname(m_interactive) learner(Y_logit): logit $Y $X
ddml E[Y|X,D], mname(m_interactive) learner(Y_lasso): pystacked $Y $X, method(lassocv)  type(reg)
ddml E[Y|X,D], mname(m_interactive) learner(Y_ridge): pystacked $Y $X, method(ridgecv)  type(reg)
ddml E[Y|X,D], mname(m_interactive) learner(Y_enet):  pystacked $Y $X, method(elasticcv) type(reg)
ddml E[Y|X,D], mname(m_interactive) learner(Y_rf):    pystacked $Y $X, method(rf)        type(reg) njobs(-1)
ddml E[Y|X,D], mname(m_interactive) learner(Y_gb):    pystacked $Y $X, method(gradboost) type(class) njobs(-1)
ddml E[Y|X,D], mname(m_interactive) learner(Y_nnet):  pystacked $Y $X, method(nnet)      type(class)

* Stacked ensemble
ddml E[Y|X,D], mname(m_interactive) learner(Y_stack): ///
    pystacked $Y $X ///
        || method(logit) ///
        || method(lassocv) ///
        || method(ridgecv) ///
        || method(elasticcv) ///
        || method(rf) ///
        || method(gradboost) ///
		|| method(nnet), ///
    type(class) njobs(-1)

*--------------------------------------------------------------------
* E[D|X]
*--------------------------------------------------------------------
ddml E[D|X], mname(m_interactive) learner(D_ols):   reg   $D $X
ddml E[D|X], mname(m_interactive) learner(D_logit): logit $D $X
ddml E[D|X], mname(m_interactive) learner(D_lasso): pystacked $D $X, method(lassocv)  type(class)
ddml E[D|X], mname(m_interactive) learner(D_ridge): pystacked $D $X, method(ridgecv)  type(class)
ddml E[D|X], mname(m_interactive) learner(D_enet):  pystacked $D $X, method(elasticcv) type(class)
ddml E[D|X], mname(m_interactive) learner(D_rf):    pystacked $D $X, method(rf)        type(class) njobs(-1)
ddml E[D|X], mname(m_interactive) learner(D_gb):    pystacked $D $X, method(gradboost) type(class) njobs(-1)
ddml E[D|X], mname(m_interactive) learner(D_nnet):  pystacked $D $X, method(nnet)      type(class)

* Stacked ensemble
ddml E[D|X], mname(m_interactive) learner(D_stack): ///
    pystacked $D $X ///
        || method(logit) ///
        || method(lassocv) ///
        || method(ridgecv) ///
        || method(elasticcv) ///
        || method(rf) ///
        || method(gradboost) ///
		|| method(nnet), ///
    type(class) njobs(-1)

ddml crossfit, mname(m_interactive) shortstack
ddml estimate, mname(m_interactive) robust allcombos
************************************************************
* 4) Export comparison table (OLS vs DDML LASSO/RF)
************************************************************

eststo:  ddml extract, mname(m_interactive) vname($Y1) show(ssweights)
* ddml extract, mname(m_interactive) vname($D)  show(ssweights)
esttab  using "${Tables}ddml_interactive_Ecoli_bi.tex", replace ///
    title("Diarrhea: OLS vs. DDML (LASSO / RF) and BEST — Partial Effect of \$D\$ on \$Y\$") ///
    booktabs label compress nonotes mtitle("OLS" "LASSO" "Random Forest" "Best") ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    keep($D) ///
    stats(N rep D_water_treatment_ss_mse Y_VeryHighRiskHome_ss1_mse Y_VeryHighRiskHome_ss0_mse, ///
	      fmt(%9.0fc %9.0fc %9.5fc %9.2fc  %9.2fc) labels(`"Observations"' `"Reps"'))	
eststo clear


													*====================================================================
													* 2. Diarrhea outcomes
													*====================================================================

start_from_final
*====================================================================
* 1.1. VARIABLE DEFINITIONS
*====================================================================
* Outcome, Treatment, Control
local Reps 5

global Y diarrhea
global D water_treatment
global X i.windex_ur i.helevel WQ27 i.country_cat i.urban i.WS1_g i.HH51
reg   $Y $D $X

*====================================================================
* 1.2. DDML – PARTIAL LINEAR MODEL (Ecoli)
*====================================================================
ddml init partial, ///
    kfolds(10) ///
    reps(`Reps') ///
    mname(m_ecoli)

*--------------------------------------------------------------------
* E[Y | X]  — Outcome model (classification)
*--------------------------------------------------------------------
ddml E[Y|X], mname(m_ecoli) learner(Y_ols):   reg   $Y $X
*ddml E[Y|X], mname(m_ecoli) learner(Y_logit): logit $Y $X
ddml E[Y|X], mname(m_ecoli) learner(Y_lasso): pystacked $Y $X, method(lassocv)  type(reg)
ddml E[Y|X], mname(m_ecoli) learner(Y_ridge): pystacked $Y $X, method(ridgecv)  type(reg)
ddml E[Y|X], mname(m_ecoli) learner(Y_enet):  pystacked $Y $X, method(elasticcv) type(reg)
ddml E[Y|X], mname(m_ecoli) learner(Y_rf):    pystacked $Y $X, method(rf)        type(reg) njobs(-1)
ddml E[Y|X], mname(m_ecoli) learner(Y_gb):    pystacked $Y $X, method(gradboost) type(reg) njobs(-1)
ddml E[Y|X], mname(m_ecoli) learner(Y_nnet):  pystacked $Y $X, method(nnet)      type(reg)

* Stacked ensemble
ddml E[Y|X], mname(m_ecoli) learner(Y_stack): ///
    pystacked $Y $X ///
        || method(lassocv) ///
        || method(ridgecv) ///
        || method(elasticcv) ///
        || method(rf) ///
        || method(gradboost) ///
		|| method(nnet), ///
    type(reg) njobs(-1)

*--------------------------------------------------------------------
* E[D|X]
*--------------------------------------------------------------------
ddml E[D|X], mname(m_ecoli) learner(D_ols):   reg   $D $X
ddml E[D|X], mname(m_ecoli) learner(D_logit): logit $D $X
ddml E[D|X], mname(m_ecoli) learner(D_lasso): pystacked $D $X, method(lassocv)  type(class)
ddml E[D|X], mname(m_ecoli) learner(D_ridge): pystacked $D $X, method(ridgecv)  type(class)
ddml E[D|X], mname(m_ecoli) learner(D_enet):  pystacked $D $X, method(elasticcv) type(class)
ddml E[D|X], mname(m_ecoli) learner(D_rf):    pystacked $D $X, method(rf)        type(class) njobs(-1)
ddml E[D|X], mname(m_ecoli) learner(D_gb):    pystacked $D $X, method(gradboost) type(class) njobs(-1)
ddml E[D|X], mname(m_ecoli) learner(D_nnet):  pystacked $D $X, method(nnet)      type(class)

* Stacked ensemble
ddml E[D|X], mname(m_ecoli) learner(D_stack): ///
    pystacked $D $X ///
        || method(logit) ///
        || method(lassocv) ///
        || method(ridgecv) ///
        || method(elasticcv) ///
        || method(rf) ///
        || method(gradboost) ///
		|| method(nnet), ///
    type(class) njobs(-1)

*--------------------------------------------------------------------
* Cross-fitting & estimation
*--------------------------------------------------------------------
ddml crossfit, mname(m_ecoli) shortstack
eststo: ddml estimate, mname(m_ecoli) robust allcombos

************************************************************
* 4) Export comparison table (OLS vs DDML LASSO/RF)
************************************************************
esttab  using "${Tables}ddml_partial_diarrhea.tex", replace ///
    title("Diarrhea: OLS vs. DDML (LASSO / RF) and BEST — Partial Effect of \$D\$ on \$Y\$") ///
    booktabs label compress nonotes mtitle("OLS" "LASSO" "Random Forest" "Best") ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    keep($D) ///
    stats(N nreps D_water_treatment_ss_mse Y_diarrhea_ss_mse, fmt(%9.0fc %9.0fc %9.4fc %9.2fc ) labels(`"Observations"' `"Reps"'))	
eststo clear

start_from_final
*====================================================================
* 1.1. VARIABLE DEFINITIONS
*====================================================================
* Outcome, Treatment, Control
local Reps 5

global Y diarrhea
global D water_treatment
global X i.windex_ur i.helevel WQ27 i.country_cat i.urban i.WS1_g 
reg   $Y $D $X

*====================================================================
* 9. DDML – INTERACTIVE MODEL (Ecoli)
*====================================================================
ddml init interactive, ///
	kfolds(10) /// 
	reps(`Reps') ///
	mname(m_interactive)

*--------------------------------------------------------------------
* E[Y | X, D] — Outcome models by treatment status
*--------------------------------------------------------------------
ddml E[Y|X,D], mname(m_interactive) learner(Y_ols):   reg   $Y $X
*ddml E[Y|X,D], mname(m_interactive) learner(Y_logit): logit $Y $X
ddml E[Y|X,D], mname(m_interactive) learner(Y_lasso): pystacked $Y $X, method(lassocv)  type(reg)
ddml E[Y|X,D], mname(m_interactive) learner(Y_ridge): pystacked $Y $X, method(ridgecv)  type(reg)
ddml E[Y|X,D], mname(m_interactive) learner(Y_enet):  pystacked $Y $X, method(elasticcv) type(reg)
ddml E[Y|X,D], mname(m_interactive) learner(Y_rf):    pystacked $Y $X, method(rf)        type(reg) njobs(-1)
ddml E[Y|X,D], mname(m_interactive) learner(Y_gb):    pystacked $Y $X, method(gradboost) type(class) njobs(-1)
ddml E[Y|X,D], mname(m_interactive) learner(Y_nnet):  pystacked $Y $X, method(nnet)      type(class)

* Stacked ensemble
ddml E[Y|X,D], mname(m_interactive) learner(Y_stack): ///
    pystacked $Y $X ///
        || method(logit) ///
        || method(lassocv) ///
        || method(ridgecv) ///
        || method(elasticcv) ///
        || method(rf) ///
        || method(gradboost) ///
		|| method(nnet), ///
    type(class) njobs(-1)

*--------------------------------------------------------------------
* E[D|X]
*--------------------------------------------------------------------
ddml E[D|X], mname(m_interactive) learner(D_ols):   reg   $D $X
ddml E[D|X], mname(m_interactive) learner(D_logit): logit $D $X
ddml E[D|X], mname(m_interactive) learner(D_lasso): pystacked $D $X, method(lassocv)  type(class)
ddml E[D|X], mname(m_interactive) learner(D_ridge): pystacked $D $X, method(ridgecv)  type(class)
ddml E[D|X], mname(m_interactive) learner(D_enet):  pystacked $D $X, method(elasticcv) type(class)
ddml E[D|X], mname(m_interactive) learner(D_rf):    pystacked $D $X, method(rf)        type(class) njobs(-1)
ddml E[D|X], mname(m_interactive) learner(D_gb):    pystacked $D $X, method(gradboost) type(class) njobs(-1)
ddml E[D|X], mname(m_interactive) learner(D_nnet):  pystacked $D $X, method(nnet)      type(class)

* Stacked ensemble
ddml E[D|X], mname(m_interactive) learner(D_stack): ///
    pystacked $D $X ///
        || method(logit) ///
        || method(lassocv) ///
        || method(ridgecv) ///
        || method(elasticcv) ///
        || method(rf) ///
        || method(gradboost) ///
		|| method(nnet), ///
    type(class) njobs(-1)

ddml crossfit, mname(m_interactive) shortstack
ddml estimate, mname(m_interactive) robust allcombos
************************************************************
* 4) Export comparison table (OLS vs DDML LASSO/RF)
************************************************************

eststo:  ddml extract, mname(m_interactive) vname($Y1) show(ssweights)
* ddml extract, mname(m_interactive) vname($D)  show(ssweights)
esttab  using "${Tables}ddml_interactive_diarrhea.tex", replace ///
    title("Diarrhea: OLS vs. DDML (LASSO / RF) and BEST — Partial Effect of \$D\$ on \$Y\$") ///
    booktabs label compress nonotes mtitle("OLS" "LASSO" "Random Forest" "Best") ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    keep($D) ///
    stats(N nreps D_water_treatment_ss_mse Y_diarrhea_ss1_mse Y_diarrhea_ss0_mse, ///
	      fmt(%9.0fc %9.0fc %9.5fc %9.2fc  %9.2fc) labels(`"Observations"' `"Reps"'))	
eststo clear


END

													*====================================================================
													* 1. Water treatment
													*====================================================================
local Reps 5

start_from_final
*====================================================================
* 1.1. VARIABLE DEFINITIONS
*====================================================================
* Outcome, Treatment, Control
global Y WQ26
global D water_treatment
global X i.windex_ur i.helevel WQ27 i.country_cat i.urban i.WS1_g 
* i.HH51

reg   $Y $D $X

*====================================================================
* 1.2. DDML – PARTIAL LINEAR MODEL (Ecoli)
*====================================================================
ddml init partial, ///
    kfolds(10) ///
    reps(`Reps') ///
    mname(m_ecoli)

*--------------------------------------------------------------------
* E[Y | X]  — Outcome model (classification)
*--------------------------------------------------------------------
ddml E[Y|X], mname(m_ecoli) learner(Y_ols):   reg   $Y $X
*ddml E[Y|X], mname(m_ecoli) learner(Y_logit): logit $Y $X
ddml E[Y|X], mname(m_ecoli) learner(Y_lasso): pystacked $Y $X, method(lassocv)  type(reg)
ddml E[Y|X], mname(m_ecoli) learner(Y_ridge): pystacked $Y $X, method(ridgecv)  type(reg)
ddml E[Y|X], mname(m_ecoli) learner(Y_enet):  pystacked $Y $X, method(elasticcv) type(reg)
ddml E[Y|X], mname(m_ecoli) learner(Y_rf):    pystacked $Y $X, method(rf)        type(reg) njobs(-1)
ddml E[Y|X], mname(m_ecoli) learner(Y_gb):    pystacked $Y $X, method(gradboost) type(reg) njobs(-1)
ddml E[Y|X], mname(m_ecoli) learner(Y_nnet):  pystacked $Y $X, method(nnet)      type(reg)

* Stacked ensemble
ddml E[Y|X], mname(m_ecoli) learner(Y_stack): ///
    pystacked $Y $X ///
        || method(lassocv) ///
        || method(ridgecv) ///
        || method(elasticcv) ///
        || method(rf) ///
        || method(gradboost) ///
		|| method(nnet), ///
    type(reg) njobs(-1)

*--------------------------------------------------------------------
* E[D|X]
*--------------------------------------------------------------------
ddml E[D|X], mname(m_ecoli) learner(D_ols):   reg   $D $X
ddml E[D|X], mname(m_ecoli) learner(D_logit): logit $D $X
ddml E[D|X], mname(m_ecoli) learner(D_lasso): pystacked $D $X, method(lassocv)  type(class)
ddml E[D|X], mname(m_ecoli) learner(D_ridge): pystacked $D $X, method(ridgecv)  type(class)
ddml E[D|X], mname(m_ecoli) learner(D_enet):  pystacked $D $X, method(elasticcv) type(class)
ddml E[D|X], mname(m_ecoli) learner(D_rf):    pystacked $D $X, method(rf)        type(class) njobs(-1)
ddml E[D|X], mname(m_ecoli) learner(D_gb):    pystacked $D $X, method(gradboost) type(class) njobs(-1)
ddml E[D|X], mname(m_ecoli) learner(D_nnet):  pystacked $D $X, method(nnet)      type(class)

* Stacked ensemble
ddml E[D|X], mname(m_ecoli) learner(D_stack): ///
    pystacked $D $X ///
        || method(logit) ///
        || method(lassocv) ///
        || method(ridgecv) ///
        || method(elasticcv) ///
        || method(rf) ///
        || method(gradboost) ///
		|| method(nnet), ///
    type(class) njobs(-1)

*--------------------------------------------------------------------
* Cross-fitting & estimation
*--------------------------------------------------------------------
ddml crossfit, mname(m_ecoli) shortstack
eststo: ddml estimate, mname(m_ecoli) robust allcombos

************************************************************
* 4) Export comparison table (OLS vs DDML LASSO/RF)
************************************************************
esttab  using "${Tables}ddml_partial_Ecoli.tex", replace ///
    title("Diarrhea: OLS vs. DDML (LASSO / RF) and BEST — Partial Effect of \$D\$ on \$Y\$") ///
    booktabs label compress nonotes mtitle("OLS" "LASSO" "Random Forest" "Best") ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    keep($D) ///
    stats(N rep D_water_treatment_ss_mse Y_WQ26_ss_mse, fmt(%9.0fc %9.0fc %9.5fc %9.1fc ) labels(`"Observations"' `"Reps"'))	
eststo clear

																		start_from_final
*====================================================================
* 1.1. VARIABLE DEFINITIONS
*====================================================================
* Outcome, Treatment, Control
global Y WQ26
global D water_treatment
global X i.windex_ur i.helevel WQ27 i.country_cat i.urban i.WS1_g 
reg   $Y $D $X

*====================================================================
* 9. DDML – INTERACTIVE MODEL (Ecoli)
*====================================================================
ddml init interactive, ///
	kfolds(10) /// 
	reps(`Reps') ///
	mname(m_interactive)

*--------------------------------------------------------------------
* E[Y | X, D] — Outcome models by treatment status
*--------------------------------------------------------------------
ddml E[Y|X,D], mname(m_interactive) learner(Y_ols):   reg   $Y $X
*ddml E[Y|X,D], mname(m_interactive) learner(Y_logit): logit $Y $X
ddml E[Y|X,D], mname(m_interactive) learner(Y_lasso): pystacked $Y $X, method(lassocv)  type(reg)
ddml E[Y|X,D], mname(m_interactive) learner(Y_ridge): pystacked $Y $X, method(ridgecv)  type(reg)
ddml E[Y|X,D], mname(m_interactive) learner(Y_enet):  pystacked $Y $X, method(elasticcv) type(reg)
ddml E[Y|X,D], mname(m_interactive) learner(Y_rf):    pystacked $Y $X, method(rf)        type(reg) njobs(-1)
ddml E[Y|X,D], mname(m_interactive) learner(Y_gb):    pystacked $Y $X, method(gradboost) type(class) njobs(-1)
ddml E[Y|X,D], mname(m_interactive) learner(Y_nnet):  pystacked $Y $X, method(nnet)      type(class)

* Stacked ensemble
ddml E[Y|X,D], mname(m_interactive) learner(Y_stack): ///
    pystacked $Y $X ///
        || method(logit) ///
        || method(lassocv) ///
        || method(ridgecv) ///
        || method(elasticcv) ///
        || method(rf) ///
        || method(gradboost) ///
		|| method(nnet), ///
    type(class) njobs(-1)

*--------------------------------------------------------------------
* E[D|X]
*--------------------------------------------------------------------
ddml E[D|X], mname(m_interactive) learner(D_ols):   reg   $D $X
ddml E[D|X], mname(m_interactive) learner(D_logit): logit $D $X
ddml E[D|X], mname(m_interactive) learner(D_lasso): pystacked $D $X, method(lassocv)  type(class)
ddml E[D|X], mname(m_interactive) learner(D_ridge): pystacked $D $X, method(ridgecv)  type(class)
ddml E[D|X], mname(m_interactive) learner(D_enet):  pystacked $D $X, method(elasticcv) type(class)
ddml E[D|X], mname(m_interactive) learner(D_rf):    pystacked $D $X, method(rf)        type(class) njobs(-1)
ddml E[D|X], mname(m_interactive) learner(D_gb):    pystacked $D $X, method(gradboost) type(class) njobs(-1)
ddml E[D|X], mname(m_interactive) learner(D_nnet):  pystacked $D $X, method(nnet)      type(class)

* Stacked ensemble
ddml E[D|X], mname(m_interactive) learner(D_stack): ///
    pystacked $D $X ///
        || method(logit) ///
        || method(lassocv) ///
        || method(ridgecv) ///
        || method(elasticcv) ///
        || method(rf) ///
        || method(gradboost) ///
		|| method(nnet), ///
    type(class) njobs(-1)

ddml crossfit, mname(m_interactive) shortstack
ddml estimate, mname(m_interactive) robust allcombos
************************************************************
* 4) Export comparison table (OLS vs DDML LASSO/RF)
************************************************************

eststo:  ddml extract, mname(m_interactive) vname($Y1) show(ssweights)
* ddml extract, mname(m_interactive) vname($D)  show(ssweights)
esttab  using "${Tables}ddml_interactive_Ecoli.tex", replace ///
    title("Diarrhea: OLS vs. DDML (LASSO / RF) and BEST — Partial Effect of \$D\$ on \$Y\$") ///
    booktabs label compress nonotes mtitle("OLS" "LASSO" "Random Forest" "Best") ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    keep($D) ///
    stats(N rep D_water_treatment_ss_mse Y_WQ26_ss1_mse Y_WQ26_ss0_mse, fmt(%9.0fc %9.0fc %9.5fc %9.1fc  %9.1fc) labels(`"Observations"' `"Reps"'))	
eststo clear




STOP AKITO

