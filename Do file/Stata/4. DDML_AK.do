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
if c(username) == "akitokamei" {
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

*===============================================================
* Household level data
*===============================================================

cap program drop start_from_final
program define   start_from_final
use "${Data_Final}MASTER_MICS_FINAL.dta", clear

end

*===============================================================
* Diarrhea (U5 children)
*===============================================================

cap program drop start_from_final_child
program define   start_from_final_child
use "${Data_Final}MASTER_MICS_FINAL_U5.dta", clear

end

local diarrhea      "Probability of under 5 children having diarrhea"
local fever         "Probability of under 5 children having fever"
local notediarrhea  "Notes: $\sym{*} p<0.10,\sym{**} p<0.05,\sym{***} p<0.01$."
local notefever     "Notes: $\sym{*} p<0.10,\sym{**} p<0.05,\sym{***} p<0.01$."
local Labeldiarrhea "diarrhea"
local Labelfever    "fever"

************************************************
* Panel B: Controls
************************************************
start_from_final_child
global Controls i.windex5 i.helevel i.country_cat i.urban i.WS1_g ///
                Any_U5 Girls_less_than15 Boys_15or_less i.Toilet i.wq27_decile
global Controls_U5 male i.age

*------------------------------------------------------------ By department ------------------------------------------------------------*

************************************************************
* 3) DDML (Household level):
************************************************************
foreach Dependent in SomeRiskHome VeryHighRiskHome {
start_from_final
gen random = runiform()
drop if random<0.95

* Globals
global Y `Dependent'
global D i.WQ15_g
global X i.windex5 i.helevel i.country_cat i.urban i.WS1_g Any_U5 Girls_less_than15 Boys_15or_less i.Toilet i.wq27_decile

	qddml $Y $D ($X), model(interactive) mname(m_stack) cmd(pystacked)  ycmdopt(type(reg) method(ols lassocv rf ridgecv gradboost)) ///
	                                                       dcmdopt(type(class) method(logit lassocv rf ridgecv gradboost)) reps(3) kfolds(3)

	*---------------------------------------------------------*
    * Save stacking weights for this department
    *---------------------------------------------------------*
    capture log close weightslog
	set trace off
    log using "${Tables}DDML_WT_weights.log", replace name(weightslog)

	* Saved macro
	ddml extract, mname(m_stack)
	ds *pystack*

	* Creating the
	capture drop mhat1 mhat0 ehat

	egen double mhat1 = rowmean(Y1_pystacked1_1 Y1_pystacked1_2 Y1_pystacked1_3)
	egen double mhat0 = rowmean(Y1_pystacked0_1 Y1_pystacked0_2 Y1_pystacked0_3)
	egen double ehat  = rowmean(D1_pystacked_1  D1_pystacked_2  D1_pystacked_3)

	summ mhat1 mhat0 ehat, detail

	*** Overview: which base learners and how they're weighted in each equation
	ddml extract, mname(m_stack) show(pystacked)

    log close weightslog

	*------------------------------------------------------------*
	* ATE
	*------------------------------------------------------------*
	* Estimate partial effect and store for table
    eststo DDML_BEST: ddml estimate, mname(m_stack) notable replay

	*------------------------------------------------------------*
	* ATET
	*------------------------------------------------------------*
	eststo DDML_ATET: ddml estimate, mname(m_stack) notable atet

	* === Build OOF predictions by averaging across folds ===
	capture drop Yhat Dhat
	egen double Yhat = rowmean(Y1_pystack*)
	egen double Dhat = rowmean(D1_pystack*)

	* === RMSE for Y ===
	capture drop __e2y __e2d
	gen double __e2y = ($Y - Yhat)^2 if e(sample)
	quietly summarize __e2y if e(sample)
	local rmsey = sqrt(r(mean))
	estadd scalar RMSE_Y = `rmsey'

	estimates restore DDML_BEST`i'
	gen double __e2d = ($D - Dhat)^2 if e(sample)
	quietly summarize __e2d if e(sample)
	local rmsed = sqrt(r(mean))

	eststo DDML_BEST, addscalars(RMSE_Y `rmsey' RMSE_D `rmsed', replace)

************************************************************
* 4) Export comparison table (OLS vs DDML LASSO/RF)
************************************************************
esttab DDML_BEST DDML_ATET using "${Tables}ddml_Water_`Dependent'_reg_method.csv", se star(* .10 ** .05 *** .01)  replace
}

END




************************************************************
* Setup
************************************************************
eststo clear
set seed 12345
start_from_final_child

************************************************************
* 3) DDML BEST:
************************************************************
foreach Dependent in diarrhea {
start_from_final_child
* gen random = runiform()
* drop if random<0.95

* Globals
global Y `Dependent'
global D water_treatment
global X i.windex5 i.helevel i.country_cat i.urban i.WS1_g Any_U5 Girls_less_than15 Boys_15or_less i.Toilet i.wq27_decile male i.age

	qddml $Y $D ($X), model(interactive) mname(m_stack) cmd(pystacked)  ycmdopt(type(reg) method(ols lassocv rf ridgecv gradboost)) ///
	                                                       dcmdopt(type(class) method(logit lassocv rf ridgecv gradboost)) reps(3) kfolds(3)

	*---------------------------------------------------------*
    * Save stacking weights for this department
    *---------------------------------------------------------*
    capture log close weightslog
	set trace off
    log using "${Tables}DDML_WT_weights`Dependent'.log", replace name(weightslog)

	* Saved macro
	ddml extract, mname(m_stack)
	ds *pystack*

	* Creating the
	capture drop mhat1 mhat0 ehat

	egen double mhat1 = rowmean(Y1_pystacked1_1 Y1_pystacked1_2 Y1_pystacked1_3)
	egen double mhat0 = rowmean(Y1_pystacked0_1 Y1_pystacked0_2 Y1_pystacked0_3)
	egen double ehat  = rowmean(D1_pystacked_1  D1_pystacked_2  D1_pystacked_3)

	summ mhat1 mhat0 ehat, detail

	*** Overview: which base learners and how they're weighted in each equation
	ddml extract, mname(m_stack) show(pystacked)

    log close weightslog

	*------------------------------------------------------------*
	* ATE
	*------------------------------------------------------------*
	* Estimate partial effect and store for table
    eststo DDML_BEST: ddml estimate, mname(m_stack) notable replay

	*------------------------------------------------------------*
	* ATET
	*------------------------------------------------------------*
	eststo DDML_ATET: ddml estimate, mname(m_stack) notable atet

	* === Build OOF predictions by averaging across folds ===
	capture drop Yhat Dhat
	egen double Yhat = rowmean(Y1_pystack*)
	egen double Dhat = rowmean(D1_pystack*)

	* === RMSE for Y ===
	capture drop __e2y __e2d
	gen double __e2y = ($Y - Yhat)^2 if e(sample)
	quietly summarize __e2y if e(sample)
	local rmsey = sqrt(r(mean))
	estadd scalar RMSE_Y = `rmsey'

	estimates restore DDML_BEST
	gen double __e2d = ($D - Dhat)^2 if e(sample)
	quietly summarize __e2d if e(sample)
	local rmsed = sqrt(r(mean))

	eststo DDML_BEST, addscalars(RMSE_Y `rmsey' RMSE_D `rmsed', replace)

************************************************************
* 4) Export comparison table (OLS vs DDML LASSO/RF)
************************************************************
esttab DDML_BEST DDML_ATET using "${Tables}ddml_Water_`Dependent'_reg.csv", se star(* .10 ** .05 *** .01)  replace
}

END


************************************************************
* 3) DDML (Household level):
************************************************************
foreach Dependent in SomeRiskHome VeryHighRiskHome {
start_from_final
* gen random = runiform()
* drop if random<0.95

* Globals
global Y `Dependent'
global D water_treatment
global X i.windex5 i.helevel i.country_cat i.urban i.WS1_g Any_U5 Girls_less_than15 Boys_15or_less i.Toilet i.wq27_decile

	qddml $Y $D ($X), model(interactive) mname(m_stack) cmd(pystacked)  ycmdopt(type(reg) method(ols lassocv rf ridgecv gradboost)) ///
	                                                       dcmdopt(type(class) method(logit lassocv rf ridgecv gradboost)) reps(3) kfolds(3)

	*---------------------------------------------------------*
    * Save stacking weights for this department
    *---------------------------------------------------------*
    capture log close weightslog
	set trace off
    log using "${Tables}DDML_WT_weights`i'.log", replace name(weightslog)

	* Saved macro
	ddml extract, mname(m_stack)
	ds *pystack*

	* Creating the
	capture drop mhat1 mhat0 ehat

	egen double mhat1 = rowmean(Y1_pystacked1_1 Y1_pystacked1_2 Y1_pystacked1_3)
	egen double mhat0 = rowmean(Y1_pystacked0_1 Y1_pystacked0_2 Y1_pystacked0_3)
	egen double ehat  = rowmean(D1_pystacked_1  D1_pystacked_2  D1_pystacked_3)

	summ mhat1 mhat0 ehat, detail

	*** Overview: which base learners and how they're weighted in each equation
	ddml extract, mname(m_stack) show(pystacked)

    log close weightslog

	*------------------------------------------------------------*
	* ATE
	*------------------------------------------------------------*
	* Estimate partial effect and store for table
    eststo DDML_BEST_`i': ddml estimate, mname(m_stack) notable replay

	*------------------------------------------------------------*
	* ATET
	*------------------------------------------------------------*
	eststo DDML_ATET_`i': ddml estimate, mname(m_stack) notable atet

	* === Build OOF predictions by averaging across folds ===
	capture drop Yhat Dhat
	egen double Yhat = rowmean(Y1_pystack*)
	egen double Dhat = rowmean(D1_pystack*)

	* === RMSE for Y ===
	capture drop __e2y __e2d
	gen double __e2y = ($Y - Yhat)^2 if e(sample)
	quietly summarize __e2y if e(sample)
	local rmsey = sqrt(r(mean))
	estadd scalar RMSE_Y = `rmsey'

	estimates restore DDML_BEST_`i'
	gen double __e2d = ($D - Dhat)^2 if e(sample)
	quietly summarize __e2d if e(sample)
	local rmsed = sqrt(r(mean))

	eststo DDML_BEST_`i', addscalars(RMSE_Y `rmsey' RMSE_D `rmsed', replace)

************************************************************
* 4) Export comparison table (OLS vs DDML LASSO/RF)
************************************************************
esttab DDML_BEST_* DDML_ATET_* using "${Tables}ddml_Water_`Dependent'_reg.csv", se star(* .10 ** .05 *** .01)  replace
}

END

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

