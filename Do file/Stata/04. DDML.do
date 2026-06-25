*--------------------------------------------------------------------
* Project: MICS – Impact of Water Treatment
* File:    DDML_Final.do
* Purpose: Estimate causal effect of household water treatment
*          on childhood diarrhea using Double/Debiased ML (DDML)
* Author:  Juan Álvaro
*--------------------------------------------------------------------

/*
Comments:
This version uses two models from the DDML package: PLM and Interactive. For this initial analysis, I (JA) took Akito's control variables from 3.Analysis.do and added a bunch of different learners to see the possible results. I saw that all of this can be recreated using only pystacked with Stata and then residualizing. This is useful since pystacked has elements that allow to graph performance plots.
Right now, the learners are:
- OLS
- Logit
- Lasso
- Ridge
- Elastic Net
- Random Forest
- Gradient Boost

Next steps:
Run all of this with Ecoli presence (either WQ27 or VeryHighRisk)
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

*====================================================================
* 2. LOAD DATA & (OPTIONAL) TEST SAMPLE
*====================================================================
use "${Data_Final}MASTER_MICS_U5_DDML.dta", clear

*====================================================================
* 3. DATA CLEANING & RECODING
*====================================================================
* Urban/rural wealth index harmonization
gen windex_ur = cond(missing(windex5u), windex5r, windex5u)

* Education: collapse into 3 categories
recode helevel (2/4 = 2)
label define helevell ///
    0 "No education" ///
    1 "Primary" ///
    2 "Secondary or higher", modify
label values helevel helevell

*====================================================================
* 4. VARIABLE DEFINITIONS
*====================================================================
* Outcome, Treatment, Control
global Y diarrhea
global D water_treatment
global X WQ27 i.windex_ur i.helevel

* IMPORTANT:
* ddml interactive requires NO missing values in Y, D, or X
keep if !missing($Y, $D)

foreach v in urban windex_ur helevel WQ11 {
    drop if missing(`v')
}

* NOTE: Sampling is for debugging/testing only
* sample 100

*====================================================================
* 5. DDML – PARTIAL LINEAR MODEL (Diarrhea)
*====================================================================
ddml init partial, ///
    kfolds(5) ///
    reps(1) ///
    mname(m_diarrhea)

*--------------------------------------------------------------------
* E[Y | X] — Outcome model (classification)
*--------------------------------------------------------------------
ddml E[Y|X], mname(m_diarrhea) learner(Y_ols):   reg   $Y $X
ddml E[Y|X], mname(m_diarrhea) learner(Y_logit): logit $Y $X
ddml E[Y|X], mname(m_diarrhea) learner(Y_lasso): pystacked $Y $X, method(lassocv) type(class)
ddml E[Y|X], mname(m_diarrhea) learner(Y_ridge): pystacked $Y $X, method(ridgecv) type(class)
ddml E[Y|X], mname(m_diarrhea) learner(Y_enet):  pystacked $Y $X, method(elasticcv) type(class)
ddml E[Y|X], mname(m_diarrhea) learner(Y_rf):    pystacked $Y $X, method(rf)        type(class) njobs(-1)
ddml E[Y|X], mname(m_diarrhea) learner(Y_gb):    pystacked $Y $X, method(gradboost) type(class) njobs(-1)
ddml E[Y|X], mname(m_diarrhea) learner(Y_nnet):  pystacked $Y $X, method(nnet)      type(class)

* Stacked ensemble
ddml E[Y|X], mname(m_diarrhea) learner(Y_stack): ///
    pystacked $Y $X ///
        || method(logit) ///
        || method(lassocv) ///
        || method(rf) ///
        || method(gradboost), ///
    type(class) njobs(-1)

*--------------------------------------------------------------------
* E[D|X]
*--------------------------------------------------------------------
ddml E[D|X], mname(m_diarrhea) learner(D_ols):   reg   $D $X
ddml E[D|X], mname(m_diarrhea) learner(D_logit): logit $D $X
ddml E[D|X], mname(m_diarrhea) learner(D_lasso): pystacked $D $X, method(lassocv) type(class)
ddml E[D|X], mname(m_diarrhea) learner(D_ridge): pystacked $D $X, method(ridgecv) type(class)
ddml E[D|X], mname(m_diarrhea) learner(D_enet):  pystacked $D $X, method(elasticcv) type(class)
ddml E[D|X], mname(m_diarrhea) learner(D_rf):    pystacked $D $X, method(rf)        type(class) njobs(-1)
ddml E[D|X], mname(m_diarrhea) learner(D_gb):    pystacked $D $X, method(gradboost) type(class) njobs(-1)
ddml E[D|X], mname(m_diarrhea) learner(D_nnet):  pystacked $D $X, method(nnet)      type(class)

* Stacked ensemble
ddml E[D|X], mname(m_diarrhea) learner(D_stack): ///
    pystacked $D $X ///
        || method(logit) ///
        || method(lassocv) ///
        || method(rf) ///
        || method(gradboost), ///
    type(class) njobs(-1)

*--------------------------------------------------------------------
* Cross-fitting & estimation (Fix: removed shortstack)
*--------------------------------------------------------------------
ddml crossfit, mname(m_diarrhea)
ddml estimate, mname(m_diarrhea) robust allcombos

*====================================================================
* 6. DDML – INTERACTIVE MODEL (Diarrhea)
*====================================================================
ddml init interactive, ///
    kfolds(5) /// 
    reps(1) /// 
    mname(m_interactive)

*--------------------------------------------------------------------
* E[Y | X, D] — Outcome models by treatment status
*--------------------------------------------------------------------
ddml E[Y|X,D], mname(m_interactive) learner(Y_ols):   reg   $Y $X
ddml E[Y|X,D], mname(m_interactive) learner(Y_logit): logit $Y $X
ddml E[Y|X,D], mname(m_interactive) learner(Y_lasso): pystacked $Y $X, method(lassocv) type(reg)
ddml E[Y|X,D], mname(m_interactive) learner(Y_ridge): pystacked $Y $X, method(ridgecv) type(reg)
ddml E[Y|X,D], mname(m_interactive) learner(Y_enet):  pystacked $Y $X, method(elasticcv) type(reg)
ddml E[Y|X,D], mname(m_interactive) learner(Y_rf):    pystacked $Y $X, method(rf)        type(reg) njobs(-1)
ddml E[Y|X,D], mname(m_interactive) learner(Y_gb):    pystacked $Y $X, method(gradboost) type(class) njobs(-1)
ddml E[Y|X,D], mname(m_interactive) learner(Y_nnet):  pystacked $Y $X, method(nnet)      type(class)

* Stacked ensemble
ddml E[Y|X,D], mname(m_interactive) learner(Y_stack): ///
    pystacked $Y $X ///
        || method(logit) ///
        || method(lassocv) ///
        || method(rf) ///
        || method(gradboost), ///
    type(class) njobs(-1)

*--------------------------------------------------------------------
* E[D|X]
*--------------------------------------------------------------------
ddml E[D|X], mname(m_interactive) learner(D_ols):   reg   $D $X
ddml E[D|X], mname(m_interactive) learner(D_logit): logit $D $X
ddml E[D|X], mname(m_interactive) learner(D_lasso): pystacked $D $X, method(lassocv) type(class)
ddml E[D|X], mname(m_interactive) learner(D_ridge): pystacked $D $X, method(ridgecv) type(class)
ddml E[D|X], mname(m_interactive) learner(D_enet):  pystacked $D $X, method(elasticcv) type(class)
ddml E[D|X], mname(m_interactive) learner(D_rf):    pystacked $D $X, method(rf)        type(class) njobs(-1)
ddml E[D|X], mname(m_interactive) learner(D_gb):    pystacked $D $X, method(gradboost) type(class) njobs(-1)
ddml E[D|X], mname(m_interactive) learner(D_nnet):  pystacked $D $X, method(nnet)      type(class)

* Stacked ensemble
ddml E[D|X], mname(m_interactive) learner(D_stack): ///
    pystacked $D $X ///
        || method(logit) ///
        || method(lassocv) ///
        || method(rf) ///
        || method(gradboost), ///
    type(class) njobs(-1)

ddml crossfit, mname(m_interactive)
ddml estimate, mname(m_interactive) robust allcombos

ddml extract, mname(m_interactive) vname($Y1) show(ssweights)
ddml extract, mname(m_interactive) vname($D) show(ssweights)
