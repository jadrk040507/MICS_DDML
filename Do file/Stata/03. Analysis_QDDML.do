*--------------------------------------------------------------------
* Project: 
* File Name: Descriptive Statistics
* Last updated: Akito on XXX
*--------------------------------------------------------------------

/*--------------------------------------------------------------------------------
    0 General program setup
-------------------------------------------------------------------------------*/

	clear               all
	capture log         close _all
	set more            off
	set varabbrev       off
	set emptycells      drop
	set seed            12345
	*set maxvar         2048
	set linesize        135	
						  
/*------------------------------------------------------------------------------
	1 Select parts of the code to run
------------------------------------------------------------------------------*/
	
	local import		0
	local deidentify	0
	local clean			0
	local tidy			0
	local construct		0
	local analyze		0
	
/*------------------------------------------------------------------------------
	2 Set file paths
------------------------------------------------------------------------------*/

	* Enter the file path to the project folder in Box for every new machine you use
	* Type 'di c(username)' to see the name of your machine
	
	else if c(username) == "akitokamei" {		
		global Dropbox "/Users/akitokamei/Library/CloudStorage/Dropbox/"
		global Overleaf "${Dropbox}Apps/Overleaf/"	
	}
	
	else if c(username) == "jadrk" {		
		global Dropbox "C:/Users/jadrk/Dropbox/"
		* global Overleaf "----"			
	}
	
	global Tables      "${Overleaf}MICS_DDML/Table/"
	global Figures     "${Overleaf}MICS_DDML/Figure/"
	* global Data_Raw   "${Dropbox}MICS_DDML/Data/1. Raw/"
	global Data_Clean "${Dropbox}MICS_DDML/Data/2. Clean/"
	global Data_Final "${Dropbox}MICS_DDML/Data/3. Final/"

clear all               
set graph off
set graph on	

*------------------------------------------------------------ Final data creation ------------------------------------------------------------*
cap program drop start_from_final
program define   start_from_final

* Open clean file
use "${Data_Final}MASTER_MICS_FINAL.dta", clear

end


************************************************************************ E-Coli ************************************************************************
start_from_final
************************************************************
* Setup
************************************************************
eststo clear

* Globals
global Y WQ26
global D water_treatment
global X WQ27 i.windex_ur i.helevel

************************************************************
* 1) OLS baseline + diagnostics
************************************************************
eststo OLS: reg $Y $D $X
	
************************************************************
* 2) DDML with two base learners via pystacked (LASSO and Random Forest)
************************************************************
	local methods "lassocv rf"
		foreach m of local methods {
		local mname = "m_`m'"
		local TAG   : display upper("`m'")

* Define DDML model (partialling-out) with chosen base learner
	* could be 5 and 10
    qddml $Y $D ($X), ///
        kfolds(5) model(interactive) mname(`mname') ///
        cmd(pystacked) cmdopt(type(reg) method(`m')) ///
		reps(5) 
		
    * Describe model structure & folds
    ddml describe, mname(`mname')
		
    * If RF, also save RF-specific info (e.g., feature importance if provided)
    if "`m'"=="rf" {
        capture ddml extract, mname(`mname') show(rf) detail ///
            saving("${Tables}diag_rf_detail.txt", replace)
    }

    * Estimate partial effect and store for table
    eststo DDML_`TAG': ddml estimate, mname(`mname') notable replay
}

************************************************************
* 3) DDML BEST: Combining the best performance one
************************************************************

	qddml $Y $D ($X), ///
		kfolds(5) model(interactive) mname(m_stack) ///
		cmd(pystacked) cmdopt(type(reg) method(lassocv rf)) ///
		reps(5) 
		
	* Overview: which base learners and how they're weighted in each equation
	ddml extract, mname(m_stack) show(pystacked)

    * Describe model structure & folds
    ddml describe, mname(m_stack)
	
	* Estimate partial effect and store for table
    eststo DDML_BEST: ddml estimate, mname(m_stack) notable replay

************************************************************
* 4) Export comparison table (OLS vs DDML LASSO/RF)
************************************************************
esttab OLS DDML_LASSOCV DDML_RF DDML_BEST using "${Table}ddml_compare_E_COLI.tex", replace ///
    title("E-Coli: OLS vs. DDML (LASSO / RF) and BEST — Partial Effect of \$D\$ on \$Y\$") ///
    booktabs label compress nonotes mtitle("OLS" "LASSO" "Random Forest" "Best") ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    keep($D) ///
    stats(N, fmt(%9.0fc) labels(`"Observations"'))	
	
	
	
************************************************************************ Diarrhea (U5) ************************************************************************
start_from_final_U5
************************************************************
* Setup
************************************************************
eststo clear

* Globals
global Y diarrhea
global D water_treatment
global X WQ27 i.windex_ur i.helevel

************************************************************
* 1) OLS baseline + diagnostics
************************************************************
eststo OLS: reg $Y $D $X
	
************************************************************
* 2) DDML with two base learners via pystacked (LASSO and Random Forest)
************************************************************
	local methods "lassocv rf"
		foreach m of local methods {
		local mname = "m_`m'"
		local TAG   : display upper("`m'")

* Define DDML model (partialling-out) with chosen base learner
	* could be 5 and 10
    qddml $Y $D ($X), ///
        kfolds(5) model(interactive) mname(`mname') ///
        cmd(pystacked) cmdopt(type(reg) method(`m')) ///
		reps(5) 
		
    * Describe model structure & folds
    ddml describe, mname(`mname')
		
    * If RF, also save RF-specific info (e.g., feature importance if provided)
    if "`m'"=="rf" {
        capture ddml extract, mname(`mname') show(rf) detail ///
            saving("${Tables}diag_rf_detail.txt", replace)
    }

    * Estimate partial effect and store for table
    eststo DDML_`TAG': ddml estimate, mname(`mname') notable replay
}

************************************************************
* 3) DDML BEST: Combining the best performance one
************************************************************

	qddml $Y $D ($X), ///
		kfolds(5) model(interactive) mname(m_stack) ///
		cmd(pystacked) cmdopt(type(reg) method(lassocv rf)) ///
		reps(5) 
		
	* Overview: which base learners and how they're weighted in each equation
	ddml extract, mname(m_stack) show(pystacked)

    * Describe model structure & folds
    ddml describe, mname(m_stack)
	
	* Estimate partial effect and store for table
    eststo DDML_BEST: ddml estimate, mname(m_stack) notable replay

************************************************************
* 4) Export comparison table (OLS vs DDML LASSO/RF)
************************************************************
esttab OLS DDML_LASSOCV DDML_RF DDML_BEST using "${Tables}ddml_compare_Diarrhea_T.tex", replace ///
    title("Diarrhea: OLS vs. DDML (LASSO / RF) and BEST — Partial Effect of \$D\$ on \$Y\$") ///
    booktabs label compress nonotes mtitle("OLS" "LASSO" "Random Forest" "Best") ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    keep($D) ///
    stats(N, fmt(%9.0fc) labels(`"Observations"'))	
	

	
	EDMN
	

	

************************************************************************ Diarrhea (U5) ************************************************************************
start_from_final_U5
************************************************************
* Setup
************************************************************
eststo clear

* Globals
global Y diarrhea
global D VeryHighRiskHome
global X water_treatment WQ27 i.windex_ur i.helevel

************************************************************
* 1) OLS baseline + diagnostics
************************************************************
eststo OLS: reg $Y $D $X
	
************************************************************
* 2) DDML with two base learners via pystacked (LASSO and Random Forest)
************************************************************
	local methods "lassocv rf"
		foreach m of local methods {
		local mname = "m_`m'"
		local TAG   : display upper("`m'")

* Define DDML model (partialling-out) with chosen base learner
	* could be 5 and 10
    qddml $Y $D ($X), ///
        kfolds(5) model(interactive) mname(`mname') ///
        cmd(pystacked) cmdopt(type(reg) method(`m')) ///
		reps(5) 
		
    * Describe model structure & folds
    ddml describe, mname(`mname')
		
    * If RF, also save RF-specific info (e.g., feature importance if provided)
    if "`m'"=="rf" {
        capture ddml extract, mname(`mname') show(rf) detail ///
            saving("${Tables}diag_rf_detail.txt", replace)
    }

    * Estimate partial effect and store for table
    eststo DDML_`TAG': ddml estimate, mname(`mname') notable replay
}

************************************************************
* 3) DDML BEST: Combining the best performance one
************************************************************

	qddml $Y $D ($X), ///
		kfolds(5) model(interactive) mname(m_stack) ///
		cmd(pystacked) cmdopt(type(reg) method(lassocv rf)) ///
		reps(5) 
		
	* Overview: which base learners and how they're weighted in each equation
	ddml extract, mname(m_stack) show(pystacked)

    * Describe model structure & folds
    ddml describe, mname(m_stack)
	
	* Estimate partial effect and store for table
    eststo DDML_BEST: ddml estimate, mname(m_stack) notable replay

************************************************************
* 4) Export comparison table (OLS vs DDML LASSO/RF)
************************************************************
esttab OLS DDML_LASSOCV DDML_RF DDML_BEST using "${Tables}ddml_compare_Diarrhea.tex", replace ///
    title("Diarrhea: OLS vs. DDML (LASSO / RF) and BEST — Partial Effect of \$D\$ on \$Y\$") ///
    booktabs label compress nonotes mtitle("OLS" "LASSO" "Random Forest" "Best") ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    keep($D) ///
    stats(N, fmt(%9.0fc) labels(`"Observations"'))	
	

	
	EDMN

