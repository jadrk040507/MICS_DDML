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
	
	else if c(username) == "Juan Alvaro" {		
		global Dropbox "C:/Users/Juan Alvaro/Dropbox/"
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

*====================================================================
* 2. LOAD DATA & (OPTIONAL) TEST SAMPLE
*====================================================================

use "${Data_Clean}MASTER_MICS_DDML_FINAL.dta", clear

*====================================================================
* 3. DATA CLEANING & RECODING
*====================================================================

* 10 tiles (deciles) of WQ27
xtile wq27_decile = WQ27, nq(10)

* Education: collapse into 3 categories
recode helevel (2/4 = 2)
label define helevell 0 "No education" 1 "Primary" 2 "Secondary or higher", modify
label values helevel helevell

* replace water_treatment=0 if WQ15_g_3==1 
gen    SomeRiskHome=RiskHome
recode SomeRiskHome 1 2=1 0=0

* Check the need for WQ11
foreach v in urban windex_ur helevel country_cat urban WS1_g wq27_decile {
    drop if missing(`v')
}

save "${Data_Final}MASTER_MICS_FINAL.dta", replace

end

*====================================================================
* 1.1. VARIABLE DEFINITIONS
*===================================================================
start_from_final

lasso linear water_treatment i.(windex_ur helevel country_cat urban WS1_g wq27_decile)
* Store selected variables
coefplot, nolabels ///
    drop(_cons) ///
    vertical ///
    yline(0) ///
	coeflabels( ///
	1.wq27_decile = "No E. coli detected" ///
    5.wq27_decile = "Low E. coli contamination" ///
    6.wq27_decile = "Moderate E. coli contamination" ///
    7.wq27_decile = "High E. coli contamination" ///
    8.wq27_decile = "Very high E. coli contamination" ///
	11.WS1_g = "Piped water" ///
    21.WS1_g = "Tube well / Borehole" ///
    31.WS1_g = "Protected well or spring" ///
    32.WS1_g = "Unprotected well or spring" ///
    51.WS1_g = "Surface or rain water" ///
    91.WS1_g = "Packaged or bottled water" ///
    96.WS1_g = "Other water sources" ///
    2.country_cat  = "Bangladesh" ///
    3.country_cat  = "Benin" ///
    4.country_cat  = "Central African Republic" ///
    5.country_cat  = "Chad" ///
    8.country_cat  = "DR Congo" ///
    9.country_cat  = "Dominican Republic" ///
    10.country_cat = "Eswatini" ///
    11.country_cat = "Fiji" ///
    12.country_cat = "Gambia" ///
    13.country_cat = "Ghana" ///
    14.country_cat = "Guinea-Bissau" ///
    15.country_cat = "Guyana" ///
    16.country_cat = "Honduras" ///
    18.country_cat = "Lao PDR" ///
    19.country_cat = "Lesotho" ///
    20.country_cat = "Madagascar" ///
    21.country_cat = "Malawi" ///
    22.country_cat = "Mongolia" ///
    24.country_cat = "Sierra Leone" ///
    25.country_cat = "Suriname" ///
    26.country_cat = "Togo" ///
    28.country_cat = "Trinidad and Tobago" ///
    31.country_cat = "Vietnam" ///
    32.country_cat = "Zimbabwe" ///
) ///
    sort(., descending) ///
    xlabel(, angle(60) labsize(vsmall)) 
	
graph export "${Figures}lasso_coefficients_water_treatment.png", replace

*--------------------------------------------------
* Out of sample
*--------------------------------------------------
start_from_final
set seed 12345
gen sample = runiform()
* Train on 70%
lasso linear water_treatment i.(windex_ur helevel country_cat urban WS1_g wq27_decile) ///
    if sample < 0.7

* Select lambda
lassoselect id = 68

predict yhat_test if sample >= 0.7

gen  sq_error_test = (water_treatment - yhat_test)^2 if sample >= 0.7
summ sq_error_test

gen yhat_bin = yhat_test > 0.5 if sample >= 0.7
tab yhat_bin water_treatment if sample >= 0.7, cell

* "Using a hold-out sample, the LASSO model predicts household water treatment with an out-of-sample MSE of XX and a classification accuracy of YY%."

*--------------------------------------------------
* ROCTAB
*--------------------------------------------------
start_from_final

* LASSO
lasso linear water_treatment i.(windex_ur helevel country_cat urban WS1_g wq27_decile)

predict yhat_lasso
roctab water_treatment yhat_lasso
* 0.8 = very strong
