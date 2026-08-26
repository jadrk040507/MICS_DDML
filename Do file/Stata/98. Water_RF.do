/*******************************************************************************
Title: Random forest
Description: This do-file implements a random forest model

Input(s):
- Data/4.Final data/Training set.dta

Output(s):
- Data/2. Working data/rf-output.dta
- data/Logs/Random forest.md

Last ran: 06/13/2024
*******************************************************************************/
/*------------------------------------------------------------------------------
	2 Set file paths
------------------------------------------------------------------------------*/

	* Enter the file path to the project folder in Box for every new machine you use
	* Type 'di c(username)' to see the name of your machine
	
	else if c(username) == "akitokamei" {		
		global box 		"/Users/akitokamei/Box Sync/MICS Water project/"                 
		global github	"/Users/akitokamei/GitHub/i-h2o-india/"
		global Overleaf "/Users/akitokamei/Dropbox/Apps/Overleaf/"
		global DataRaw  "${box}01. 2_Pilot/Data/1_raw/"
		
	}
	
	global Table  "${Overleaf}MICS_Water/Table/"
	global Figure "${Overleaf}MICS_Water/Figure/"
	global Data   "${box}Data/"
	
********************************************************************************
**# Set up log
********************************************************************************

capture log close
set graph on
log using "${Table}MICS_WaterRandom forest.md", replace

* global outcome VeryHighRiskHome
global outcome WQ26

********************************************************************************
**# Configure models
********************************************************************************
* use  "${Data}Cleaned_Pooled_MICS6_Africa_2.dta", clear
use "${Data}MASTER_MICS_RF.dta", clear

gen id=_n
set seed 1381261
gen Random=runiform(0,1)
gen split=Random
recode split 0/0.5=1 0.5/1=2
tab split, m

* recode	
gen G_WS1=WS1
recode G_WS1 11 12 13 14=11 31 32=31 41 42=41 92=91  51 61 62 71 72=96

	label define G_WS1l 11 "WS: Piped Water" 21 "WS: Tube well/borehole"  31 "WS: Dug well" 41 "WS: Spring" 81 "WS: Surface Water" 91 "WS: Packaged water" 96 "WS: Other", modify
	label values G_WS1 G_WS1l

gen G_WS10=WS10
recode G_WS10 6=98
	label define G_WS10l 0 "Treat: Nothing" 1 "Treat: Boil" 2 "Treat: Bleach/Chlorine" 3 "Treat: Stain with a cloth" 4 "Treat: Filter" 5 "Treat: Soler" 7 "Treat: Aquatabs/PUR" 8 "Treat: Add tablet" 98 "Treat: Other" 99 "Treat: Do not know/missing", modify
	label values G_WS10 G_WS10l
	
	* WS1_11 WS1_12 WS1_13 WS1_14 WS1_21 WS1_31 WS1_32 WS1_41 WS1_42 WS1_51 WS1_61 WS1_62 WS1_71 WS1_72 WS1_81 WS1_91 WS1_92 WS1_96
	* WS10_0 WS10_1 WS10_2 WS10_3 WS10_6 WS10_7 WS10_98 WS10_99
	
* Variable construction
global V_source   G_WS1_11 G_WS1_21 G_WS1_31 G_WS1_41 G_WS1_81 G_WS1_91 G_WS1_96
global V_treat    G_WS10_0 G_WS10_1 G_WS10_2 G_WS10_3 G_WS10_7 G_WS10_98 G_WS10_99
global V_country  country_cat_2 country_cat_3 country_cat_4 country_cat_5 country_cat_8 country_cat_9 country_cat_10 country_cat_11 country_cat_12 country_cat_13 country_cat_14 country_cat_15 country_cat_16 country_cat_17 country_cat_19 country_cat_20 country_cat_21 country_cat_22 country_cat_23 country_cat_24 country_cat_25 country_cat_26 country_cat_27 country_cat_29 country_cat_32 country_cat_33
global V_simple   $V_source $V_treat $V_country  urban Open_defecation

	* Create Dummy:  WS1,  WS10,  WS3
	foreach v in G_WS1 G_WS10 country_cat {
	levelsof `v'
	foreach value in `r(levels)' {
		gen     `v'_`value'=0
		replace `v'_`value'=1 if `v'==`value'
		replace `v'_`value'=. if `v'==.
		label var `v'_`value' "`: label (`v') `value''"
	}
	}

/*
tab RiskHome RiskSource,m
hexplot RiskSource  RiskHome, values(format(%9.1f)) aspectratio(1) legend(off) ///
                              color(HCL reds, intensity(.6) reverse ) p(lc(black) lalign(center)) bins(5) ///
							  xlabel(0 "No risk" 1 "Moderate risk" 2 "Very high risk") xtitle("Point of use") ///
							  ylabel(0 "No risk" 1 "Moderate risk" 2 "Very high risk") ytitle("Source") ///
							  sizeprop
graph export "${Figure}TabSourceHome.eps", replace
*/

save           "${Data}MASTER_MICS_RF_Home.dta", replace
savesome using "${Data}MASTER_MICS_RF_Home0.dta" if RiskSource==0, replace
savesome using "${Data}MASTER_MICS_RF_Home1.dta" if RiskSource==1, replace

* graph bar VeryHighRiskHome, over(G_WS10, label(angle(45)))
* graph bar VeryHighRiskHome, over(G_WS1)

global RFcontrols G_WS1_11 G_WS1_21 G_WS1_31 G_WS1_41 G_WS1_81 G_WS1_91 G_WS1_96 ///
                  water_treatment ///
				  G_WS10_0 G_WS10_1 G_WS10_2 G_WS10_3 G_WS10_7 G_WS10_98 G_WS10_99 ///
				  WS3_1 WS3_2 WS3_3 ///
				  urban Open_defecation ///
				  
				local Main "Desciptive statistics by the level of source water contamination"
local LabelMain "Desc1"
local noteMain "Notes: WQ29: Ask Jeremy how to control this. Clean Primary Water Source wit Sujey. Discuss the variable after location. Clean more and decide what to include"
					 
foreach k in RFcontrols {
* Mean
	eststo  model0: estpost summarize $`k' if RiskHome==0
	eststo  model1: estpost summarize $`k' if RiskHome==1
	eststo  model2: estpost summarize $`k' if RiskHome==2

esttab model0 model1 model2 using "${Table}Descript_`k'_Risk.tex", title("``k''" \label{`Label`k''}) ///
	   cell("mean (fmt(2) label(_))") stats(N, fmt("%9.0fc") label(Observations) ) /// 
	   mtitles("No risk" "Moderate risk" "High risk") nonum ///
	   substitute( ".00" "" "{l}{\footnotesize" "{p{0.87\linewidth}}{\footnotesize" ///
				   "&           _&           _&           _&           _&           _\\" "" ///
				   "Piped water (Dwelling)" "\textbf{Primary Water Source} \\\hline Piped water (Dwelling)" ///
				   "Location: In own dwelling" "\textbf{Location} \\\hline Location: In own dwelling" ///
                   "Any water treatment for tested" "\textbf{Water that is tested} \\\hline Any water treatment" ///
				   "Any water treatment for primary" "\textbf{Primary Water Source} \\\hline Any water treatment" ///
				   "-0 " "0" ///
				   "Treat:"  "~~~" "Location:"  "~~~" ///
				   ) ///
	   label  note("`note`k''")  ///
	   replace 
	   }
	  

	  									********************************************************************************
										**                      # Some risk at the source (Random Forest)
										********************************************************************************


use "${Data}MASTER_MICS_RF_Home.dta", clear
keep RiskSource_0_12 $V_simple id split

*** Run random forest estimate ***

* Parameters: 
* Number of variables = controls/3 = 119/3 = 40
* Depth = 5 for regressions

rforest RiskSource_0_12 $V_simple if split == 1, type(reg) iter(500) seed(1666994) 

*** Also compute R2 ***
	 
*** Create a copy of the variable-importance matrix stored in e()

	matrix importance = e(importance)
	svmat importance
	gen oob_error = e(OOB_Error)
	gen features = e(features) 
	gen obs =  e(Observations)

*** Error on the test set ***

predict p if split == 2

gen validation_rmse1 = `e(RMSE)'
gen validation_mae1 = `e(MAE)'

label variable  validation_rmse1 "RMSE"
label variable  validation_mae1 "MAE"
label variable  oob_error "Out-of-bag Error"
label variable  features "Number of predictors"
label variable  obs "Number of observations"

*** Full sample prediction ***

predict p_all 

* hist p_all, by(split)
* hist needall, by(split)

drop p 

save "${Data}rf-output.dta", replace

eststo clear
use "${Data}rf-output.dta", clear
eststo temp1: reg RiskSource_0_12 p_all if split==1
sum RiskSource_0_12 if split==1
estadd scalar Min = r(min) : temp1
estadd scalar Max = r(max) : temp1
eststo temp2: reg RiskSource_0_12 p_all if split==2
sum RiskSource_0_12 if split==2
estadd scalar Min = r(min) : temp2
estadd scalar Max = r(max) : temp2
esttab using "${Table}RFP.tex",label se ar2 title("The performance of the model for the training and testing sample" \label{RF}) nonotes nobase ///
			 mtitle("Training" "Testing") drop(p_all _cons) ///
			 stats(r2_a rmse Min Max   N, fmt(%9.2fc %9.2fc %9.2fc %9.2fc %9.0fc) labels(`"Adjusted \(R^{2}\)"' `"RMSE"' `"Min"' `"Max"'  `"Observations"')) ///
			 starlevels(\sym{*} 0.10 \sym{**} 0.05 \sym{***} 0.010) b(2) ///
			 substitute("{l}{\footnotesize" "{p{0.5\linewidth}}{\footnotesize" ///
			 "=1" "" ///
			 ) ///
			 addnote("Note: ") ///	
			 replace
eststo clear



reg RiskSource_0_12 p_all
twoway (lowess RiskSource_0_12 p_all if split==1) ///
       (lowess RiskSource_0_12 p_all if split==2, msize(tiny)) ///
       (lfit   RiskSource_0_12 RiskSource_0_12 if split==2, lpattern(dot) lcolor(black)) , ///
	   legend(order(1 "Training sample" 2 "Test sample" 3 "45 degree line")) ///
	   xtitle("Actual risk") ///
	   ytitle("Predicted value")
graph export "${Figure}RFP.eps", replace
	   
********************************************************************************
**# Variable Importance
********************************************************************************
use "${Data}rf-output.dta", clear
keep RiskSource_0_12 $V_simple id split importance1

*** Generate new variable id to be used for labeling ***
	gen names=""

*** Attach unique labels to individual columns in the chart ***
        local mynames : rownames importance
        local k : word count `mynames'
            // If there are more variables than observations
            if `k'>_N {
                set obs `k'
            }
            forvalues i = 1(1)`k' {
                local aword : word `i' of `mynames'
                local alabel : variable label `aword'
                if ("`alabel'"!="") quietly replace names= "`alabel'" in `i'
                else quietly replace names= "`aword'" in `i'
            }
			

sort importance1

* Drop rows with missing information
drop if importance1 ==. | names == ""

* Split into 4 panels
gen row = _n
gen group = 1
* replace group = 2 if row >= 29 & row < 58
* replace group = 3 if row >= 58 & row < 87
* replace group = 4 if row >= 87

graph hbar importance1 if group == 1, over(names, sort(1) label(labsize(vsmall))) ytitle("") ///
	nofill noext dots(mcolor(gs10)) ylab(0(0.1)1, glcolor(gs15) glstyle(solid)) plotregion(lcolor(black) lwidth(.2) )  ///
	graphregion(color(white))
graph export "${Figure}RFI.eps", replace

										********************************************************************************
										**                      # Some risk at the source (Random Forest)
										********************************************************************************


use "${Data}MASTER_MICS_RF_Home1.dta", clear
drop if RiskHome==0
keep NoRiskHome_01_2 $V_simple id split

*** Run random forest estimate ***

* Parameters: 
* Number of variables = controls/3 = 119/3 = 40
* Depth = 5 for regressions

rforest NoRiskHome_01_2 $V_simple if split == 1, type(reg) iter(500) seed(1666994) 

*** Also compute R2 ***
	 
*** Create a copy of the variable-importance matrix stored in e()

	matrix importance = e(importance)
	svmat importance
	gen oob_error = e(OOB_Error)
	gen features = e(features) 
	gen obs =  e(Observations)

*** Error on the test set ***

predict p if split == 2

gen validation_rmse1 = `e(RMSE)'
gen validation_mae1 = `e(MAE)'

label variable  validation_rmse1 "RMSE"
label variable  validation_mae1 "MAE"
label variable  oob_error "Out-of-bag Error"
label variable  features "Number of predictors"
label variable  obs "Number of observations"

*** Full sample prediction ***

predict p_all 

* hist p_all, by(split)
* hist needall, by(split)

drop p 

save "${Data}rf-output.dta", replace

eststo clear
use "${Data}rf-output.dta", clear
eststo temp1: reg NoRiskHome_01_2 p_all if split==1
sum NoRiskHome_01_2 if split==1
estadd scalar Min = r(min) : temp1
estadd scalar Max = r(max) : temp1
eststo temp2: reg NoRiskHome_01_2 p_all if split==2
sum NoRiskHome_01_2 if split==2
estadd scalar Min = r(min) : temp2
estadd scalar Max = r(max) : temp2
esttab using "${Table}RFP1.tex",label se ar2 title("The performance of the model for the training and testing sample: Determinants of having very high risk drinking water from the housheolds with some contamination" \label{ETR1}) nonotes nobase ///
			 mtitle("Training" "Testing") drop(p_all _cons) ///
			 stats(r2_a rmse Min Max   N, fmt(%9.2fc %9.2fc %9.2fc %9.2fc %9.0fc) labels(`"Adjusted \(R^{2}\)"' `"RMSE"' `"Min"' `"Max"'  `"Observations"')) ///
			 starlevels(\sym{*} 0.10 \sym{**} 0.05 \sym{***} 0.010) b(2) ///
			 substitute("{l}{\footnotesize" "{p{1\linewidth}}{\footnotesize" ///
			 "=1" "" ///
			 ) ///
			 addnote("Note: The base of the socio-economic level is the two lowest quintile poor and very poor. Standard errors clustered at the primary sampling unit in parentheses, $\sym{*} p<.10,\sym{**} p<.05,\sym{***} p<.01$") ///	
			 replace
eststo clear



reg NoRiskHome_01_2 p_all
twoway (lowess NoRiskHome_01_2 p_all if split==1) ///
       (lowess NoRiskHome_01_2 p_all if split==2, msize(tiny)) ///
       (lfit   NoRiskHome_01_2 NoRiskHome_01_2 if split==2, lpattern(dot) lcolor(black)) , ///
	   legend(order(1 "Training sample" 2 "Test sample" 3 "45 degree line")) ///
	   xtitle("Actual risk") ///
	   ytitle("Predicted value")
graph export "${Figure}RFP1.eps", replace
	   
********************************************************************************
**# Variable Importance
********************************************************************************
use "${Data}rf-output.dta", clear
keep NoRiskHome_01_2 $V_simple id split importance1

*** Generate new variable id to be used for labeling ***
	gen names=""

*** Attach unique labels to individual columns in the chart ***
        local mynames : rownames importance
        local k : word count `mynames'
            // If there are more variables than observations
            if `k'>_N {
                set obs `k'
            }
            forvalues i = 1(1)`k' {
                local aword : word `i' of `mynames'
                local alabel : variable label `aword'
                if ("`alabel'"!="") quietly replace names= "`alabel'" in `i'
                else quietly replace names= "`aword'" in `i'
            }
			

sort importance1

* Drop rows with missing information
drop if importance1 ==. | names == ""

* Split into 4 panels
gen row = _n
gen group = 1
* replace group = 2 if row >= 29 & row < 58
* replace group = 3 if row >= 58 & row < 87
* replace group = 4 if row >= 87

graph hbar importance1 if group == 1, over(names, sort(1) label(labsize(vsmall))) ytitle("") ///
	nofill noext dots(mcolor(gs10)) ylab(0(0.1)1, glcolor(gs15) glstyle(solid)) plotregion(lcolor(black) lwidth(.2) )  ///
	graphregion(color(white))
graph export "${Figure}RFI1.eps", replace


										********************************************************************************
										**                      # No risk at the source (Random Forest)
										********************************************************************************


use "${Data}MASTER_MICS_RF_Home0.dta", clear
keep RiskHome_0_12 $V_simple id split

*** Run random forest estimate ***

* Parameters: 
* Number of variables = controls/3 = 119/3 = 40
* Depth = 5 for regressions

rforest  RiskHome_0_12 $V_simple if split == 1, type(reg) iter(500) seed(1666994) 

*** Also compute R2 ***
	 
*** Create a copy of the variable-importance matrix stored in e()

	matrix importance = e(importance)
	svmat importance
	gen oob_error = e(OOB_Error)
	gen features = e(features) 
	gen obs =  e(Observations)

*** Error on the test set ***

predict p if split == 2

gen validation_rmse1 = `e(RMSE)'
gen validation_mae1 = `e(MAE)'

label variable  validation_rmse1 "RMSE"
label variable  validation_mae1 "MAE"
label variable  oob_error "Out-of-bag Error"
label variable  features "Number of predictors"
label variable  obs "Number of observations"

*** Full sample prediction ***

predict p_all 

* hist p_all, by(split)
* hist needall, by(split)

drop p 

save "${Data}rf-output.dta", replace

use "${Data}rf-output.dta", clear

eststo temp1: reg RiskHome_0_12 p_all if split==1
sum RiskHome_0_12 if split==1
estadd scalar Min = r(min) : temp1
estadd scalar Max = r(max) : temp1
eststo temp2: reg RiskHome_0_12 p_all if split==2
sum RiskHome_0_12 if split==2
estadd scalar Min = r(min) : temp2
estadd scalar Max = r(max) : temp2
esttab using "${Table}RFP0.tex",label se ar2 title("The performance of the model for the training and testing sample: Determinants of having some E.Coli in the drinking water from the free from contamination water source" \label{ETR1}) nonotes nobase ///
			 mtitle("Training" "Testing") drop(p_all _cons) ///
			 stats(r2_a rmse Min Max   N, fmt(%9.2fc %9.2fc %9.2fc %9.2fc %9.0fc) labels(`"Adjusted \(R^{2}\)"' `"RMSE"' `"Min"' `"Max"'  `"Observations"')) ///
			 starlevels(\sym{*} 0.10 \sym{**} 0.05 \sym{***} 0.010) b(2) ///
			 substitute("{l}{\footnotesize" "{p{1\linewidth}}{\footnotesize" ///
			 "=1" "" ///
			 ) ///
			 addnote("Note: The base of the socio-economic level is the two lowest quintile poor and very poor. Standard errors clustered at the primary sampling unit in parentheses, $\sym{*} p<.10,\sym{**} p<.05,\sym{***} p<.01$") ///	
			 replace
eststo clear



twoway (lowess RiskHome_0_12 p_all if split==1, msize(tiny) msymbol(Oh)) ///
       (lowess RiskHome_0_12 p_all if split==2, msize(tiny)) ///
       (lfit RiskHome_0_12 RiskHome_0_12 if split==2, lpattern(dot) lcolor(black)) , ///
	   legend(order(1 "Training sample" 2 "Test sample" 3 "45 degree line")) ///
	   xtitle("Actual risk") ///
	   ytitle("Predicted value")
graph export "${Figure}RFP0.eps", replace


********************************************************************************
**# Variable Importance
********************************************************************************
use "${Data}rf-output.dta", clear
keep RiskHome_0_12 $V_simple id split importance1

*** Generate new variable id to be used for labeling ***
	gen names=""

*** Attach unique labels to individual columns in the chart ***
        local mynames : rownames importance
        local k : word count `mynames'
            // If there are more variables than observations
            if `k'>_N {
                set obs `k'
            }
            forvalues i = 1(1)`k' {
                local aword : word `i' of `mynames'
                local alabel : variable label `aword'
                if ("`alabel'"!="") quietly replace names= "`alabel'" in `i'
                else quietly replace names= "`aword'" in `i'
            }
			

sort importance1

* Drop rows with missing information
drop if importance1 ==. | names == ""

* Split into 4 panels
gen row = _n
gen group = 1
* replace group = 2 if row >= 29 & row < 58
* replace group = 3 if row >= 58 & row < 87
* replace group = 4 if row >= 87

graph hbar importance1 if group == 1, over(names, sort(1) label(labsize(vsmall))) ytitle("") ///
	nofill noext dots(mcolor(gs10)) ylab(0(0.1)1, glcolor(gs15) glstyle(solid)) plotregion(lcolor(black) lwidth(.2) )  ///
	graphregion(color(white))
graph export "${Figure}RFI0.eps", replace


#del ;
	graph hbar importance1 if group == 1, ///
	over(names, sort(1) label(labsize(vsmall))) ///
	ytitle("") ///
	ysize(1) nofill noext dots(mcolor(gs10)) ///
	ylab(0(0.1)1, glcolor(gs15) glstyle(solid)) ///
	plotregion(lcolor(black) lwidth(.2) )  ///
	graphregion(color(white)) ///
	name(A,replace) nodraw ///
;
#del cr

/*
#del ;
	graph hbar importance1 if group == 2, ///
	over(names, sort(1) label(labsize(vsmall))) ///
	ytitle("") ///
	ysize(1) nofill noext dots(mcolor(gs10)) ///
	ylab(0(0.1)1, glcolor(gs15) glstyle(solid)) ///
	plotregion(lcolor(black) lwidth(.2) )  ///
	graphregion(color(white)) ///
	name(B,replace) nodraw ///
;
#del cr

graph combine A B, col(2)  b1(Importance) iscale(*0.75) graphregion(color(white))
graph export "${Figure}Random Forest Importance_more.png", as(png) height(1200) replace
*/

END
 
 
********************************************************************************
**# Predictions
********************************************************************************

use "${Data}rf-output.dta", clear 
rename (p_all RiskHome_0_12) (needall1 needall2)

drop id
gen id = _n

reshape long needall, i(id) j(type)

gen name = ""
replace name = "Prediction" if type == 1
replace name = "Observed" if type == 2

save "${Data}wide_pred.dta", replace

recode needall 0/0.5=0 0.5/1=1

twoway (histogram needall if name == "Prediction", bin(50) color(blue%30)) ///
	   (histogram needall if name == "Observed",   bin(50) color(grey%30)), ///
	   legend(symxsize(4) rows(1) order(1 "Prediction" 2 "Observed" )) ///
	   title("Random Forest Predictions", size(medium)) ///
	   xtitle("Percentage of beneficiaries") ///
	   plotregion(lcolor(black) lwidth(.2) ) ///
	   graphregion(color(white))
	   
graph export "${Figure}Random Forest Prediction.png", as(png) height(1200) replace

tabstat needall, stat(mean sd min max) by(name)
 

********************************************************************************
**## Predictions by variables
********************************************************************************

use "${Data}wide_pred.dta", clear
collapse (mean) needall (max) phone2015, by(munid name)

egen deciles = xtile(phone2015), n(10) by(name)

label variable deciles "Percentage of households with access to a phone (deciles)"


reshape wide needall, i(munid) j(name) string

#del ;
	graph bar needallObserved needallPrediction, ///
	over(deciles) ///
	bar(1, color(150 79 142)) bar(2, color(255 191 128)) ///
	intensity(40) ///
	ylab(0(0.01)0.1, glcolor(gs15) glstyle(solid) angle(0))
	ytitle("% of beneficiaries", size(small)) ///
	title("Percentage of households with access to a phone" "at the municipality level in deciles", size(medium)) ///
	legend(symxsize(4) rows(1) order( 2 "Observed" 1 "Random Forest Prediction")) ///
	plotregion(lcolor(black) lwidth(.2) ) 
	graphregion(color(white))

	;
#del cr



********************************************************************************
**# Other method
********************************************************************************
gen oob_error1 = .
gen validation_rmse1 = .
gen validation_mae1 = .
gen iter1 = .
local j = 0

forvalues i = 100(20)500 {
	local j = `j' + 1
	rforest needall has_* av_* dew precipprob precipcover uvindex daylight_hours temp_max ///
	   temp_min humidity precip windspeed winddir sealevelpressure cloudcover ///
	   visibility solarradiation solarenergy computer2015 electric2015 ///
	   internet2015 phone2015 tv2015 ///
	   altitude popdens5 SES_v9 if split == 1, ///
	type(reg) iter(`i') numvars(1) seed(1666994)
	replace iter1 = `i' in `j'
	replace oob_error1 = `e(OOB_Error)' in `j'

	replace validation_rmse1 = `e(RMSE)' in `j'
	replace validation_mae1 = `e(MAE)' in `j'
	drop p
}


label variable oob_error1 "Out-of-bag error"
label variable iter1 "Iterations"
label variable validation_rmse1 "Validation RMSE"
scatter oob_error1 iter1, mcolor(blue) msize(tiny) 

|| scatter validation_mae1 iter1, mcolor(red) msize(tiny)


EDNB



gen     V_interact=.
replace V_interact=1 if G_WS1==11 & WS9==2
replace V_interact=2 if G_WS1==21 & WS9==2
replace V_interact=3 if G_WS1==31 & WS9==2
replace V_interact=4 if G_WS1==41 & WS9==2
replace V_interact=5 if G_WS1==81 & WS9==2
replace V_interact=6 if G_WS1==91 & WS9==2
replace V_interact=7 if G_WS1==96 & WS9==2

replace V_interact=11 if G_WS1==11 & WS9==1
replace V_interact=12 if G_WS1==21 & WS9==1
replace V_interact=13 if G_WS1==31 & WS9==1
replace V_interact=14 if G_WS1==41 & WS9==1
replace V_interact=15 if G_WS1==81 & WS9==1
replace V_interact=16 if G_WS1==91 & WS9==1
replace V_interact=17 if G_WS1==96 & WS9==1

	label define V_interactl 1 "WS: Piped Water (N)" 2 "WS: Tube well/borehole (N)"  3 "WS: Dug well (N)" 4 "WS: Spring (N)" 5 "WS: Surface Water (N)" 6 "WS: Packaged water (N)" 7 "WS: Other (N)" 11 "WS: Piped Water (Y)" 12 "WS: Tube well/borehole (Y)" 13 "WS: Dug well (Y)" 14 "WS: Spring (Y)" 15 "WS: Surface Water (Y)" 16 "WS: Packaged water (Y)" 17 "WS: Other (Y)", modify
	label values V_interact V_interactl
	graph bar VeryHighRiskHome if G_WS1==31 | G_WS1==96 | G_WS1==11, over(V_interact)


global RFcontrolsML V_interact_1 V_interact_2 V_interact_3 V_interact_4 V_interact_5 V_interact_6 V_interact_7  ///
                    V_interact_11 V_interact_12 V_interact_13 V_interact_14 V_interact_15 V_interact_16 V_interact_17  ///
				    WS3_1 WS3_2 WS3_3 ///
				    urban Open_defecation ///
					water_treatment
				


