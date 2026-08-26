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
fre age

global Controls i.windex5 i.helevel i.country_cat i.urban i.WS1_g ///
                Any_U5 Girls_less_than15 Boys_15or_less i.Toilet i.wq27_decile
global Controls_U5 male i.age
			   
foreach i in diarrhea fever {

    *----------------------------
    * Panel A output (No control)
    *----------------------------
    eststo clear

    * Across (WQ15_g spec)
    eststo: reg `i' i.WQ15_g $Controls_U5, cluster(HHID)
    sum `i' if water_treatment==0
    estadd scalar Mean = r(mean)

    * Across (water_treatment spec)
    eststo: reg `i' water_treatment $Controls_U5, cluster(HHID)
    sum `i' if water_treatment==0
    estadd scalar Mean = r(mean)

    * By RiskSource (0/1/2), each with two specs
    foreach k in 0 1 2 {
        eststo: reg `i' i.WQ15_g $Controls_U5 if RiskSource==`k', cluster(HHID)
        sum `i' if RiskSource==`k' & water_treatment==0
        estadd scalar Mean = r(mean)

        eststo: reg `i' water_treatment $Controls_U5 if RiskSource==`k', cluster(HHID)
        sum `i' if RiskSource==`k' & water_treatment==0
        estadd scalar Mean = r(mean)
    }

    * LaTeX fragment for Panel A (basic)
    esttab using "${Tables}Est_OLS_`i'.tex", ///
        label se ar2 ///
        stats(Mean r2_a N, fmt(%9.2fc %9.2fc %9.0fc) ///
              labels(`"Mean"' `"Adjusted \(R^{2}\)"' `"Observations"')) ///
        nobase nonotes drop(_cons) nomtitle ///
        varlabels( ///
            WQ26  "E. coli detected in drinking water" ///
            water_treatment "Any water treatment" ///
            0.WQ15_g "Nothing" 1.WQ15_g "Boiling" 2.WQ15_g "Chlorination" ///
            3.WQ15_g "Strain/Settle" 98.WQ15_g "Other treatment" 99.WQ15_g "Do not know" ///
        ) ///
        star(* .10 ** .05 *** .01) b(3) ///
		indicate("Controls=*age*") ///
        mgroups("Across" "Low risk at source" "Medium risk at source" "High risk at source", ///
                pattern(1 0 1 0 1 0 1 0) ///
                prefix(\multicolumn{@span}{c}{) suffix(}) span ///
                erepeat(\cmidrule(lr){@span})) ///
        order(water_treatment) ///
        substitute("{l}{\footnotesize" "{p{0.93\linewidth}}{\footnotesize" ///
                   "Sprin_g" "\textbf{Season (base=winter)} \\ \hline Spring" ///
                   "Holiday" "\hline Holiday" ///
                   "Mist/Cloudy " "\textbf{Weather (base=clear)} \\ \hline Mist/Cloudy " ///
                   "Any water treatment" "Any treatment" ///
                   ) ///
        replace

    *----------------------------
    * Panel B output (extended)
    *----------------------------
    eststo clear

    * Across (WQ15_g spec)
    eststo: reg `i' i.WQ15_g $Controls_U5 $Controls , cluster(HHID)
    sum `i' if water_treatment==0
    estadd scalar Mean = r(mean)

    * Across (water_treatment spec)
    eststo: reg `i' water_treatment $Controls_U5 $Controls, cluster(HHID)
    sum `i' if water_treatment==0
    estadd scalar Mean = r(mean)

    * By RiskSource (0/1/2), each with two specs
    foreach k in 0 1 2 {
        eststo: reg `i' i.WQ15_g $Controls_U5 $Controls if RiskSource==`k', cluster(HHID)
        sum `i' if RiskSource==`k' & water_treatment==0
        estadd scalar Mean = r(mean)

        eststo: reg `i' water_treatment $Controls_U5 $Controls if RiskSource==`k', cluster(HHID)
        sum `i' if RiskSource==`k' & water_treatment==0
        estadd scalar Mean = r(mean)
    }

    * LaTeX fragment for Panel B (extended)
    esttab using "${Tables}Est_OLS_`i'_extend.tex", ///
        label se ar2 ///
        stats(Mean r2_a N, fmt(%9.2fc %9.2fc %9.0fc) ///
              labels(`"Mean"' `"Adjusted \(R^{2}\)"' `"Observations"')) ///
        nobase nonotes drop(_cons) nomtitle ///
        indicate("Controls=*windex5 *country_cat *helevel *urban *WS1_g Any_U5 Girls_less_than15 Boys_15or_less *Toilet* *wq27_decile*") ///
        varlabels( ///
            WQ26  "E. coli detected in drinking water" ///
            water_treatment "Any water treatment" ///
            0.WQ15_g "Nothing" 1.WQ15_g "Boiling" 2.WQ15_g "Chlorination" ///
            3.WQ15_g "Strain/Settle" 98.WQ15_g "Other treatment" 99.WQ15_g "Do not know" ///
        ) ///
        star(* .10 ** .05 *** .01) b(3) ///
        mgroups("Across" "Low risk at source" "Medium risk at source" "High risk at source", ///
                pattern(1 0 1 0 1 0 1 0) ///
                prefix(\multicolumn{@span}{c}{) suffix(}) span ///
                erepeat(\cmidrule(lr){@span})) ///
        order(water_treatment) ///
        substitute("{l}{\footnotesize" "{p{0.93\linewidth}}{\footnotesize" ///
                   "Any water treatment" "Any treatment" ///
				   "Male" "\textbf{\shortstack[l]{Child\\characteristics}} \\ \hline Male" ///
				   "Age 1" "\textbf{Age (Base=0)}\\ Age 1" ///
                   ) ///
        replace

    *----------------------------
    * Combine Panel A + Panel B into one wrapper table
    *----------------------------
    file close _all
    file open fh using "${Tables}Est_OLS_`i'_full.tex", write replace

    file write fh "\begin{table}[htbp]\centering" _n
    file write fh "\caption{OLS Estimates for ``i''}\label{tab:Est_OLS_`i'_full}" _n
    file write fh "\begin{tabular}{l}" _n
    file write fh "\hline \hline" _n

    * Panel A
    file write fh "\textbf{Panel A: No controls}\\[2pt]" _n
    file write fh "\input{Table/Est_OLS_`i'.tex}\\[6pt]" _n

    * Panel B
    file write fh "\textbf{Panel B: Controls}\\[2pt]" _n
    file write fh "\input{Table/Est_OLS_`i'_extend.tex}\\[4pt]" _n

    * Notes
    file write fh "\multicolumn{1}{p{0.95\linewidth}}{`note`i''} \\" _n
    file write fh "\hline" _n
    file write fh "\end{tabular}" _n
    file write fh "\end{table}" _n

    file close fh
}

* End
file close _all



*===============================================================
* Combine BASIC and EXTENDED into one LaTeX table (Panel A / B)
* for each outcome: VeryHighRiskHome, SomeRiskHome
*===============================================================

************************************************
* Panel A: No controls
************************************************
start_from_final

local VeryHighRiskHome      "Probability of high E. coli contamination ($>$ 100 CFU) in household drinking water"
local SomeRiskHome          "Probability of any E. coli contamination in household drinking water"
local noteVeryHighRiskHome  "Notes: $\sym{*} p<0.10,\sym{**} p<0.05,\sym{***} p<0.01$."
local noteSomeRiskHome      "Notes: $\sym{*} p<0.10,\sym{**} p<0.05,\sym{***} p<0.01$."
local LabelVeryHighRiskHome "VeryHighRiskHome"
local LabelSomeRiskHome     "SomeRiskHome"

************************************************
* Panel B: Controls
************************************************
start_from_final

global XEXTEND i.windex5 i.helevel i.country_cat i.urban i.WS1_g ///
               Any_U5 Girls_less_than15 Boys_15or_less ///
			   i.Toilet i.wq27_decile

************************************************
* Loop over outcomes, create:
*   1) Est_EColi_OLS_<outcome>.tex        (Panel A)
*   2) Est_EColi_OLS_<outcome>_extend.tex (Panel B)
*   3) Est_EColi_OLS_<outcome>_full.tex   (Panels A+B wrapper)
************************************************

foreach i in VeryHighRiskHome SomeRiskHome {

    *----------------------------
    * Panel A output (No control)
    *----------------------------
    eststo clear

    * Across (WQ15_g spec)
    eststo: reg `i' i.WQ15_g 
    sum `i' if water_treatment==0
    estadd scalar Mean = r(mean)

    * Across (water_treatment spec)
    eststo: reg `i' water_treatment 
    sum `i' if water_treatment==0
    estadd scalar Mean = r(mean)

    * By RiskSource (0/1/2), each with two specs
    foreach k in 0 1 2 {
        eststo: reg `i' i.WQ15_g  if RiskSource==`k'
        sum `i' if RiskSource==`k' & water_treatment==0
        estadd scalar Mean = r(mean)

        eststo: reg `i' water_treatment  if RiskSource==`k'
        sum `i' if RiskSource==`k' & water_treatment==0
        estadd scalar Mean = r(mean)
    }

    * LaTeX fragment for Panel A (basic)
    esttab using "${Tables}Est_EColi_OLS_`i'.tex", ///
        label se ar2 ///
        stats(Mean r2_a N, fmt(%9.2fc %9.2fc %9.0fc) ///
              labels(`"Mean"' `"Adjusted \(R^{2}\)"' `"Observations"')) ///
        nobase nonotes drop(_cons) nomtitle ///
        varlabels( ///
            WQ26  "E. coli detected in drinking water" ///
            water_treatment "Any water treatment" ///
            0.WQ15_g "Nothing" 1.WQ15_g "Boiling" 2.WQ15_g "Chlorination" ///
            3.WQ15_g "Strain/Settle" 98.WQ15_g "Other treatment" 99.WQ15_g "Do not know" ///
        ) ///
        star(* .10 ** .05 *** .01) b(3) ///
        mgroups("Across" "Low risk at source" "Medium risk at source" "High risk at source", ///
                pattern(1 0 1 0 1 0 1 0) ///
                prefix(\multicolumn{@span}{c}{) suffix(}) span ///
                erepeat(\cmidrule(lr){@span})) ///
        order(water_treatment) ///
        substitute("{l}{\footnotesize" "{p{0.93\linewidth}}{\footnotesize" ///
                   "Sprin_g" "\textbf{Season (base=winter)} \\ \hline Spring" ///
                   "Holiday" "\hline Holiday" ///
                   "Mist/Cloudy " "\textbf{Weather (base=clear)} \\ \hline Mist/Cloudy " ///
                   "Any water treatment" "Any treatment" ///
                   "=1" "" ) ///
        replace

    *----------------------------
    * Panel B output (extended)
    *----------------------------
    eststo clear

    * Across (WQ15_g spec)
    eststo: reg `i' i.WQ15_g $XEXTEND
    sum `i' if water_treatment==0
    estadd scalar Mean = r(mean)

    * Across (water_treatment spec)
    eststo: reg `i' water_treatment $XEXTEND
    sum `i' if water_treatment==0
    estadd scalar Mean = r(mean)

    * By RiskSource (0/1/2), each with two specs
    foreach k in 0 1 2 {
        eststo: reg `i' i.WQ15_g $XEXTEND if RiskSource==`k'
        sum `i' if RiskSource==`k' & water_treatment==0
        estadd scalar Mean = r(mean)

        eststo: reg `i' water_treatment $XEXTEND if RiskSource==`k'
        sum `i' if RiskSource==`k' & water_treatment==0
        estadd scalar Mean = r(mean)
    }

    * LaTeX fragment for Panel B (extended)
    esttab using "${Tables}Est_EColi_OLS_`i'_extend.tex", ///
        label se ar2 ///
        stats(Mean r2_a N, fmt(%9.2fc %9.2fc %9.0fc) ///
              labels(`"Mean"' `"Adjusted \(R^{2}\)"' `"Observations"')) ///
        nobase nonotes drop(_cons) nomtitle ///
        indicate("Controls=*windex5 *country_cat *helevel *urban *WS1_g Any_U5 Girls_less_than15 Boys_15or_less *wq27_decile* *Toilet*") ///
        varlabels( ///
            WQ26  "E. coli detected in drinking water" ///
            water_treatment "Any water treatment" ///
            0.WQ15_g "Nothing" 1.WQ15_g "Boiling" 2.WQ15_g "Chlorination" ///
            3.WQ15_g "Strain/Settle" 98.WQ15_g "Other treatment" 99.WQ15_g "Do not know" ///
        ) ///
        star(* .10 ** .05 *** .01) b(3) ///
        mgroups("Across" "Low risk at source" "Medium risk at source" "High risk at source", ///
                pattern(1 0 1 0 1 0 1 0) ///
                prefix(\multicolumn{@span}{c}{) suffix(}) span ///
                erepeat(\cmidrule(lr){@span})) ///
        order(water_treatment) ///
        substitute("{l}{\footnotesize" "{p{0.93\linewidth}}{\footnotesize" ///
                   "Sprin_g" "\textbf{Season (base=winter)} \\ \hline Spring" ///
                   "Holiday" "\hline Holiday" ///
                   "Mist/Cloudy " "\textbf{Weather (base=clear)} \\ \hline Mist/Cloudy " ///
                   "Any water treatment" "Any treatment" ///
                   "=1" "" ) ///
        replace

    *----------------------------
    * Combine Panel A + Panel B into one wrapper table
    *----------------------------
    file close _all
    file open fh using "${Tables}Est_EColi_OLS_`i'_full.tex", write replace

    file write fh "\begin{table}[htbp]\centering" _n
    file write fh "\caption{OLS Estimates for ``i''}\label{tab:Est_EColi_OLS_`i'_full}" _n
    file write fh "\begin{tabular}{l}" _n
    file write fh "\hline \hline" _n

    * Panel A
    file write fh "\textbf{Panel A: No controls}\\[2pt]" _n
    file write fh "\input{Table/Est_EColi_OLS_`i'.tex}\\[6pt]" _n

    * Panel B
    file write fh "\textbf{Panel B: Controls}\\[2pt]" _n
    file write fh "\input{Table/Est_EColi_OLS_`i'_extend.tex}\\[4pt]" _n

    * Notes
    file write fh "\multicolumn{1}{p{0.95\linewidth}}{`note`i''} \\" _n
    file write fh "\hline" _n
    file write fh "\end{tabular}" _n
    file write fh "\end{table}" _n

    file close fh
}

* End
file close _all


