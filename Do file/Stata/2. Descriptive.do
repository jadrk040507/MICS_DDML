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

* Final data use macro	
cap program drop start_final
program define   start_final

use "${Data_Final}MASTER_MICS_FINAL.dta", clear

end

* Final data use macro	
cap program drop start_final_U5
program define   start_final_U5

use "${Data_Final}MASTER_MICS_FINAL_U5.dta", clear

label var diarrhea "Diarrhea"
label var age "Age in years"
label var male "Male"

 bys Country HH1 HH2: gen NumU5=_N
 label var NumU5 "Number of U5 children"

end

start_final
tabplot Country WQ15_g, showval(format(%9.0f)) bfcolor(blue*0.6) subtitle("Frequency") ///
    xla(, angle(45) labsize(small)) yla(, labsize(small)) xsize(10) ysize(12)
graph export "${Figures}Freq_Country_Treatment_HH.eps", replace

start_final_U5			   
tabplot Country WQ15_g, showval(format(%9.0f)) bfcolor(blue*0.6) subtitle("Frequency") ///
    xla(, angle(45) labsize(small)) yla(, labsize(small)) xsize(10) ysize(12)
graph export "${Figures}Freq_Country_Treatment_Child.eps", replace

*-----------------------------
*  Table 0b: Extended descriptive statistics by source contamination
*-----------------------------
start_final_U5	
replace diarrhea=diarrhea*100		   

global U5Main diarrhea NumU5 age male
			   
tab1 $U5Main	
replace water_carrier_edu = . if water_carrier_edu == 98
mdesc $U5Main

local U5Main  "Child Characteristics by Water Source Contamination Level among Households with U5 children"
local LabelU5Main "Desc2"
local noteU5Main  "Notes: The table presents summary statistics for children under five years of age from 25 countries included in the analysis sample. Diarrhea is measured as a binary indicator and is reported as a percentage of children who experienced diarrhea in the past two weeks."

foreach k in U5Main {
    * Means by contamination level
	eststo model0: estpost summarize $`k' [aw=hhweight] 
    eststo model1: estpost summarize $`k' [aw=hhweight] if RiskSource==0
    eststo model2: estpost summarize $`k' [aw=hhweight] if RiskSource==1
    eststo model3: estpost summarize $`k' [aw=hhweight] if RiskSource==2
	
	* Min
start_final_U5	
replace diarrhea=diarrhea*100		   

	foreach i in $`k' {
	egen min_`i'=min(`i')
	replace `i'=min_`i'
	}
	eststo  model6: estpost summarize $`k'
	
* Max
start_final_U5	
replace diarrhea=diarrhea*100		   

	foreach i in $`k' {
	egen max_`i'=max(`i')
	replace `i'=max_`i'
	}
	eststo  model7: estpost summarize $`k'
	
* Missing 
start_final_U5	
replace diarrhea=diarrhea*100		   
	
	foreach i in $`k' {
	egen `i'_Miss=rowmiss(`i')
	egen max_`i'=sum(`i'_Miss)
	replace `i'=max_`i'
	}
	eststo  model8: estpost summarize $`k'

    esttab model0 model1 model2 model3 using "${Tables}Descript_`k'_Risk.tex", ///
        title("``k''" \label{`Label`k''}) ///
        cell("mean (fmt(2) label(_))") stats(N, fmt("%9.0fc") label(N)) ///
        mtitles("All" "Low risk" "Moderate risk" "High risk") nonum ///
        substitute( ///
            ".00" "" ///
            "{l}{\footnotesize" "{p{0.8\linewidth}}{\footnotesize" ///
            "&           _&           _&           _&           _\\" "" ///
            "Diarrhea" "Diarrhea (percent)" ///
            "Piped water" "\textbf{Primary water source} \\\hline Piped connection" ///
            "12,271" "12,062 (21.3\%)" ///
            "-0 " "0" ///
            "Treat:" "~~~" ///
            "Location:" "~~~" ///
        ) ///
        label note("`note`k''") ///
        replace
}

eststo clear

*-----------------------------
*  Table 0b: Extended descriptive statistics by source contamination
*-----------------------------
start_final
* water_carrier_edu_0 water_carrier_edu_1 water_carrier_edu_2 water_carrier_edu_3 ///
	           water_carrier_edu_4 water_carrier_edu_5 water_carrier_edu_6 water_carrier_edu_98
			   
global ExtMain windex5_1 windex5_2 windex5_3 windex5_4 windex5_5 ///
			   helevel_0 helevel_1 helevel_2 helevel_98 ///
			   WS1_g_11 WS1_g_21 WS1_g_31 WS1_g_32 WS1_g_51 WS1_g_91 WS1_g_96 ///
			   Toilet_1 Toilet_2 Toilet_3 Toilet_98 ///
               Any_U5 Girls_less_than15 Boys_15or_less urban ///
			   water_treatment WQ15_g_0 WQ15_g_1 WQ15_g_2 WQ15_g_3 WQ15_g_98 ///
	           WQ27 WQ26
			   
tab1 $ExtMain
	
replace water_carrier_edu = . if water_carrier_edu == 98

	
mdesc $ExtMain
tab RiskSource, m

local ExtMain  "Household Characteristics by Water Source Contamination Level"
local LabelExt "Desc1"
local noteExt  "Notes: The table presents extended household characteristics across 25 countries. The number of CFUs per 100mL of E. coli is capped at 101 if the number is higher than 100. The mean of the blank water test is 0.67 CFUs per 100mL."

foreach k in ExtMain {
    * Means by contamination level
	eststo model0: estpost summarize $`k' [aw=hhweight] 
    eststo model1: estpost summarize $`k' [aw=hhweight] if RiskSource==0
    eststo model2: estpost summarize $`k' [aw=hhweight] if RiskSource==1
    eststo model3: estpost summarize $`k' [aw=hhweight] if RiskSource==2
	
	* Min
start_final
	foreach i in $`k' {
	egen min_`i'=min(`i')
	replace `i'=min_`i'
	}
	eststo  model6: estpost summarize $`k'
	
* Max
start_final 
	foreach i in $`k' {
	egen max_`i'=max(`i')
	replace `i'=max_`i'
	}
	eststo  model7: estpost summarize $`k'
	
* Missing 
start_final 
	
	foreach i in $`k' {
	egen `i'_Miss=rowmiss(`i')
	egen max_`i'=sum(`i'_Miss)
	replace `i'=max_`i'
	}
	eststo  model8: estpost summarize $`k'

    esttab model0 model1 model2 model3 using "${Tables}Descript_`k'_Risk_ext.tex", ///
        title("``k''" \label{`LabelExt'}) ///
        cell("mean (fmt(2) label(_))") stats(N, fmt("%9.0fc") label(N)) ///
        mtitles("All" "Low risk" "Moderate risk" "High risk") nonum ///
        substitute( ///
            ".00" "" ///
            "{l}{\footnotesize" "{p{0.96\linewidth}}{\footnotesize" ///
            "&           _&           _&           _&           _\\" "" ///
            ///
            "Piped water" "\textbf{Primary water source} \\\hline Piped connection" ///
            "Tube well or borehole" "Tube well / borehole" ///
            "Protected well" "Protected well or spring" ///
            "Unprotected well" "Unprotected well or spring" ///
            "Protected spring" "Protected spring" ///
            "Unprotected spring" "Unprotected spring" ///
            "Rainwater" "Surface or rainwater" ///
            "Tanker truck or cart" "Packaged / bottled water" ///
            "Surface water" "Surface or rainwater" ///
            ///
            "Source water test (100ml)" "\textbf{Water test results} \\\hline Source sample (CFU/100mL)" ///
            "Point of use water test (100ml)" "Household water test (CFU/100mL)" ///
            ///
            "Poorest" "\textbf{Household socioeconomic status} \\\hline Poorest quintile" ///
            "Any treatment (Water tested)" "\textbf{Water treatment for tested water} \\\hline Any treatment" ///
			"Flush toilet" "\textbf{Sanitation facility} \\\hline Flush toilet" ///
            "Chlorinate" "Chlorine / Aquatabs / PUR" ///
            "Solar disinfection" "Strain or settle" ///
            "Other treatment" "Other method" ///
            "Girls_less_than15" "Water carrier: Girls younger than 15" "Boys_15or_less" "Water carrier: Boys younger than 15" ///
            "No education" "\textbf{HH Education} \\\hline No education" ///
            "Have U5 children " "\textbf{Household Demographic} \\\hline Have any children under age 5" ///
            "PipedWater" "Piped source" ///
            "WellandSpringWater" "Well or spring source" ///
            ///
            "Water carrier education" "\textbf{Water collector characteristics} \\\hline Years of education (non-missing)" ///
            ///
            "25,518" "23,595 (41.6\%)" ///
            "21,844" "21,064 (37.1\%)" ///
            "12,271" "12,062 (21.3\%)" ///
            "-0 " "0" ///
            "Treat:" "~~~" ///
            "Location:" "~~~" ///
        ) ///
        label note("`noteExt'") ///
        replace
}

eststo clear

END


/*

/*-----------------------------
     Table 0: Desciptive statistics by the level of source water contamination
-----------------------------*/

start_final
sum WQ29
tab water_carrier_edu,m
* WS9 WS10_0 WS10_1 WS10_2 WS10_3 WS10_6 WS10_7 WS10_98 WS10_99
global Main urban windex5_1 windex5_2 windex5_3 windex5_4 windex5_5 ///
			WS1_g_11 WS1_g_21 WS1_g_31 WS1_g_32 WS1_g_51 WS1_g_91 WS1_g_96 ///
			water_treatment  ///
			WQ15_g_0 WQ15_g_1 WQ15_g_2 WQ15_g_3 WQ15_g_98 ///
			WQ27 WQ26
			
mdesc $Main
tab RiskSource,m

local Main "Household Characteristics by Water Source Contamination Level (share)"
local LabelMain "Desc1"
local noteMain "Notes: The table presents the household characteristics across 25 countries. The number of CFUs per 100mL of E. Coli is capped at 101 if the number is higher than 100. The mean of the blank water test is 0.67 CFUs per 100mL."
					 
foreach k in Main {
* Mean
	eststo  model0: estpost summarize $`k' [aw=hhweight] if RiskSource==0
	eststo  model1: estpost summarize $`k' [aw=hhweight] if RiskSource==1
	eststo  model2: estpost summarize $`k' [aw=hhweight] if RiskSource==2

esttab model0 model1 model2 using "${Tables}Descript_`k'_Risk.tex", title("``k''" \label{`Label`k''}) ///
	   cell("mean (fmt(2) label(_))") stats(N, fmt("%9.0fc") label(N) ) /// 
	   mtitles("Low risk" "Moderate risk" "High risk") nonum ///
	   substitute( ".00" "" "{l}{\footnotesize" "{p{0.96\linewidth}}{\footnotesize" ///
				   "                    &           _&           _&           _\\" "" ///
				   "Piped water" "\textbf{Primary water source} \\\hline Piped water" ///
				   "Location: In own dwelling" "\textbf{Location} \\\hline Location: In own dwelling" ///
                   "Source water test (100ml)" "\textbf{Water test results} \\\hline Source water test (100ml)" ///
				   "Poorest" "\textbf{Socioeconomic level} \\\hline Poorest" ///
				   "Basic water service" "\textbf{Water source category} \\\hline Basic water service" ///
				   "Any water treatment for primary" "\textbf{Primary water source} \\\hline Any water treatment" ///
				   "25,518" "25,518 (42.8\%)" "21,844" "21,844 (36.6\%)" "12,271" "12,271 (20.6\%)" ///  
				   "-0 " "0" ///
				   "Treat:"  "~~~" "Location:"  "~~~" ///
				   ) ///
	   label  note("`note`k''")  ///
	   replace 
	   }
eststo clear

