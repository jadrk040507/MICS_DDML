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
	
	global Tables      "${Overleaf}MICS_Water/Table/"
	global Figures     "${Overleaf}MICS_Water/Figure/"
	* global Data_Raw   "${Dropbox}MICS_DDML/Data/1. Raw/"
	global Data_Clean "${Dropbox}MICS_DDML/Data/2. Clean/"
	global Data_Final "${Dropbox}MICS_DDML/Data/3. Final/"
	

clear all               
set graph off
set graph on	


/*---------------------------------------------------     E-Coli - Water Treatment ---------------------------------------------------*/
	
	use "${Data_Clean}Cleaned_Pooled_MICS6_Africa_Latam_Asia_2.dta", clear
	* Dropping
	* Tonga (27), Tuvalu (30), Kiribati (17), Turks (29)
	drop if country_cat==30   | country_cat==17  | country_cat==27 | country_cat==29
	* 59,704
	count

	* drop if Country=="Guinea Bissau"
	replace WQ29=. if WQ29==998
	recode  WS1 61 62=61 71 72=71 91 92=91
	replace WS3=998 if WS3==.
	recode  WS3 4 9=998

	foreach i in WS10 WQ15 {
	gen      `i'=0
	replace  `i'=1 if `i'A=="A"
	replace  `i'=2 if `i'B=="B"
	replace  `i'=3 if `i'C=="C"
	replace  `i'=4 if `i'D=="D"
	replace  `i'=5 if `i'E=="E"
	replace  `i'=6 if `i'F=="F"
	replace  `i'=7 if `i'G=="G"
	replace  `i'=8 if `i'H=="H"
	replace  `i'=98 if `i'X=="X"
	replace  `i'=99 if `i'Z=="Z"
	replace  `i'=998 if `i'NR=="?"	
	recode   `i' 4 5 8=98
	recode   `i' 998=99
	}
	
	* WQ15 (water treatment methods are missing if WQ14 is do not know or no response)
	tab     WQ14 water_treatment,m
	replace WQ15=. if WQ14==8 | WQ14==9
	
	label define WS10l 0 "Treat: Nothing" 1 "Treat: Boil" 2 "Treat: Bleach/Chlorine" 3 "Treat: Stain with a cloth" 4 "Treat: Filter" 5 "Treat: Soler" 6 "Treat: Let it settle" 7 "Treat: Aquatabs/PUR" 8 "Treat: Add tablet" 98 "Treat: Other" 99 "Treat: Do not know/missing", modify
	label values WS10 WQ15 WS10l
	
	* Grouping water treatment
	gen    WQ15_g=WQ15
	recode WQ15_g 2 7=2 3 6=3
	label define WQ15_gl 0 "Treat: Nothing" 1 "Treat: Boil" 2 "Treat: Chlorine/Aquatabs/PUR" 3 "Treat: Strain/Settle" 4 "Treat: Filter" 5 "Treat: Soler" 8 "Treat: Add tablet" 98 "Treat: Other" 99 "Treat: Do not know/missing", modify
	label values WQ15_g WQ15_gl
	
	* Grouping water treatment
	gen    WS10_g=WS10
	recode WS10_g 2 7=2 3 6=3
	label define WS10_gl 0 "Treat: Nothing" 1 "Treat: Boil" 2 "Treat: Chlorine/Aquatabs/PUR" 3 "Treat: Strain/Settle" 4 "Treat: Filter" 5 "Treat: Soler" 8 "Treat: Add tablet" 98 "Treat: Other" 99 "Treat: Do not know/missing", modify
	label values WS10_g WS10_gl
	
	gen    WS1_g=WS1
	recode WS1_g 11/14=11 31 41=31 32 42=32 61 71 .=96 51 81=51
	label define WS1_gl 11 "Piped water" 21 "Tube/Well/Borehole" 31 "Protected well/spring" 32 "Unprotected well/spring" 51 "Surface/Rain water" 91 "Packaged/Bottled water" 96 "Others", modify
	label values WS1_g WS1_gl
	label define windex5l 1 "Poorest" 2 "Poor" 3 "Middle" 4 "Rich" 5 "Richest", modify
	label values windex5 windex5l

	* Create Dummy
	replace water_carrier_edu=98 if water_carrier_edu==.
	
	foreach v in WS1 WS3 WS10 WQ15 WQ15_g WS1_g helevel water_carrier_edu windex5 {
	levelsof `v'
	foreach value in `r(levels)' {
		gen     `v'_`value'=0
		replace `v'_`value'=1 if `v'==`value'
		replace `v'_`value'=. if `v'==.
		label var `v'_`value' "`: label (`v') `value''"
	}
	}
	
	label var WS1_11 "Piped water (Dwelling)"
	label var WS1_12 "Piped water (Yard/plot)"
	label var WS1_13 "Piped water (Neighbor)"
	label var WS1_14 "Piped water (Public)"
	label var WS1_21 "Borehall"
	label var WS1_31 "Protected well"	
	label var WS1_32 "Unprotected well"	
	label var WS1_41 "Protected spring"	
	label var WS1_42 "Unprotected spring"	
	label var WS1_81 "Surface water"
	label var WS1_91 "Packaged water (Sachet/bottle)"
	label var WS3_1 "Location: In own dwelling"
	label var water_treatment "Any treatment (Water tested)"
	label var WS9 "Any water treatment for primary"
	label var urban "Urban"
	label var Basic_water_service "Basic water service"
	label var Limited_water_service "Limited water service"
	label var Surface_water_service "Surface water service"
	label var Unimproved_water_service "Unimproved water service"
	
	recode WS9 2=0
	recode water_treatment 2=0

	*comment to Akito: the following is the code for water storage
	replace WQ12 = . if WQ12 >= 8
		
		gen water_straight_from_source = 1 if WQ12 == 1
		replace water_straight_from_source = 0 if WQ12 != 1 & WQ12 != .
		
		gen water_stored_covered = 1 if WQ12 == 2
		replace water_stored_covered = 0 if WQ12 == 1 | WQ12 == 3	
		
		gen water_stored_uncovered = 1 if WQ12 == 3
		replace water_stored_uncovered = 0 if WQ12 == 1 | WQ12 == 2

		* Initialize the variable to 0 (not rainy season)
				gen rainy_season = 0

				* Sierra Leone
				replace rainy_season = 1 if Country == "Sierra Leone" & (HH5M >= 5 & HH5M <= 11)

				* Benin
				replace rainy_season = 1 if Country == "Benin" & ((HH5M >= 3 & HH5M <= 7) | (HH5M == 9 | HH5M == 10))

				* Central African Republic
				replace rainy_season = 1 if Country == "Central African Republic" & (HH5M >= 4 & HH5M <= 10)

				* Chad
				replace rainy_season = 1 if Country == "Chad" & (HH5M >= 6 & HH5M <= 9)

				* DR Congo
				replace rainy_season = 1 if Country == "DR Congo" & (HH5M >= 11 | HH5M <= 3)

				* Eswatini (Swaziland)
				replace rainy_season = 1 if Country == "Eswatini" & (HH5M >= 10 | HH5M <= 3)

				* The Gambia
				replace rainy_season = 1 if Country == "Gambia" & (HH5M >= 6 & HH5M <= 10)

				* Ghana
				replace rainy_season = 1 if Country == "Ghana" & (HH5M >= 4 & HH5M <= 11)

				* Guinea Bissau
				replace rainy_season = 1 if Country == "Guinea Bissau" & (HH5M >= 6 & HH5M <= 10)

				* Lesotho
				replace rainy_season = 1 if Country == "Lesotho" & (HH5M >= 10 | HH5M <= 4)

				* Madagascar
				replace rainy_season = 1 if Country == "Madagascar" & (HH5M >= 11 | HH5M <= 4)

				* Malawi
				replace rainy_season = 1 if Country == "Malawi" & (HH5M >= 11 | HH5M <= 4)

				* Nigeria
				replace rainy_season = 1 if Country == "Nigeria" & (HH5M >= 4 & HH5M <= 10)

				* Togo
				replace rainy_season = 1 if Country == "Togo" & ((HH5M >= 4 & HH5M <= 7) | (HH5M >= 9 & HH5M <= 11))

				* Zimbabwe
				replace rainy_season = 1 if Country == "Zimbabwe" & (HH5M >= 11 | HH5M <= 3)
				
replace PSU=psu if PSU==.
replace PSU=HH1 if country_cat==12
replace PSU=HH1 if country_cat==18
replace PSU=HH1 if country_cat==24

egen Cluster_var=group(country_cat PSU)

gen    NoRiskHome_0_12=RiskHome
recode NoRiskHome_0_12 0=1 1 2=0

gen    NoRiskHome_01_2=RiskHome
recode NoRiskHome_01_2 0 1=1 2=0

gen    RiskHome_0_12=RiskHome
recode RiskHome_0_12 0=0 1 2=1
tab WQ15_g RiskSource,m
gen    RiskSource_0_12=RiskSource
recode RiskSource_0_12 0=0 1 2=1

label var RiskSource_0_12 "Some E.Coli"

gen     water_treatment3=water_treatment
foreach i in C F {
replace water_treatment3=2 if  WQ15`i'=="`i'"
}

gen       Any_U5 =HH55
recode    Any_U5 0=0 1/20=1
label var Any_U5 "Have U5 children"

gen     Region=.
replace Region=1 if Country=="Benin" | Country=="Central African Republic" | Country=="Chad" | Country=="DR Congo" | Country=="Eswatini" | Country=="Gambia" | Country=="Ghana" | Country=="Guinea Bissau" | Country=="Lesotho" | Country=="Madagascar" | Country=="Malawi" | Country=="Sierra Leone" | Country=="Togo" | Country=="Zimbabwe"
replace Region=3 if Country=="Bangladesh" | Country=="Lao"  | Country=="Mongolia"  | Country=="Nepal" | Country=="Viet Nam"
replace Region=2 if Country=="Dominican Republic" | Country=="Fiji" | Country=="Guyana" | Country=="Honduras"  | Country=="Jamaica" | Country=="Kiribati" | Country=="Tonga"  | Country=="Trinidad and Tobago" | Country=="Turks and Caocos Islands"  | Country=="Tuvalu" | Country=="Suriname"

* No water treatment response recorded
* Since this is the same as water_treatment
* X samples are dropped since water treatment is missing (Household responded do not know or no response to water treatment question)
tab  WQ14 water_treatment,m
drop if (WQ14==8 | WQ14==9 | WQ14==.)
drop WQ14
replace WS1 =96 if WS1==.

save "${Data_Final}MASTER_MICS_DDML.dta", replace

/*---------------------------------------------------     E-Coli - Diarrhea (HH With children under age 5) ---------------------------------------------------*/

use "${Data_Clean}MASTER_MICS_U4_INDIV_U5_WQ.dta", clear
* Comment to Sujey: How do you treat do not know or missing for cought? I though this should be either missing or 0
	*response: not know and missing are coded as missing (.). The cleaned sickness variables are: fever and diarrhea
* gen Cough=CA16
* recode Cough 1=1 2=0 8 9=.
* Discuss why the raw data has changed. It shouldn't have. Can you explain what you mean?
drop WS15

	replace WQ29=. if WQ29==998
	recode  WS1 61 62=61 71 72=71 91 92=91
	replace WS3=998 if WS3==.
	recode WS3 4 9=998
	
	drop if WS9==8 | WS9==9 
	drop if water_treatment==8 | water_treatment==9

	foreach i in WS10 WQ15 {
	gen      `i'=0
	replace  `i'=1 if `i'A=="A"
	replace  `i'=2 if `i'B=="B"
	replace  `i'=3 if `i'C=="C"
	replace  `i'=4 if `i'D=="D"
	replace  `i'=5 if `i'E=="E"
	replace  `i'=6 if `i'F=="F"
	replace  `i'=7 if `i'G=="G"
	replace  `i'=8 if `i'H=="H"
	replace  `i'=98 if `i'X=="X"
	replace  `i'=99 if `i'Z=="Z"
	replace  `i'=998 if `i'NR=="?"	
	recode   `i' 4 5 8=98
	recode   `i' 998=99
	}
	
	tab  WQ15 water_treatment,m
	label define WS10l 0 "Treat: Nothing" 1 "Treat: Boil" 2 "Treat: Bleach/Chlorine" 3 "Treat: Strain with a cloth" 4 "Treat: Filter" 5 "Treat: Soler" 6 "Treat: Let it settle" 7 "Treat: Aquatabs/PUR" 8 "Treat: Add tablet" 98 "Treat: Other" 99 "Treat: Do not know/missing", modify
	label values WS10 WQ15 WS10l
	
	* Grouping water treatment
	gen    WQ15_g=WQ15
	recode WQ15_g 2 7=2 3 6=3
	label define WQ15_gl 0 "Treat: Nothing" 1 "Treat: Boil" 2 "Treat: Chlorine/Aquatabs/PUR" 3 "Treat: Strain/Settle" 4 "Treat: Filter" 5 "Treat: Soler" 8 "Treat: Add tablet" 98 "Treat: Other" 99 "Treat: Do not know/missing", modify
	label values WQ15_g WQ15_gl
	
	* Grouping water treatment
	gen    WS10_g=WS10
	recode WS10_g 2 7=2 3 6=3
	label define WS10_gl 0 "Treat: Nothing" 1 "Treat: Boil" 2 "Treat: Chlorine/Aquatabs/PUR" 3 "Treat: Strain/Settle" 4 "Treat: Filter" 5 "Treat: Soler" 8 "Treat: Add tablet" 98 "Treat: Other" 99 "Treat: Do not know/missing", modify
	label values WS10_g WS10_gl
	
	recode helevel 2 3 4=2
	* recode helevel 3 4=3
	label define helevell 0 "No education" 1 "Primary" 2 "Sec or higher", modify
	label values helevel helevell
	tab helevel Region,m

	* Create Dummy
	foreach v in WS1 WS3 WS10 WQ15 WQ15_g WS10_g {
	levelsof `v'
	foreach value in `r(levels)' {
		gen     `v'_`value'=0
		replace `v'_`value'=1 if `v'==`value'
		replace `v'_`value'=. if `v'==.
		label var `v'_`value' "`: label (`v') `value''"
	}
	}
	
	label var WS1_11 "Piped water (Dwelling)"
	label var WS1_12 "Piped water (Yard/plot)"
	label var WS1_13 "Piped water (Neighbor)"
	label var WS1_14 "Piped water (Public)"
	label var WS1_21 "Borehall"
	label var WS1_31 "Protected well"	
	label var WS1_32 "Unprotected well"	
	label var WS1_41 "Protected spring"	
	label var WS1_42 "Unprotected spring"	
	label var WS1_81 "Surface water"
	label var WS1_91 "Packaged water (Sachet/bottle)"
	label var WS3_1 "Location: In own dwelling"
	label var water_treatment "Any treatment (Water tested)"
	label var WS9 "Any water treatment for primary"
	label var urban "Urban"
	label var Basic_water_service "Basic water service"
	label var Limited_water_service "Limited water service"
	label var Surface_water_service "Surface water service"
	label var Unimproved_water_service "Unimproved water service"
	
	recode WS9 2=0
	recode water_treatment 2=0
	
	* Initialize the variable to 0 (not rainy season)
				gen rainy_season = 0

				* Sierra Leone
				replace rainy_season = 1 if Country == "Sierra Leone" & (HH5M >= 5 & HH5M <= 11)

				* Benin
				replace rainy_season = 1 if Country == "Benin" & ((HH5M >= 3 & HH5M <= 7) | (HH5M == 9 | HH5M == 10))

				* Central African Republic
				replace rainy_season = 1 if Country == "Central African Republic" & (HH5M >= 4 & HH5M <= 10)

				* Chad
				replace rainy_season = 1 if Country == "Chad" & (HH5M >= 6 & HH5M <= 9)

				* DR Congo
				replace rainy_season = 1 if Country == "DR Congo" & (HH5M >= 11 | HH5M <= 3)

				* Eswatini (Swaziland)
				replace rainy_season = 1 if Country == "Eswatini" & (HH5M >= 10 | HH5M <= 3)

				* The Gambia
				replace rainy_season = 1 if Country == "Gambia" & (HH5M >= 6 & HH5M <= 10)

				* Ghana
				replace rainy_season = 1 if Country == "Ghana" & (HH5M >= 4 & HH5M <= 11)

				* Guinea Bissau
				replace rainy_season = 1 if Country == "Guinea Bissau" & (HH5M >= 6 & HH5M <= 10)

				* Lesotho
				replace rainy_season = 1 if Country == "Lesotho" & (HH5M >= 10 | HH5M <= 4)

				* Madagascar
				replace rainy_season = 1 if Country == "Madagascar" & (HH5M >= 11 | HH5M <= 4)

				* Malawi
				replace rainy_season = 1 if Country == "Malawi" & (HH5M >= 11 | HH5M <= 4)

				* Nigeria
				replace rainy_season = 1 if Country == "Nigeria" & (HH5M >= 4 & HH5M <= 10)

				* Togo
				replace rainy_season = 1 if Country == "Togo" & ((HH5M >= 4 & HH5M <= 7) | (HH5M >= 9 & HH5M <= 11))

				* Zimbabwe
				replace rainy_season = 1 if Country == "Zimbabwe" & (HH5M >= 11 | HH5M <= 3)

* Since this is the same as water_treatment
drop WQ14

gen     water_treatment3=water_treatment
foreach i in C F {
replace water_treatment3=2 if  WQ15`i'=="`i'"
}

drop if country_cat==30  | country_cat==11 | country_cat==17  | country_cat==27 | country_cat==29

save "${Data_Final}MASTER_MICS_U5_DDML.dta", replace



/* Previous paper


/*-----------------------------
     Table 2: The likelihood of applying water treatment by the intial contamination level of source water and education level
-----------------------------*/
start_clean
gen     windex_ur=windex5u
replace windex_ur=windex5r if windex_ur==.
recode windex_ur 1 2=1 3 4 5=2
gen          windex5_categ=windex5
recode       windex5_categ 1 2=1 3=2 4 5=3
label define windex5_categl 1 "Poor" 2 "Middle" 3 "Rich/Richest", modify
label values windex5_categ windex5_categl

gen Diff_Qual=WQ26-WQ27
tab helevel,m
global ControlsVar i.country_cat i.urban i.WS1_g i.windex5_categ 
mdesc country_cat urban WS1_g windex5_categ water_treatment hhweight RiskSource Cluster_var

eststo: reg water_treatment i.RiskSource  $ControlsVar , cluster(Cluster_var)
sum water_treatment 
estadd scalar Mean = r(mean)

eststo: reg water_treatment i.RiskSource  $ControlsVar if Region==1 , cluster(Cluster_var)
sum water_treatment if Region==1
estadd scalar Mean = r(mean)
eststo: reg water_treatment i.RiskSource  $ControlsVar if Region==2 , cluster(Cluster_var)
sum water_treatment if Region==2
estadd scalar Mean = r(mean)
eststo: reg water_treatment i.RiskSource  $ControlsVar if Region==3, cluster(Cluster_var)
sum water_treatment if Region==3
estadd scalar Mean = r(mean)

foreach i in WQ15_g_1 WQ15_g_2 WQ15_g_3 WQ15_g_98 {
eststo: reg `i' i.RiskSource  $ControlsVar , cluster(Cluster_var)
sum `i'
estadd scalar Mean = r(mean)
}

/* Socio interaction
eststo: reg water_treatment i.RiskSource_0_12##i.windex5_categ $ControlsVar if Region==1, cluster(Cluster_var)
sum water_treatment  if Region==1
estadd scalar Mean = r(mean)
*/

esttab using "${Table}Est_Treat_Risk_AF_LATAM_AS.tex",label se ar2 title("Water Source E.Coli Contamination on the Probability of Water Treatment" \label{ETR0}) nonotes nobase nocons ///
             indicate("Country FE= *country_cat" "Controls= *WS1_g *urban *windex5_categ") ///
			 mtitle("Overall" "Africa" "\shortstack[c]{Latin\\America}" "Asia" "Boil" "\shortstack[c]{Chlorine\\Aquatabs\\PUR}" "\shortstack[c]{Strain\\Settle}" "Other") ///
			 stats(Mean r2_a N, fmt(%9.2fc %9.2fc %9.0fc) labels(`"Mean"' `"Adjusted \(R^{2}\)"' `"Observations"')) ///
			 mgroups("\shortstack[c]{Dependent var: Water treated\\(yes = 1, no = 0)}" "\shortstack[c]{Dependent var: By method\\(yes = 1, no = 0)}", pattern(1 0 0 0 1 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) /// 
			 starlevels(\sym{*} 0.10 \sym{**} 0.05 \sym{***} 0.010) b(2) ///
			 substitute("{l}{\footnotesize" "{p{1\linewidth}}{\footnotesize" ///
			 "=1" "" ///
			 "Moderate to high risk&" "\multicolumn{5}{l}{Base: Low risk} \\ Moderate to high risk&" ///
			 ) ///
			 addnote("Note: All regressions control for country fixed effects, water source type, urban or rural location, and household socioeconomic status based on an asset-based wealth index. Standard errors are clustered at the primary sampling unit level (in parentheses). Significance levels are indicated as follows: $\sym{*} p<.10,\sym{**} p<.05,\sym{***} p<.01$. $\sym{*}$") ///	
			 replace
eststo clear

/*-----------------------------
     Table 2.1: By the source
-----------------------------*/
start_clean
gen     windex_ur=windex5u
replace windex_ur=windex5r if windex_ur==.
recode windex_ur 1 2=1 3 4 5=2
gen          windex5_categ=windex5
recode       windex5_categ 1 2=1 3=2 4 5=3
label define windex5_categl 1 "Poor" 2 "Middle" 3 "Rich/Richest", modify
label values windex5_categ windex5_categl

recode helevel 2 3 4=2
* recode helevel 3 4=3
label define helevell 0 "No education" 1 "Primary" 2 "Sec or higher", modify
label values helevel helevell
tab helevel Region,m

gen Diff_Qual=WQ26-WQ27
tab helevel,m
global ControlsVar i.country_cat i.urban i.WS1_g i.windex5_categ 
mdesc country_cat urban WS1_g windex5_categ water_treatment hhweight RiskSource Cluster_var

eststo: reg water_treatment i.RiskSource  $ControlsVar , cluster(Cluster_var)
sum water_treatment 
estadd scalar Mean = r(mean)

foreach i in WS1_g_11 WS1_g_21 WS1_g_31 WS1_g_32 WS1_g_51 WS1_g_91 {
eststo: reg water_treatment i.RiskSource  $ControlsVar if `i'==1 , cluster(Cluster_var)
sum `i'
estadd scalar Mean = r(mean)
}

esttab using "${Table}Est_Treat_Risk_AF_LATAM_AS_Source.tex", ///
             label se ar2 title("Water Source E.Coli Contamination on the Probability of Water Treatment" \label{ETRSource}) nonotes nobase nocons ///
             indicate("Country FE= *country_cat" "Controls= *WS1_g *urban *windex5_categ") ///
			 mtitle("Overall" "\shortstack[c]{Piped\\water}" "\shortstack[c]{Tube\\Well\\Borehole}" "\shortstack[c]{Protected\\well\\spring}" ///
			        "\shortstack[c]{Unprotected\\well\\spring}" "\shortstack[c]{Surface\\Rainfall}" "\shortstack[c]{Packaged\\Bottle}") ///
			 stats(Mean r2_a N, fmt(%9.2fc %9.2fc %9.0fc) labels(`"Mean"' `"Adjusted \(R^{2}\)"' `"Observations"')) ///
			 mgroups("\shortstack[c]{Dependent var: Water treated (yes = 1, no = 0)}", pattern(1 0 0 0 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) /// 
			 starlevels(\sym{*} 0.10 \sym{**} 0.05 \sym{***} 0.010) b(2) ///
			 substitute("{l}{\footnotesize" "{p{1\linewidth}}{\footnotesize" ///
			 "=1" "" ///
			 "Moderate to high risk&" "\multicolumn{5}{l}{Base: Low risk} \\ Moderate to high risk&" ///
			 ) ///
			 addnote("Note: All regressions control for country fixed effects, water source type, urban or rural location, and household socioeconomic status based on an asset-based wealth index. Standard errors are clustered at the primary sampling unit level (in parentheses). Significance levels are indicated as follows: $\sym{*} p<.10,\sym{**} p<.05,\sym{***} p<.01$. $\sym{*}$") ///	
			 replace
eststo clear


END



/*-----------------------------
     Table 0: Desciptive statistics by the level of source water contamination
-----------------------------*/
start_clean
sum WQ29
tab water_carrier_edu,m
* WS9 WS10_0 WS10_1 WS10_2 WS10_3 WS10_6 WS10_7 WS10_98 WS10_99
global Main urban windex5_1 windex5_2 windex5_3 windex5_4 windex5_5 ///
			WS1_g_11 WS1_g_21 WS1_g_31 WS1_g_32 WS1_g_51 WS1_g_91 WS1_g_96 ///
			water_treatment  ///
			WQ15_g_0 WQ15_g_1 WQ15_g_2 WQ15_g_3 WQ15_g_98 ///
			WQ27 WQ26
			*  WQ15_g_99  WQ29  rainy_season Any_U5
			
mdesc $Main
tab RiskSource,m

local Main "Household Characteristics by the Level of Water Source Contamination"
local LabelMain "Desc1"
local noteMain "Notes: The table presents the household characteristics across 25 countries. The number of CFUs per 100mL of E. Coli is capped at 101 if the number is higher than 100. The mean of the blank water test is 0.67 CFUs per 100mL."
					 
foreach k in Main {
* Mean
	eststo  model0: estpost summarize $`k' [aw=hhweight] if RiskSource==0
	eststo  model1: estpost summarize $`k' [aw=hhweight] if RiskSource==1
	eststo  model2: estpost summarize $`k' [aw=hhweight] if RiskSource==2

esttab model0 model1 model2 using "${Table}Descript_`k'_Risk.tex", title("``k''" \label{`Label`k''}) ///
	   cell("mean (fmt(2) label(_))") stats(N, fmt("%9.0fc") label(Observations) ) /// 
	   mtitles("Low risk" "Medium risk" "High risk") nonum ///
	   substitute( ".00" "" "{l}{\footnotesize" "{p{0.85\linewidth}}{\footnotesize" ///
				   "                    &           _&           _&           _\\" "" ///
				   "Piped water" "\textbf{Primary Water Source} \\\hline Piped water" ///
				   "Location: In own dwelling" "\textbf{Location} \\\hline Location: In own dwelling" ///
                   "Source water test (100ml)" "\textbf{Water test results} \\\hline Source water test (100ml)" ///
				   "Poorest" "\textbf{Socioeconomic level} \\\hline Poorest" ///
				   "Basic water service" "\textbf{Water source category} \\\hline Basic water service" ///
				   "Any water treatment for primary" "\textbf{Primary Water Source} \\\hline Any water treatment" ///
				   "-0 " "0" ///
				   "Treat:"  "~~~" "Location:"  "~~~" ///
				   ) ///
	   label  note("`note`k''")  ///
	   replace 
	   }
eststo clear


End




/*-----------------------------
     Table 0: Desciptive statistics by region
-----------------------------*/
start_clean
tab water_carrier_edu,m
* WS9 WS10_0 WS10_1 WS10_2 WS10_3 WS10_6 WS10_7 WS10_98 WS10_99
global Main urban Any_U5 windex5_1 windex5_2 windex5_3 windex5_4 windex5_5 ///
			WS1_g_11 WS1_g_21 WS1_g_31 WS1_g_32 WS1_g_51 WS1_g_91 WS1_g_96 ///
			water_treatment  ///
			WQ15_g_0 WQ15_g_1 WQ15_g_2 WQ15_g_3 WQ15_g_98 ///
			WQ27 WQ26
			*  WQ15_g_99  WQ29  rainy_season Any_U5
			
* Basic_water_service Limited_water_service Surface_water_service Unimproved_water_service  ///
* water_straight_from_source water_stored_covered water_stored_uncovered

local Main "Household characteristics by the level of source water contamination"
local LabelMain "Desc2"
local noteMain "Notes: The table presents the household characteristics across 24 countries. The number of CFUs per 100mL of E. Coli is capped at 101 if the number is higher than 100. The mean of the blank water test is 0.66 CFUs per 100mL."
					 
foreach k in Main {
* Mean
	eststo  model0: estpost summarize $`k' [aw=hhweight] if Region==1
	eststo  model1: estpost summarize $`k' [aw=hhweight] if Region==2
	eststo  model2: estpost summarize $`k' [aw=hhweight] if Region==3

esttab model0 model1 model2 using "${Table}Descript_`k'_Region.tex", title("``k''" \label{`Label`k''}) ///
	   cell("mean (fmt(2) label(_))") stats(N, fmt("%9.0fc") label(Observations) ) /// 
	   mtitles("Africa" "Latin America" "Asia") nonum ///
	   substitute( ".00" "" "{l}{\footnotesize" "{p{0.85\linewidth}}{\footnotesize" ///
				   "                    &           _&           _&           _\\" "" ///
				   "Piped water" "\textbf{Primary Water Source} \\\hline Piped water" ///
				   "Location: In own dwelling" "\textbf{Location} \\\hline Location: In own dwelling" ///
                   "Source water test (100ml)" "\textbf{Water test results} \\\hline Source water test (100ml)" ///
				   "Poorest" "\textbf{Socioeconomic level} \\\hline Poorest" ///
				   "Basic water service" "\textbf{Water source category} \\\hline Basic water service" ///
				   "Any water treatment for primary" "\textbf{Primary Water Source} \\\hline Any water treatment" ///
				   "-0 " "0" ///
				   "Treat:"  "~~~" "Location:"  "~~~" ///
				   ) ///
	   label  note("`note`k''")  ///
	   replace 
	   }
eststo clear



			                                                   /*-----------------------------
																    Understanding the data
																-----------------------------*/

/*-----------------------------
     Table 0: Selection
-----------------------------*/
foreach c in DominicanRepublic Honduras Suriname {
use  "${Data}Africa_Latam_Asia_Pooled_hh_data_2_Selection.dta", clear
replace Country="DominicanRepublic" if Country=="Dominican Republic"
keep if Country=="`c'"
keep if Selected==1

global Main urban Any_U5 windex5_1 windex5_2 windex5_3 windex5_4 windex5_5 ///
			WS1_g_11 WS1_g_21 WS1_g_31 WS1_g_32 WS1_g_51 WS1_g_91 WS1_g_96			
* Basic_water_service Limited_water_service Surface_water_service Unimproved_water_service  ///
* water_straight_from_source water_stored_covered water_stored_uncovered
* WS9 WS10_0 WS10_1 WS10_2 WS10_3 WS10_6 WS10_7 WS10_98 WS10_99

local Main "Household Characteristics for In/out Final Sample (`c')"
local LabelMain`c' "Desc`c'"
local noteMain "Notes: The table presents household characteristics for two groups: those included in the analysis sample—with complete information (consent, valid water quality data from both the source and stored drinking water, and valid water treatment data)—and those excluded from the analysis sample."
					 
foreach k in Main {
	eststo  model0: estpost summarize $`k' [aw=hhweight] if Reason==1 & Selected==1
	eststo  model1: estpost summarize $`k' [aw=hhweight] if Reason!=1 & Selected==1 
	
	preserve
	foreach i in $`k' {
	reg `i' i.Reason_1 [aw=hhweight]
	matrix b = r(table)
	scalar b_1 = b[1,2]
	replace `i'=b_1
	}
	eststo model3: estpost summarize $`k'
	
	restore
	foreach i in $`k' {
	reg `i' i.Reason_1 [aw=hhweight]
	matrix b = r(table)
	scalar p_1 = b[4,2]
	replace `i'=p_1
	}
	eststo model4: estpost summarize $`k'

esttab model0 model1 model3 model4 using "${Table}Descript_`k'_Selection_`c'.tex", title("``k''" \label{`Label`k'`c''}) ///
	   cell("mean (fmt(2) label(_))") stats(N, fmt("%9.0fc") label(Observations) ) /// 
	   mtitles("\shortstack[c]{In sample}" "\shortstack[c]{Out sample}" "Diff" "P-value") nonum ///
	   substitute( ".00" "" "{l}{\footnotesize" "{p{0.85\linewidth}}{\footnotesize" ///
				   " &           _&           _&           _&           _\\" "" ///
				   "Piped water" "\textbf{Primary Water Source} \\\hline Piped water" ///
				   "Location: In own dwelling" "\textbf{Location} \\\hline Location: In own dwelling" ///
                   "Source water test (100ml)" "\textbf{Water test results} \\\hline Source water test (100ml)" ///
				   "Poorest" "\textbf{Socioeconomic level} \\\hline Poorest" ///
				   "Basic water service" "\textbf{Water source category} \\\hline Basic water service" ///
				   "Any water treatment for primary" "\textbf{Primary water source} \\\hline Any water treatment" ///
				   "-0 " "0" "-0&" "0&" ///
				   "Treat:"  "~~~" "Location:"  "~~~" ///
				   ) ///
	   label  note("`note`k''")  ///
	   replace 
eststo clear
}
}


* 496,717
use "${Data}Africa_Latam_Asia_Pooled_hh_data_2.dta", clear
* Sample less than 1000: Tonga (27), Tuvalu (30), Kiribati (17), Turks (29)
drop if Country=="Tuvalu"   | Country=="Kiribati"  | Country=="Tonga" | Country=="Turks and Caocos Islands"
* Drop countries with no data (To be checked: Sujey)
drop if Country=="Afghanistan" | Country=="Costa Rica" | Country=="Argentina" | Country=="Cuba" | Country=="Jamaica"

* Drophousehold did not consent: 28,022 observations deleted)
drop if HH12==2

gen     Region=.
replace Region=1 if Country=="Benin" | Country=="Central African Republic" | Country=="Chad" | Country=="DR Congo" | Country=="Eswatini" | Country=="Gambia" | Country=="Ghana" | Country=="Guinea Bissau" | Country=="Lesotho" | Country=="Madagascar" | Country=="Malawi" | Country=="Sierra Leone" | Country=="Togo" | Country=="Zimbabwe"
replace Region=3 if Country=="Bangladesh" | Country=="Lao"  | Country=="Mongolia"  | Country=="Nepal" | Country=="Viet Nam"
replace Region=2 if Country=="Dominican Republic" | Country=="Fiji" | Country=="Guyana" | Country=="Honduras"  | Country=="Jamaica" | Country=="Kiribati" | Country=="Tonga"  | Country=="Trinidad and Tobago" | Country=="Turks and Caocos Islands"  | Country=="Tuvalu" | Country=="Suriname"
* drop if Country=="Cuba" | Country=="Jamaica" | Country=="Argentina"
replace Country="Central African Rep" if Country=="Central African Republic"

gen     urban = 1 if HH6 == 1
replace urban = 0 if HH6 == 2
gen       Any_U5 =HH55
recode    Any_U5 0=0 1/20=1
label var Any_U5 "Have U5 children"
gen    WS1_g=WS1
recode WS1_g 11/14=11 31 41=31 32 42=32 61 71 .=96 51 81=51
label define WS1_gl 11 "Piped water" 21 "Tube/Well/Borehole" 31 "Protected well/spring" 32 "Unprotected well/spring" 51 "Surface/Rain water" 91 "Packaged/Bottled water" 96 "Others", modify
label values WS1_g WS1_gl
label define windex5l 1 "Poorest" 2 "Poor" 3 "Middle" 4 "Rich" 5 "Richest", modify
label values windex5 windex5l

gen     Total_HH=1
replace HH9=2  if HH9A==2 & Country=="Bangladesh"
replace WQ31=. if HH9A==2 & Country=="Bangladesh"
gen     Selected=HH9
recode  Selected 2=0 1=1

* Total 415,005: 69354 selected
tab HH9 Selected,m

gen     Data_filled=0
replace Data_filled=1 if WQ26!=. &  WQ27!=.
* Togo has some missing value issue
replace WQ31=1 if WQ26!=. |  WQ27!=. &  Country=="Togo"
* 11 cases where results info itself is missing
replace WQ31=96 if WQ31==. & Selected==1
replace WQ31=. if HH9==2

gen     Reason=. 
replace Reason=0 if Selected==0
replace Reason=1 if WQ31==1
replace Reason=2 if WQ8==2
* Water not given 
replace Reason=3 if WQ31==2 |  WQ31==3 | WQ31==96
* WQ26 and WQ27 data (HHs that didn't have the water quality module)
* Dropping if HH missing: Both
replace Reason=3 if (WQ26 == . & WQ27 == .) & Reason==1
* Dropping if HH missing: Either
replace Reason=3 if (WQ26 == . | WQ27 == .) & Reason==1
* Dropping if Water testing results are "Not possible to read the results o" or other number
replace Reason=3 if (WQ26>101 | WQ27>101) & Reason==1
* Dropping if Water Treatment info is missing
replace Reason=4 if (WQ14==8 | WQ14==9 | WQ14==.) & Reason==1

label define Reasonl 0 "Not selected" 1 "Completed" 2 "Not consented" 3 "Test result missing" 4 "No treat", modify
label values Reason Reasonl

	* Create Dummy
	foreach v in Reason windex5 WS1_g {
	levelsof `v'
	foreach value in `r(levels)' {
		gen     `v'_`value'=0
		replace `v'_`value'=1 if `v'==`value'
		replace `v'_`value'=. if `v'==.
		label var `v'_`value' "`: label (`v') `value''"
	}
	}
label var urban "Urban"

* Total 415,005: 69354 selected
tab HH9 Selected,m
save "${Data}Africa_Latam_Asia_Pooled_hh_data_2_Selection.dta", replace

use "${Data}Africa_Latam_Asia_Pooled_hh_data_2_Selection.dta", clear
* 64420
tab Reason Selected,m
* Collapse at the country level
expand 2, gen(Total)
replace Country="Total" if Total==1
replace Region=99       if Total==1
collapse  Region (sum) Total_HH Selected  Reason_2 Reason_3 Reason_4 Reason_1, by(Country)
gen Share_Selected=round(100*(Selected/Total_HH),0.1) 
gen Share_Complete=round(100*(Reason_1/Selected),0.1) 
order Country Total_HH Selected Share_Selected Reason_2 Reason_3 Reason_4 Reason_1 Share_Complete, first
sort Share_Complete  Reason_1

*  Reason_2 Reason_3 Reason_4
label var Total_HH "Total HH (N)"
label var Selected "Total selected (N)"
label var Share_Selected "Selected share (%)"
label var Reason_1 "Complete(N)"
label var Share_Complete "Complete(%)"
label var Reason_2 "Not consented"
label var Reason_3 "Test missing"
label var Reason_4 "Treat missing"

sort Region Share_Complete
drop Region

texsave $Variables using "${Table}Table_Overall_Atrrition.tex", ///
		footnote("Notes: Households that did not consent to the full survey are excluded from the sample, as their household characteristics are not available for analysis.") ///
		title("Number of Household Surveys and Sample Included in the Analysis") ///
		label("SurNum") ///
		replace varlabels frag location(htbp)
eststo clear

/*-----------------------------
     Figure 2: Water quality at PoC (source) and PoU (household) by the Water Treatment
-----------------------------*/

start_clean
keep NoRiskHome ModerateHighRiskHome VeryHighRiskHome NoRiskSource ModerateHighRiskSource VeryHighRiskSource water_treatment WQ15_g hhweight
gen ID=_n
foreach i in NoRisk ModerateHighRisk VeryHighRisk {
	rename `i'Source `i'1
	rename `i'Home `i'2
	replace `i'1=100 if `i'1==1
	replace `i'2=100 if `i'2==1
}
reshape long NoRisk ModerateHighRisk VeryHighRisk, i(ID water_treatment WQ15_g) j(Loc)
label define Locl 1 "Source" 2 "Drinking", modify
label values Loc Locl
label define WQ15_gl 0 "No treatment" 1 "Boil" 2 `" "Chlorine" "Aquatabs" "PUR" "' 3 `" "Strain" "Settle" "', modify
graph bar  VeryHighRisk ModerateHighRisk  NoRisk if WQ15_g!=98 & WQ15_g!=99, ///
      stack over(Loc, label(angle(45) labsize(normal))) over(WQ15_g) ///
	  blabel(bar, position(inside) format(%9.0f) color(white))  ///
	  legend(order(3 "Low risk" 2 "Moderate" "risk" 1 "High risk")) ///
	  bar(1, color(maroon)) bar(2, color(orange)) bar(3, color(navy))
graph export "${Figure}Desc2.eps", replace 

/*-----------------------------
     Figure 1 (Water treament)
-----------------------------*/
start_clean
collapse WQ15_g_1 WQ15_g_2 WQ15_g_3 WQ15_g_98 WQ15_g_99    WQ15_g_0 [aw=hhweight], by(Country)
expand 2, gen(Total)
replace Country="Total" if Total==1
gen     sortvar=WQ15_g_0
replace sortvar=WQ15_g_0+100 if Total==1
graph bar WQ15_g_1 WQ15_g_2 WQ15_g_3 WQ15_g_98 WQ15_g_99    WQ15_g_0 , ///
      over(Country , label(angle(90)) sort(sortvar)) stack per ///
      blabel(bar, position(center) size(vsmall) format(%9.0f) color(black)) ///
	  ytitle("") ///
	  bar(6, color(black*0.1)) ///
      legend(order(1 "Boil" ///
	               2 "Chlorine" "Aquatabs" "PUR" 3 "Strain" "Settle"  ///
				   4 "Other" 5 "Do not" "know" 6 "Nothing" ) )
graph export "${Figure}Water_Treat.eps", replace  


/*-----------------------------
     Figure 1 (Revised)
-----------------------------*/
start_clean	
collapse VeryHighRiskSource ModerateHighRiskSource NoRiskSource [aw=hhweight], by(Country)
expand 2, gen(Total)
replace Country="Total" if Total==1
gen     sortvar=VeryHighRiskSource
replace sortvar=sortvar+100 if Total==1
graph bar   VeryHighRiskSource ModerateHighRiskSource NoRiskSource, ///
      over(Country , label(angle(90) labsize(small)) sort(sortvar)) stack per ///
      blabel(bar, position(center) size(vsmall) format(%9.0f) color(white)) ///
	  ytitle("") ///
      legend(order(3 "Low risk (<1)" 2 "Moderate risk" "(1 to 100)" 1 "High risk (>100)" ) ) ///
				   bar(1, color(maroon)) bar(2, color(orange)) bar(3, color(navy))
graph export "${Figure}Water_Category1.eps", replace     


start_clean	
foreach i in VeryHighRiskSource ModerateHighRiskSource NoRiskSource {
	gen     T_`i'_0=0
	replace T_`i'_0=1 if `i'==1 & water_treatment==0
	gen     T_`i'_1=0
	replace T_`i'_1=1 if `i'==1 & water_treatment==1
}
collapse T_VeryHighRiskSource_0 T_VeryHighRiskSource_1 T_ModerateHighRiskSource_0 T_ModerateHighRiskSource_1 NoRiskSource VeryHighRiskSource [aw=hhweight], by(Country)
expand 2, gen(Total)
replace Country="Total" if Total==1
gen     sortvar=VeryHighRiskSource
replace sortvar=sortvar+100 if Total==1

graph bar   T_VeryHighRiskSource_0 T_VeryHighRiskSource_1 T_ModerateHighRiskSource_0 T_ModerateHighRiskSource_1 NoRiskSource, ///
      over(Country , label(angle(90) labsize(small)) sort(sortvar)) stack per ///
      blabel(bar, position(center) size(vsmall) format(%9.0f) color(white)) ///
	  ytitle("") ///
      legend(order(5 "Low risk (<1)" 4 "Moderate risk" "(1 to 100)" "Treated" ///
	               3 "Moderate risk" "(1 to 100)" "Not treated"   ///
				   2 "High risk (>100)" "Treated" 1 "High risk (>100)" "Not treated" )) ///
				   bar(1, color(maroon*0.3)) bar(2, color(maroon)) ///
				   bar(3, color(orange*0.3)) bar(4, color(orange))bar(5, color(navy))
graph export "${Figure}Water_Category2.eps", replace     


/*-----------------------------
     Figure 2 (Revised)
-----------------------------*/
start_clean	
collapse VeryHighRiskHome ModerateHighRiskHome NoRiskHome [aw=hhweight], by(Country)
expand 2, gen(Total)
replace Country="Total" if Total==1
gen     sortvar=VeryHighRiskHome
replace sortvar=VeryHighRiskHome+100 if Total==1

graph bar   VeryHighRiskHome ModerateHighRiskHome NoRiskHome , ///
      over(Country , label(angle(90) labsize(small)) sort(sortvar)) stack per ///
      blabel(bar, position(center) size(vsmall) format(%9.0f) color(white)) ///
	  ytitle("") ///
      legend(order(3 "Low risk (<1)" 2 "Moderate risk" "(1 to 100)" 1 "High risk (>100)") ) ///
				   bar(1, color(maroon)) bar(2, color(orange)) bar(3, color(navy))
graph export "${Figure}Water_Category_Home1.eps", replace     


start_clean	
foreach i in VeryHighRiskHome ModerateHighRiskHome NoRiskHome {
	gen     T_`i'_0=0
	replace T_`i'_0=1 if `i'==1 & water_treatment==0
	gen     T_`i'_1=0
	replace T_`i'_1=1 if `i'==1 & water_treatment==1
}

collapse T_VeryHighRiskHome_0 T_VeryHighRiskHome_1 T_ModerateHighRiskHome_0 T_ModerateHighRiskHome_1 T_NoRiskHome_0 T_NoRiskHome_1 VeryHighRiskHome [aw=hhweight], by(Country)
expand 2, gen(Total)
replace Country="Total" if Total==1
gen     sortvar=VeryHighRiskHome
replace sortvar=VeryHighRiskHome+100 if Total==1

graph bar   T_VeryHighRiskHome_0 T_VeryHighRiskHome_1 T_ModerateHighRiskHome_0 T_ModerateHighRiskHome_1 T_NoRiskHome_0 T_NoRiskHome_1, ///
      over(Country , label(angle(90) labsize(small)) sort(sortvar)) stack per ///
      blabel(bar, position(center) size(vsmall) format(%9.0f) color(white)) ///
	  ytitle("") ///
      legend(order(6 "Low risk (<1)" "Treated" 5 "Low risk (<1)" "Not treated"  ///
	               4 "Moderate risk" "(1 to 100)" "Treated" 3 "Moderate risk" "(1 to 100)" "Not treated"   ///
				   2 "High risk (>100)" "Treated" 1 "High risk (>100)" "Not treated" ) ) ///
				   bar(1, color(maroon*0.3)) bar(2, color(maroon)) ///
				   bar(3, color(orange*0.3)) bar(4, color(orange)) ///
				   bar(5, color(navy*0.3))   bar(6, color(navy))
graph export "${Figure}Water_Category_Home2.eps", replace  


/*-----------------------------
     Figure 1 (Original)
-----------------------------*/
start_clean	

gen Category=.

* sHow to consider strain
* replace water_treatment=0 if WQ15==3
replace Category=1 if ModerateHighRiskHome==1 & water_treatment==0
replace Category=2 if ModerateHighRiskHome==1 & water_treatment==1
replace Category=3 if ModerateHighRiskHome==0 & water_treatment==1
replace Category=4 if ModerateHighRiskHome==0 & water_treatment==0

	* Create Dummy
	foreach v in Category {
	levelsof `v'
	foreach value in `r(levels)' {
		gen     `v'_`value'=0
		replace `v'_`value'=1 if `v'==`value'
		replace `v'_`value'=. if `v'==.
		label var `v'_`value' "`: label (`v') `value''"
	}
	}

gen sortvar=Category_1+Category_2

graph bar Category_* [aw=hhweight] ,over(Country , label(angle(90) labsize(small)) sort(sortvar)) stack per ///
      blabel(bar, position(center) size(vsmall) format(%9.0f) color(white)) ///
	  ytitle("") ///
      legend(order(1 "Not treated and" "E.Coli present" 2 "Treated but" "E.Coli present" ///
	               3 "Treated and" "E.Coli free" 4 "Not treated" "and E.Coli free") ) ///
				   bar(1, color(cranberry)) bar(2, color(orange)) bar(3, color(green)) bar(4, color(blue))
graph export "${Figure}Water_Category.eps", replace     

			                                                   /*-----------------------------
																    Transition matrix
																-----------------------------*/


start_clean
tab RiskHome RiskSource,m
* aspectratio(1)
foreach i in 0 1 2 3 98 {
hexplot RiskSource  RiskHome if WQ15_g_`i'==1, values(format(%9.0f) size(large)) legend(off)  ///
                              color(HCL blues, intensity(.6) reverse ) p(lc(black) lalign(center)) bins(5) ///
							  xlabel(0 "Low risk" 1 "Moderate risk" 2 "High risk") xtitle("Drinking water") ///
							  ylabel(0 "Low risk" 1 "Moderate risk" 2 "High risk") ytitle("Water source") ///
							  sizeprop 
graph export "${Figure}TabSourceHome_Treat`i'.eps", replace	
}

start_clean
tab WQ15_g

recode WQ15_g 98 99=4
recode WS10_g 98 99=4
heatplot WQ15_g WS10_g, values(format(%9.0f)) aspectratio(1) legend(off) ///
                              color(HCL blues, intensity(.6) reverse ) p(lc(black) lalign(center)) bins(5) ///
							  xlabel(0 "Nothing" 1 "Boil" 2 "Chlorine" 3 "Strain/Still" 4 "Other") xtitle("General treatment") ///
							  ylabel(0 "Nothing" 1 "Boil" 2 "Chlorine" 3 "Strain/Still" 4 "Other") ytitle("Treatment for tested water") ///
							  sizeprop
graph export "${Figure}Treat_Test_general.eps", replace	


/*-----------------------------
     Table 4: The reduction in E.Coli from water treatment
-----------------------------*/
start_clean	
* keep if Region==1
gen windex5_categ=windex5
recode windex5_categ 1 2=1 3=2 4 5=3
label define windex5_categl 1 "Poor" 2 "Middle" 3 "Rich/Richest", modify
label values windex5_categ windex5_categl
global ControlsVar i.country_cat i.WS1_g i.urban i.windex5_categ
* global ControlsVar i.country_cat i.urban i.windex5_categ

local NoteEColi "Notes: All regressions control for country fixed effects, water source type, urban or rural location, and household socioeconomic status based on an asset-based wealth index. Standard errors, clustered at the primary sampling unit, are in parentheses. Significance level at $\sym{***}p < 0.01, \sym{**}p < 0.05, \sym{*}p < 0.1$."

egen HH_ID=group(HH1 HH2 country_cat)

mdesc NoRiskHome_01_2 country_cat urban WS1_g windex5_categ hhweight RiskSource
foreach i in 0 1 2 {
	eststo: reg NoRiskHome_01_2 i.WQ15_g $ControlsVar if RiskSource==`i' , cluster(Cluster_var)
	sum NoRiskHome_01_2 if RiskSource==`i' & water_treatment==0
	estadd scalar Mean = r(mean)
}

foreach i in 0 1 2 {
	eststo: reg NoRiskHome_0_12 i.WQ15_g $ControlsVar if RiskSource==`i' , cluster(Cluster_var)
	sum NoRiskHome_0_12 if RiskSource==`i' & water_treatment==0
	estadd scalar Mean = r(mean)
}

esttab using "${Table}Est_Main.tex",label se ar2 title("Water Treatment on the Drinking Water E.Coli Contamination" \label{TR0}) nonotes nobase nocons ///
			 mtitle("No risk" "\shortstack[c]{Moderate\\to high risk}" "\shortstack[c]{Very high\\risk}" "No risk" "\shortstack[c]{Moderate\\to high risk}" "\shortstack[c]{Very high\\risk}") ///
			 indicate("Country FE= *country_cat"  "Controls= *WS1* *urban *windex5_categ") ///
			 drop(98.WQ15_g 99.WQ15_g _cons) ///
			 mgroups("\shortstack[c]{Dependent var: Not very high-risk\\(yes = 1, no = 0)}" "\shortstack[c]{Dependent var: Free from E.Coli\\(yes = 1, no = 0)}", pattern(1 0 0 1 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) /// 
			 stats(Mean r2_a N, fmt(%9.2fc %9.2fc %9.0fc) labels(`"Mean"' `"Adjusted \(R^{2}\)"' `"Observations"')) ///
			 starlevels(\sym{*} 0.10 \sym{**} 0.05 \sym{***} 0.010) b(2) ///
			 substitute("{l}{\footnotesize" "{p{1\linewidth}}{\footnotesize" ///
			 "Treat: Boil" "\multicolumn{3}{l}{Relative to no treatment} \\ Treat: Boil" ///
			 "&\multicolumn{1}{c}{No risk}&\multicolumn{1}{c}{\shortstack[c]{Moderate\\to high risk}}&\multicolumn{1}{c}{\shortstack[c]{Very high\\risk}}&\multicolumn{1}{c}{No risk}&\multicolumn{1}{c}{\shortstack[c]{Moderate\\to high risk}}&\multicolumn{1}{c}{\shortstack[c]{Very high\\risk}}\\" " &\multicolumn{1}{c}{No risk}&\multicolumn{1}{c}{\shortstack[c]{Moderate\\to high risk}}&\multicolumn{1}{c}{\shortstack[c]{Very high\\risk}}&\multicolumn{1}{c}{No risk}&\multicolumn{1}{c}{\shortstack[c]{Moderate\\to high risk}}&\multicolumn{1}{c}{\shortstack[c]{Very high\\risk}}\\" ///
			 "Treat:" "" ///
			 ) ///
			 addnote("`NoteEColi'") ///	
			 replace
eststo clear



/*-----------------------------
     Table 1 (Revised)
-----------------------------*/

start_clean

collapse WQ15_g_0 WQ15_g_1 WQ15_g_2 WQ15_g_3 WQ15_g_98 WQ15_g_99 [aw=hhweight], by(Country)
foreach i in WQ15_g_0 WQ15_g_1 WQ15_g_2 WQ15_g_3 WQ15_g_98 WQ15_g_99 {
	replace `i'=round(100*`i',0.1) 
	format `i' %9.2f
	* tostring `i', replace force
}

label var WQ15_g_0 "Nothing"
label var WQ15_g_1 "Boil"
label var WQ15_g_2 "Chlorine/Aquatabs/PUR"
label var WQ15_g_3 "Strain/Settle"
label var WQ15_g_98 "Other"
label var WQ15_g_99 "Do not know"

egen wanted = concat(*),p("&") format("%9.0f")
file open wanted using "wanted.tex", replace write
forv i = 1/`=_N'{
    file write wanted "`=wanted[`i']'\\" _newline(1)
}
file close wanted
				   
texsave using "${Table}Table_Treat_Country.tex", ///
		footnote("Notes: The total for each region is calculated as the simple average of the national averages across countries within that region.") ///
		title("???") ///
		label("Treat") ///
		replace varlabels frag location(htbp)
eststo clear



*  ///
				   bar(1, color(maroon)) ///
				   bar(2, color(orange)) bar(3, color(navy))


/*-----------------------------
     Table X (Appendix)
-----------------------------*/
start_clean
gen sortvar=VeryHighRiskSource
graph bar   VeryHighRiskSource ModerateHighRiskSource NoRiskSource [aw=hhweight], ///
      over(WS1_g , label(angle(90) labsize(small)) sort(sortvar)) stack per ///
      blabel(bar, position(center) size(vsmall) format(%9.0f) color(white)) ///
	  ytitle("") ///
      legend(order(3 "Low risk (<1)" 2 "Moderate risk" "(1 to 100)" 1 "High risk (>100)" ) ) ///
				   bar(1, color(maroon)) bar(2, color(orange)) bar(3, color(navy))
graph export "${Figure}Source_Ecoli.eps", replace     

													
 
start_clean	
gen     Treat_infreq=.
replace Treat_infreq=1 if water_treatment==1
replace Treat_infreq=2 if water_treatment==0 & WS9==1
replace Treat_infreq=3 if water_treatment==0 & WS9==0
tab Treat_infreq,m


graph bar NoRiskSource ModerateHighRiskSource VeryHighRiskSource, stack over(Treat_infreq)
graph bar NoRiskHome ModerateHighRiskHome VeryHighRiskHome, stack over(Treat_infreq)

/*-----------------------------
     Figure 1
-----------------------------*/
start_clean	

gen Category=.

replace Category=1 if NoRiskHome==1 & water_treatment==0 & NoRiskSource==1
replace Category=2 if NoRiskHome==1 & water_treatment==0 & NoRiskSource==0
replace Category=3 if NoRiskHome==1 & water_treatment==1 & NoRiskSource==1
replace Category=4 if NoRiskHome==1 & water_treatment==1 & NoRiskSource==0
replace Category=5 if NoRiskHome==0 & water_treatment==0 & NoRiskSource==1
replace Category=6 if NoRiskHome==0 & water_treatment==0 & NoRiskSource==0
replace Category=7 if NoRiskHome==0 & water_treatment==1 & NoRiskSource==1
replace Category=8 if NoRiskHome==0 & water_treatment==1 & NoRiskSource==0
/*
& water_treatment==0
replace Category=2 if NoRiskHome==1 & water_treatment==1 & NoRiskSource==0
replace Category=3 if NoRiskHome==1 & water_treatment==1 & NoRiskSource==1
replace Category=4 if NoRiskHome==0 & water_treatment==1
replace Category=5 if NoRiskHome==0 & water_treatment==0 & NoRiskSource==0
replace Category=6 if NoRiskHome==0 & water_treatment==0 & NoRiskSource==1
*/

label define Categoryl 1 "Not treated no risk water and no E.Coli" 2 "Not treated risk water and no E.Coli" ///
                       3 "Treated no risk water and no E.Coli" 4 "Treated risk water and no E.Coli" ///
					   5 "Not treated no risk water and E.Coli" 6 "Not treated risk water and E.Coli" ///
                       7 "Treated no risk water and E.Coli" 8 "Treated risk water and E.Coli" ///
					   , modify
label values Category Categoryl

	* Create Dummy
	foreach v in Category {
	levelsof `v'
	foreach value in `r(levels)' {
		gen     `v'_`value'=0
		replace `v'_`value'=1 if `v'==`value'
		replace `v'_`value'=. if `v'==.
		label var `v'_`value' "`: label (`v') `value''"
	}
	}

gen sortvar=Category_1+Category_2

graph bar Category_* [aw=hhweight] ,over(Country , label(angle(90) labsize(vsmall)) sort(sortvar)) stack per ///
      blabel(bar, position(center) size(vsmall) format(%9.0f) color(white)) ///
	  ytitle("") ///
      legend(order(1 "Not treated no risk water" "and no E.Coli" 2 "Not treated risk water" "and no E.Coli" ///
                       3 "Treated no risk water" "and no E.Coli" 4 "Treated risk water" "and no E.Coli" ///
					   5 "Not treated no risk water" "and E.Coli" 6 "Not treated risk water" "and E.Coli" ///
                       7 "Treated no risk water" "and E.Coli" 8 "Treated risk water" "and E.Coli" ///
					   )) 
					   
					   ///
				   bar(1, color(blue)) bar(2, color(orange)) bar(3, color(green)) bar(4, color(blue))
* graph export "${Figure}Water_Category_det.eps", replace     

END




/*-----------------------------
     Table 1: Water Treatment Rate
-----------------------------*/
start_clean

gen     water_treatment_risk=water_treatment
replace water_treatment_risk=. if NoRiskSource==1
gen     water_treatment_vhrisk=water_treatment
replace water_treatment_vhrisk=. if NoRiskSource==1 | ModerateHighRiskSource==1

gen     water_treatment_vhrisk_Ex=water_treatment_vhrisk
replace water_treatment_vhrisk_Ex=0 if water_treatment3==2 
replace water_treatment_vhrisk_Ex=. if NoRiskSource==1 | ModerateHighRiskSource==1

collapse WQ15_g_0 WQ15_g_1 WQ15_g_2 WQ15_g_3 WQ15_g_98 WQ15_g_99 ///
         water_treatment water_treatment_risk NoRiskSource ModerateHighRiskSource VeryHighRiskSource water_treatment_vhrisk water_treatment_vhrisk_Ex [pw=hhweight], by(country_cat Region)

foreach i in WQ15_g_0 WQ15_g_1 WQ15_g_2 WQ15_g_3 WQ15_g_98 WQ15_g_99 water_treatment water_treatment_risk NoRiskSource ModerateHighRiskSource VeryHighRiskSource water_treatment_vhrisk water_treatment_vhrisk_Ex {
	replace `i'=`i'*100
	replace `i'=round(`i', 0.1)
	format `i' %9.2f
}

decode country_cat,gen(country_cat_str)
drop country_cat
sort Region water_treatment
gen  Order=_n
sort Order

global Variables country_cat_str water_treatment NoRiskSource ModerateHighRiskSource VeryHighRiskSource water_treatment_risk water_treatment_vhrisk water_treatment_vhrisk_Ex
global Variables_num water_treatment NoRiskSource ModerateHighRiskSource VeryHighRiskSource water_treatment_risk water_treatment_vhrisk water_treatment_vhrisk_Ex

label var water_treatment "Any water treatment (%)"
label var NoRiskSource "No risk (0 MPN)"
label var ModerateHighRiskSource "Moderate risk (1-100 MPN)"
label var VeryHighRiskSource "Very high risk ($>$100 MPN)"
label var water_treatment_risk "Some E.Coli ($>$1 MPN)"
label var water_treatment_vhrisk "Very high risk of E.Coli ($>$100 MPN)"
label var water_treatment_vhrisk_Ex "Very high risk of E.coli (excl. strain/settle)"

preserve

gen     OBJECTID=.
replace OBJECTID=67 if country_cat_str=="Zimbabwe"
replace OBJECTID=27 if country_cat_str=="Ghana"
replace OBJECTID=11 if country_cat_str=="Central African Republic"
replace OBJECTID=19 if country_cat_str=="DR Congo"
replace OBJECTID=4 if country_cat_str=="Benin"
replace OBJECTID=37 if country_cat_str=="Lesotho"
replace OBJECTID=12 if country_cat_str=="Chad"
replace OBJECTID=53 if country_cat_str=="Sierra Leone"
replace OBJECTID=41 if country_cat_str=="Malawi"
replace OBJECTID=61 if country_cat_str=="Gambia"
replace OBJECTID=62 if country_cat_str=="Togo"
replace OBJECTID=40 if country_cat_str=="Madagascar"
replace OBJECTID=29 if country_cat_str=="Guinea Bissau"
* Eswatini	9
savesome if OBJECTID!=9 using "${Data}/Map/Each.dta", replace

* gen flag=1
* collapse $Variables_num , by(flag)

tempfile total
gen flag=1
collapse $Variables_num , by(flag Region)
foreach i in water_treatment NoRiskSource ModerateHighRiskSource VeryHighRiskSource water_treatment_risk water_treatment_vhrisk water_treatment_vhrisk_Ex {
    replace `i'=round(`i', 0.1)
}

save `total', replace

restore
append using `total'
replace country_cat_str="Africa" if country_cat_str=="" & Region==1
replace country_cat_str="Latin America" if country_cat_str=="" & Region==2
replace country_cat_str="Asia" if country_cat_str=="" & Region==3
replace country_cat_str="CAR" if country_cat_str=="Central African Republic"
replace Order=0.5 if country_cat_str=="Africa"
replace Order=14.5 if country_cat_str=="Latin America"
replace Order=19.5 if country_cat_str=="Asia"
sort Order
texsave $Variables using "${Table}Table_Overall_Country_AF_LAT_AS.tex", ///
		footnote("Notes: The total for each region is calculated as the simple average of the national averages across countries within that region.") ///
		headerlines("& (1) & (2) & (3) & (4) & (5) & (6) & (7) \\ &&\multicolumn{3}{c}{\shortstack[c]{E.coli in source water (\%)}}&\multicolumn{3}{c}{\shortstack[c]{Water treatment\\ Among households with (\%)}} \\ \cmidrule(lr){3-5} \cmidrule(lr){6-8}") ///
		title("Water Treatment Practices Based on Source Water Quality") ///
		label("TreatRisk") ///
		replace varlabels frag location(htbp)
eststo clear


/*-----------------------------
     Table 3+: By source
-----------------------------*/
start_clean
gen     windex_ur=windex5u
replace windex_ur=windex5r if windex_ur==.
mdesc windex_ur
recode windex_ur 1 2=1 3 4 5=2
gen windex5_categ=windex5
recode windex5_categ 1 2=1 3=2 4 5=3
label define windex5_categl 1 "Poor" 2 "Middle" 3 "Rich/Richest", modify
label values windex5_categ windex5_categl
* recode helevel 2 3 4=2
recode helevel 3 4=3
label define helevell 0 "No education" 1 "Primary" 2 "Lower sec" 3 "Upper sec or more", modify
label values helevel helevell

gen Diff_Qual=WQ26-WQ27
tab helevel,m
global ControlsVar i.country_cat i.urban i.WS1 i.windex5_categ

eststo: reg water_treatment i.RiskSource_0_12  $ControlsVar , cluster(Cluster_var)
sum water_treatment
estadd scalar Mean = r(mean)

eststo: reg water_treatment i.RiskSource_0_12  $ControlsVar if WS1==11 | WS1==12 | WS1==13 | WS1==14, cluster(Cluster_var)
sum water_treatment if WS1==11 | WS1==12 | WS1==13 | WS1==14
estadd scalar Mean = r(mean)
eststo: reg water_treatment i.RiskSource_0_12  $ControlsVar if WS1==21 | WS1==22, cluster(Cluster_var)
sum water_treatment if WS1==21 | WS1==22
estadd scalar Mean = r(mean)
eststo: reg water_treatment i.RiskSource_0_12  $ControlsVar if WS1==31 | WS1==32, cluster(Cluster_var)
sum water_treatment if WS1==31 | WS1==32
estadd scalar Mean = r(mean)
eststo: reg water_treatment i.RiskSource_0_12  $ControlsVar if WS1==41 | WS1==42, cluster(Cluster_var)
sum water_treatment  if WS1==41 | WS1==42
estadd scalar Mean = r(mean)
eststo: reg water_treatment i.RiskSource_0_12  $ControlsVar if WS1==81, cluster(Cluster_var)
sum water_treatment if WS1==81
estadd scalar Mean = r(mean)
eststo: reg water_treatment i.RiskSource_0_12  $ControlsVar if WS1==91 , cluster(Cluster_var)
sum water_treatment if WS1==91
estadd scalar Mean = r(mean)

esttab using "${Table}Est_Treat_Risk_AF_LATAM_AS_Source.tex",label se ar2 title("The Likelihood of Applying Water Treatment Based on the Contamination Level of the Source Water" \label{ETRB}) nonotes nobase nocons ///
             indicate("Country FE= *country_cat" "WS FE= *WS1") ///
			 order(*RiskSource_0_12*) ///
			 mtitle("Overall" "\shortstack[c]{Piped\\water}" "\shortstack[c]{Tube\\Well\\Borehole}" "\shortstack[c]{Protected\\well/spring}" "\shortstack[c]{Unprotected\\well/spring}" "\shortstack[c]{Surface\\Rain water}" "\shortstack[c]{Packaged\\bottle}") ///
			 stats(Mean r2_a N, fmt(%9.2fc %9.2fc %9.0fc) labels(`"Mean"' `"Adjusted \(R^{2}\)"' `"Observations"')) ///
			 mgroups("Dependent var: treated water (yes = 1, no = 0)", pattern(1 0 0 0 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) /// 
			 starlevels(\sym{*} 0.10 \sym{**} 0.05 \sym{***} 0.010) b(2) ///
			 substitute("{l}{\footnotesize" "{p{1\linewidth}}{\footnotesize" ///
			 "=1" "" ///
			 "Some E.Coli       &" "\multicolumn{6}{l}{\textbf{A. Water contamination at source (relative to no E.Coli)}} \\ \hline Some E.Coli       &" ///
			 "Urban" "\multicolumn{5}{l}{\textbf{B. Residence (relative to rural)}} \\ \hline Urban" ///
			 "Middle              &" "\multicolumn{5}{l}{\textbf{C. Socioeconomics (relative to poor/very poor)}} \\ \hline Middle&" ///
			 ) ///
			 addnote("Note: Column (1) presents the likelihood of conducting water treatment across the entire sample. Columns (2)-(6) display the effects by sub-samples based on different water sources. Columns (7) interact the presence of E. coli with socioeconomics levels. The reference group for socioeconomic status includes households in the lowest two quintiles (poor and very poor). Standard errors are clustered at the primary sampling unit level (in parentheses). Significance levels are indicated as follows: $\sym{*} p<.10,\sym{**} p<.05,\sym{***} p<.01$. $\sym{*}$") ///	
			 replace
eststo clear





EN



/*-----------------------------
     Appendix
-----------------------------*/
start_clean	

recode helevel 2 3 4=2
* recode helevel 3 4=3
tab helevel,m
label define helevell 0 "No education" 1 "Primary" 2 "Sec or higher", modify
label values helevel helevell

label var water_treatment "Water Treatment"

* Wingdex
gen windex5_categ=windex5
recode windex5_categ 1 2=1 3=2 4 5=3
label define windex5_categl 1 "Poor" 2 "Middle" 3 "Rich/Richest", modify
label values windex5_categ windex5_categl

* start_clean_Diarrhea	
egen HH_ID=group(HH1 HH2 country_cat)

foreach i in 0 1 2 {
	eststo: reg NoRiskHome_01_2 i.water_treatment##i.windex5_categ i.country_cat i.urban i.WS1 i.windex5_categ if RiskSource==`i', cluster(Cluster_var)
	sum NoRiskHome_01_2 if RiskSource==`i' & water_treatment==0
	estadd scalar Mean = r(mean)
}

foreach i in 0 1 2 {
	eststo: reg NoRiskHome_0_12 i.water_treatment##i.windex5_categ i.country_cat i.urban i.WS1 i.windex5_categ if RiskSource==`i' , cluster(Cluster_var)
	sum NoRiskHome_0_12 if RiskSource==`i' & water_treatment==0
	estadd scalar Mean = r(mean)
}

esttab using "${Table}Est_Main_WealthEdu.tex",label se ar2 title("The reduction in E.Coli from water treatment by socioeconomic level" \label{TR1}) nonotes nobase nocons ///
			 mtitle("No risk" "\shortstack[c]{Moderate\\to high risk}" "\shortstack[c]{Very high\\risk}" "No risk" "\shortstack[c]{Moderate\\to high risk}" "\shortstack[c]{Very high\\risk}") ///
			 indicate("Country FE= *country_cat" "WS FE= *WS1") ///
			 drop( _cons) order(1.water_treatment* *windex5_categ*) ///
			 mgroups("Dependent var: Not very high-risk" "Dependent var: Free from E.Coli", pattern(1 0 0 1 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) /// 
			 stats(Mean r2_a N, fmt(%9.2fc %9.2fc %9.0fc) labels(`"Mean"' `"Adjusted \(R^{2}\)"' `"Observations"')) ///
			 starlevels(\sym{*} 0.10 \sym{**} 0.05 \sym{***} 0.010) b(2) ///
			 substitute("{l}{\footnotesize" "{p{1\linewidth}}{\footnotesize" ///
			 "Water Treatment=1   &" "\multicolumn{3}{l}{\textbf{A. Water treatment (relative to no treatment)}} \\ \hline Water Treatment=1   &" ///
			 "Urban" "\multicolumn{5}{l}{\textbf{C. Residence (relative to rural)}} \\ \hline Urban" ///
			 "Tube/Well/Borehole" "\multicolumn{5}{l}{\textbf{B. Water source (relative to piped water)}} \\ \hline Tube/Well/Borehole" ///
			 "Middle              &" "\multicolumn{5}{l}{\textbf{B. Socioeconomics (relative to very poor or poor)}} \\ \hline Middle              &" ///
"&\multicolumn{1}{c}{No risk}&\multicolumn{1}{c}{\shortstack[c]{Moderate\\to high risk}}&\multicolumn{1}{c}{\shortstack[c]{Very high\\risk}}&\multicolumn{1}{c}{No risk}&\multicolumn{1}{c}{\shortstack[c]{Moderate\\to high risk}}&\multicolumn{1}{c}{\shortstack[c]{Very high\\risk}}\\" "By sub-sample &\multicolumn{1}{c}{No risk}&\multicolumn{1}{c}{\shortstack[c]{Moderate\\to high risk}}&\multicolumn{1}{c}{\shortstack[c]{Very high\\risk}}&\multicolumn{1}{c}{No risk}&\multicolumn{1}{c}{\shortstack[c]{Moderate\\to high risk}}&\multicolumn{1}{c}{\shortstack[c]{Very high\\risk}}\\" ///
			 "Treat:" "" ///
			 ) ///
			 addnote("`NoteTakeup'") ///	
			 replace
eststo clear

	
			 
			 END


																

* "&\multicolumn{1}{c}{(1)}" "&\multicolumn{7}{c}{Dependent variable: treatwater (yes = 1, no = 0)} \\&\multicolumn{1}{c}{(1)}" ///

EN







/*-----------------------------
     Table 4: The reduction in E.Coli from water treatment
-----------------------------*/
start_clean	
* keep if Region==1
gen windex5_categ=windex5
recode windex5_categ 1 2=1 3=2 4 5=3
label define windex5_categl 1 "Poor" 2 "Middle" 3 "Rich/Richest", modify
label values windex5_categ windex5_categl
global ControlsVar i.country_cat i.WS1_g i.urban i.windex5_categ
* global ControlsVar i.country_cat i.urban i.windex5_categ

local NoteEColi "Notes: The estimation includes country fixed effects. Standard errors, clustered at the primary sampling unit, are in parentheses. Significance level at $\sym{***}p < 0.01, \sym{**}p < 0.05, \sym{*}p < 0.1$."

egen HH_ID=group(HH1 HH2 country_cat)

mdesc NoRiskHome_01_2 country_cat urban WS1_g windex5_categ hhweight RiskSource
foreach i in 0 1 2 {
	eststo: reg NoRiskHome_01_2 i.WQ15_g $ControlsVar if RiskSource==`i' [aw=hhweight], cluster(Cluster_var)
	sum NoRiskHome_01_2 if RiskSource==`i' & water_treatment==0
	estadd scalar Mean = r(mean)
}

foreach i in 0 1 2 {
	eststo: reg NoRiskHome_0_12 i.WQ15_g $ControlsVar if RiskSource==`i' [aw=hhweight], cluster(Cluster_var)
	sum NoRiskHome_0_12 if RiskSource==`i' & water_treatment==0
	estadd scalar Mean = r(mean)
}

esttab using "${Table}Est_Main_weights.tex",label se ar2 title("The E.Coli Contamination in Drinking Water by the Source Contamination" \label{TR0}) nonotes nobase nocons ///
			 mtitle("No risk" "\shortstack[c]{Moderate\\to high risk}" "\shortstack[c]{Very high\\risk}" "No risk" "\shortstack[c]{Moderate\\to high risk}" "\shortstack[c]{Very high\\risk}") ///
			 indicate("Country FE= *country_cat" ) ///
			 drop(98.WQ15_g 99.WQ15_g _cons) ///
			 mgroups("Dependent var: Not very high-risk" "Dependent var: Free from E.Coli", pattern(1 0 0 1 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) /// 
			 stats(Mean r2_a N, fmt(%9.2fc %9.2fc %9.0fc) labels(`"Mean"' `"Adjusted \(R^{2}\)"' `"Observations"')) ///
			 starlevels(\sym{*} 0.10 \sym{**} 0.05 \sym{***} 0.010) b(2) ///
			 substitute("{l}{\footnotesize" "{p{1\linewidth}}{\footnotesize" ///
			 "Treat: Boil" "\multicolumn{3}{l}{\textbf{A. Water treatment (relative to no treatment)}} \\ \hline Treat: Boil" ///
			 "Urban" "\multicolumn{5}{l}{\textbf{C. Residence (relative to rural)}} \\ \hline Urban" ///
			 "Tube/Well/Borehole" "\multicolumn{5}{l}{\textbf{B. Water source (relative to piped water)}} \\ \hline Tube/Well/Borehole" ///
			 "Middle" "\multicolumn{5}{l}{\textbf{D. Socioeconomics (relative to very poor or poor)}} \\ \hline Middle" ///
			 "&\multicolumn{1}{c}{No risk}&\multicolumn{1}{c}{\shortstack[c]{Moderate\\to high risk}}&\multicolumn{1}{c}{\shortstack[c]{Very high\\risk}}&\multicolumn{1}{c}{No risk}&\multicolumn{1}{c}{\shortstack[c]{Moderate\\to high risk}}&\multicolumn{1}{c}{\shortstack[c]{Very high\\risk}}\\" "By sub-sample &\multicolumn{1}{c}{No risk}&\multicolumn{1}{c}{\shortstack[c]{Moderate\\to high risk}}&\multicolumn{1}{c}{\shortstack[c]{Very high\\risk}}&\multicolumn{1}{c}{No risk}&\multicolumn{1}{c}{\shortstack[c]{Moderate\\to high risk}}&\multicolumn{1}{c}{\shortstack[c]{Very high\\risk}}\\" ///
			 "Treat:" "" ///
			 ) ///
			 addnote("`NoteEColi'") ///	
			 replace
eststo clear



END



start_clean			
global Main urban Any_U5 windex5_1 windex5_2 windex5_3 windex5_4 windex5_5 ///
			WS1_g_11 WS1_g_21 WS1_g_31 WS1_g_32 WS1_g_51 WS1_g_91 WS1_g_96 ///
			water_treatment  ///
			WQ15_g_0 WQ15_g_1 WQ15_g_2 WQ15_g_3 WQ15_g_98 ///
			WQ27 WQ26
			*  WQ15_g_99  WQ29  rainy_season Any_U5

local Main "Descriptive statistics by countries"
local LabelMain "Outside"
	
foreach k in Main {
	levelsof country_cat
	foreach i in `r(levels)' {
	eststo  model0: estpost summarize $`k' [aw=hhweight]
		eststo  model`i': estpost summarize $`k' [aw=hhweight] if country_cat==`i'
	}
esttab model0 model22 model31 using "${Table}Descript_`k'_C1.tex", title("``k''" \label{`Label`k''}) ///
	   cell("mean (fmt(2) label(_))") stats(N, fmt("%9.0fc") label(Observations) ) /// 
	   mtitles("Mean" "Mon" "Viet") nonum ///
	   substitute( ".00" "" "{l}{\footnotesize" "{p{0.87\linewidth}}{\footnotesize" ///
				   "Piped water (Dwelling)" "\textbf{Primary Water Source} \\\hline Piped water (Dwelling)" ///
				   "Location: In own dwelling" "\textbf{Location} \\\hline Location: In own dwelling" ///
				   "Any water treatment for tested" "\textbf{Water that is tested} \\\hline Any water treatment" ///
				   "Any water treatment for primary" "\textbf{Primary Water Source} \\\hline Any water treatment" ///
				   "-0 " "0" ///
				   "Expenditure sch: " "~~~" "Treat:"  "~~~" "Location:"  "~~~" ///
				   ) ///
	   label  note("`note`k''")  ///
replace 

/*
esttab model0 model11 model13 model14 model15  model16  model17  model18 using "${Table}Descript_`k'_C2.tex", title("``k''" \label{`Label`k''}) ///
	   cell("mean (fmt(2) label(_))") stats(N, fmt("%9.0fc") label(Observations) ) /// 
	   mtitles("Mean" "Fiji" "Ghana" "Guinea" "Guya" "Hon" "Jama" "Kirib") nonum ///
	   substitute( ".00" "" "{l}{\footnotesize" "{p{0.87\linewidth}}{\footnotesize" ///
				   "Piped water (Dwelling)" "\textbf{Primary Water Source} \\\hline Piped water (Dwelling)" ///
				   "Location: In own dwelling" "\textbf{Location} \\\hline Location: In own dwelling" ///
				   "Any water treatment for tested" "\textbf{Water that is tested} \\\hline Any water treatment" ///
				   "Any water treatment for primary" "\textbf{Primary Water Source} \\\hline Any water treatment" ///
				   "-0 " "0" ///
				   "Treat:"  "~~~" "Location:"  "~~~" ///
				   ) ///
	   label  note("`note`k''")  ///
replace 

esttab model0 model19 model20 model21 model22  model23  model24  model25 using "${Table}Descript_`k'_C3.tex", title("``k''" \label{`Label`k''}) ///
	   cell("mean (fmt(2) label(_))") stats(N, fmt("%9.0fc") label(Observations) ) /// 
	   mtitles("Mean" "Lao" "?" "Mada" "Mala" "Mong" "Nepal") nonum ///
	   substitute( ".00" "" "{l}{\footnotesize" "{p{0.87\linewidth}}{\footnotesize" ///
				   "Piped water (Dwelling)" "\textbf{Primary Water Source} \\\hline Piped water (Dwelling)" ///
				   "Location: In own dwelling" "\textbf{Location} \\\hline Location: In own dwelling" ///
				   "Any water treatment for tested" "\textbf{Water that is tested} \\\hline Any water treatment" ///
				   "Any water treatment for primary" "\textbf{Primary Water Source} \\\hline Any water treatment" ///
				   "-0 " "0" ///
				   "Treat:"  "~~~" "Location:"  "~~~" ///
				   ) ///
	   label  note("`note`k''")  ///
replace 

esttab model0 model26 model27 model28 model29  model30  model31  model32 using "${Table}Descript_`k'_C4.tex", title("``k''" \label{`Label`k''}) ///
	   cell("mean (fmt(2) label(_))") stats(N, fmt("%9.0fc") label(Observations) ) /// 
	   mtitles("Mean" "SUR" "Togo" "Tonga" "Trini" "Turk" "Tuva" "Viet" "Zimb") nonum ///
	   substitute( ".00" "" "{l}{\footnotesize" "{p{0.87\linewidth}}{\footnotesize" ///
				   "Piped water (Dwelling)" "\textbf{Primary Water Source} \\\hline Piped water (Dwelling)" ///
				   "Location: In own dwelling" "\textbf{Location} \\\hline Location: In own dwelling" ///
				   "Any water treatment for tested" "\textbf{Water that is tested} \\\hline Any water treatment" ///
				   "Any water treatment for primary" "\textbf{Primary Water Source} \\\hline Any water treatment" ///
				   "-0 " "0" ///
				   "Treat:"  "~~~" "Location:"  "~~~" ///
				   ) ///
	   label  note("`note`k''")  ///
replace 
*/

	   }


	   


/*-----------------------------
     Figure 1: Prevalence of Diarrhea among U5 children by Water Treatment
-----------------------------*/
start_clean_Diarrhea
gen flag=1
replace diarrhea=100 if diarrhea==1
bys water_treatment3 country_cat Country: egen Flag=sum(flag)
drop if Flag<15
tab Flag,m

collapse diarrhea (sum) flag [pw=hhweight] , by(water_treatment3 country_cat Country) 

label var diarrhea "Diarrhea prevalence (%) among children under age 5"
sort water_treatment3 diarrhea
gen ID=_n
replace ID=. if water_treatment3==1
replace ID=. if water_treatment3==2
sort  country_cat ID
bys country_cat: replace ID=ID[_n-1] if ID==.
bys country_cat: replace ID=ID[_n-2] if ID==.
twoway (scatter diarrhea ID if water_treatment3==0, mlab(Country) mlabs(vsmall) mstyle(X) mlabp(12)) ///
       (scatter diarrhea ID if water_treatment3==1, m(D)) ///
	   (scatter diarrhea ID if water_treatment3==2, m(X)) , ///
	   xtitle("") xlabel(none) ///
       legend(order(1 "Diarrhea among" "households without" "any water treatment" " " 2 "Diarrhea among" "households with" "water treatment" "(boiling, chlorine," "other)" " " 3 "Diarrhea among" "households with" "water treatment" "(strain and settle)") ring(0) position(10) )
		graph export "${Figure}Water_Diarrhea_Treat.eps", replace     




END







/*-----------------------------
     Table 0: Desciptive statistics by the level of source water contamination
-----------------------------*/
start_clean
tab water_carrier_edu,m
* WS9 WS10_0 WS10_1 WS10_2 WS10_3 WS10_6 WS10_7 WS10_98 WS10_99
global Main urban Any_U5 windex5_1 windex5_2 windex5_3 windex5_4 windex5_5 ///
			WS1_g_11 WS1_g_21 WS1_g_31 WS1_g_32 WS1_g_51 WS1_g_91 WS1_g_96 ///
			water_treatment  ///
			WQ15_g_0 WQ15_g_1 WQ15_g_2 WQ15_g_3 WQ15_g_98 ///
			WQ27 WQ26
			*  WQ15_g_99  WQ29  rainy_season Any_U5
			
* Basic_water_service Limited_water_service Surface_water_service Unimproved_water_service  ///
* water_straight_from_source water_stored_covered water_stored_uncovered

local Main "Household characteristics by the level of source water contamination"
local LabelMain "Desc1"
local noteMain "Notes: The table presents the household characteristics across 24 countries. The final sample size is 61,095 households from 24 countries, with 26,029 (42.6 percent) in the no-risk category, 22,553 (36.9 percent) in the moderate-risk category, and 12,513 (20.5 percent) in the very high-risk category at the water source. The number of CFUs per 100mL of E. Coli is capped at 101 if the number is higher than 100. The mean of the blank water test is 0.66 CFUs per 100mL."
					 
foreach k in Main {
* Mean
	eststo  model0: estpost summarize $`k' [aw=hhweight] if RiskSource==0
	eststo  model1: estpost summarize $`k' [aw=hhweight] if RiskSource==1
	eststo  model2: estpost summarize $`k' [aw=hhweight] if RiskSource==2

esttab model0 model1 model2 using "${Table}Descript_`k'_Risk.tex", title("``k''" \label{`Label`k''}) ///
	   cell("mean (fmt(2) label(_))") stats(N, fmt("%9.0fc") label(Observations) ) /// 
	   mtitles("No risk" "Moderate-high risk" "Very high risk") nonum ///
	   substitute( ".00" "" "{l}{\footnotesize" "{p{0.85\linewidth}}{\footnotesize" ///
				   "                    &           _&           _&           _\\" "" ///
				   "Piped water" "\textbf{Primary Water Source} \\\hline Piped water" ///
				   "Location: In own dwelling" "\textbf{Location} \\\hline Location: In own dwelling" ///
                   "Source water test (100ml)" "\textbf{Water test results} \\\hline Source water test (100ml)" ///
				   "Poorest" "\textbf{Socioeconomic level} \\\hline Poorest" ///
				   "Basic water service" "\textbf{Water source category} \\\hline Basic water service" ///
				   "Any water treatment for primary" "\textbf{Primary Water Source} \\\hline Any water treatment" ///
				   "-0 " "0" ///
				   "Treat:"  "~~~" "Location:"  "~~~" ///
				   ) ///
	   label  note("`note`k''")  ///
	   replace 
	   }
eststo clear




* shp2dta using "${Data}/Map/IPUMSI_world_release2024.shp", database(${Data}Map/phdb) coordinates(${Data}Map/phxy) genid(id) genc(c) replace
use "${Data}Map/phdb.dta",clear
drop if x_c<-55.90
drop if x_c>51.81
foreach i in 13 51 22 232 269 279 8 59 60 32 15 7 35 272 141 127 147 243 181 113 163 {
	drop if OBJECTID==`i'	
}
drop if y_c>31.94
drop if id==284
merge 1:1 OBJECTID using "${Data}/Map/Each.dta"

savesome x_c y_c CNTRY_NAME if _merge==3 using "${Data}/Map/Each_label.dta" ,replace

replace water_treatment_vhrisk_Ex=100-water_treatment_vhrisk_Ex
spmap water_treatment_vhrisk_Ex using "${Data}Map/phxy.dta" ,id(id) ndfcolor(gray) fcolor(Reds) ///
       label(data("${Data}/Map/Each_label.dta") xcoord(x_c)               ///
       ycoord(y_c) label(CNTRY_NAME) color(black) size(*0.5))
graph export "${Figure}Map_water_treatment_vhrisk_EX.eps", replace     

END


/*-----------------------------
     Table 3: The likelihood of applying water treatment by the intial contamination level of source water and education level (robustness)
-----------------------------*/
start_clean
gen     windex_ur=windex5u
replace windex_ur=windex5r if windex_ur==.
mdesc windex_ur
* recode helevel 2 3 4=2
recode helevel 3 4=3
recode windex_ur 1 2=1 3 4 5=2
gen windex5_categ=windex5
recode windex5_categ 1 2=1 3=2 4 5=3
label define windex5_categl 1 "Poor" 2 "Middle" 3 "Rich/Richest", modify
label values windex5_categ windex5_categl
label define helevell 0 "No education" 1 "Primary" 2 "Lower sec" 3 "Upper sec or more", modify
label values helevel helevell

global ControlsVar i.country_cat i.urban i.WS1 i.rainy_season

eststo: reg water_treatment i.VeryHighRiskSource  i.windex5_categ $ControlsVar [aw=hhweight], cluster(Cluster_var)
sum water_treatment
estadd scalar Mean = r(mean)
eststo: reg water_treatment i.VeryHighRiskSource##i.windex5_categ $ControlsVar [aw=hhweight], cluster(Cluster_var)
sum water_treatment
estadd scalar Mean = r(mean)
eststo: reg water_treatment i.VeryHighRiskSource##i.windex5_categ $ControlsVar if WS1==11 | WS1==12 | WS1==13 | WS1==14 [aw=hhweight], cluster(Cluster_var)
sum water_treatment if WS1==11 | WS1==12 | WS1==13 | WS1==14
estadd scalar Mean = r(mean)
eststo: reg water_treatment i.VeryHighRiskSource##i.windex5_categ $ControlsVar if WS1==21 | WS1==22 [aw=hhweight], cluster(Cluster_var)
sum water_treatment if WS1==21 | WS1==22
estadd scalar Mean = r(mean)
eststo: reg water_treatment i.VeryHighRiskSource##i.windex5_categ $ControlsVar if WS1==31 | WS1==32 [aw=hhweight], cluster(Cluster_var)
sum water_treatment if WS1==31 | WS1==32
estadd scalar Mean = r(mean)
eststo: reg water_treatment i.VeryHighRiskSource##i.windex5_categ $ControlsVar if WS1==41 | WS1==42 [aw=hhweight], cluster(Cluster_var)
sum water_treatment  if WS1==41 | WS1==42
estadd scalar Mean = r(mean)
eststo: reg water_treatment i.VeryHighRiskSource##i.windex5_categ $ControlsVar if WS1==81 [aw=hhweight], cluster(Cluster_var)
sum water_treatment if WS1==81
estadd scalar Mean = r(mean)
esttab using "${Table}Est_Treat_Risk_High.tex",label se ar2 title("The likelihood of applying water treatment by the intial contamination level of source water and education level (appendix)" \label{ETR1}) nonotes nobase nocons ///
             indicate("Country FE= *country_cat" "WS FE= *WS1") ///
			 mtitle("Overall" "Overall" "Piped" "Borehall" "Well" "Spring" "Surface") ///
			 stats(Mean r2_a N, fmt(%9.2fc %9.2fc %9.0fc) labels(`"Mean"' `"Adjusted \(R^{2}\)"' `"Observations"')) ///
			 starlevels(\sym{*} 0.10 \sym{**} 0.05 \sym{***} 0.010) b(2) ///
			 substitute("{l}{\footnotesize" "{p{1\linewidth}}{\footnotesize" ///
			 "=1" "" ///
			 ) ///
			 addnote("Note: The base of the socio-economic level is the two lowest quintile poor and very poor. Standard errors clustered at the primary sampling unit in parentheses, $\sym{*} p<.10,\sym{**} p<.05,\sym{***} p<.01$") ///	
			 replace
eststo clear


		
		END




/*-----------------------------
     Figure 1: Relationship between E-coli Count of PoC (Source) and PoU (Household) Water among Households with no Water Treatment
-----------------------------*/
start_clean
twoway (scatter WQ26 WQ27 if WQ15_g==0) (lowess WQ26 WQ27 if WQ15_g==0) (function y=x , range(0 100)) , ///
        xtitle("PoC: Source (100 ml)") ytitle("PoU: Household (100 ml)") ///
		legend(off)
graph export "${Figure}Desc1.eps", replace     

   

/*-----------------------------
     Figure 3: Relationship between E-coli of Source/Drinking Water and Diarrhoea and Fever
-----------------------------*/ 
start_clean_Diarrhea
sum diarrhea fever

twoway (hist WQ27, yaxis(2) color(green) ) (hist WQ26,yaxis(2) ytitle("% of HH with diarrhea or fever") fcolor(none) lcolor(black))  ///
       (lowess diarrhea WQ26)  (lowess diarrhea WQ27) ///
       (lowess fever    WQ26)  (lowess fever    WQ27) ///
	   , ///
		legend(order(1 "Histogram E-coli in source water" 2 "Histogram E-coli in drinking water"  ///
		             3 "(1) Diarrhea (Drinking water)" 4 "(2) Diarrhea (Source water)" 5 "(3) Fever (Drinking water)" 6 "(4) Fever (Source water)" )) ///
					 xtitle("E-coli count (Source or Drinking Water)")
		graph export "${Figure}WaterQ_Morbidity.eps", replace     


END























END





/*-----------------------------
     Table 3: Direct and Indirect Effect of Water Treatment (Mediation analayis)
-----------------------------*/
set more off
* use "${data}Water_Final.dta", clear
start_clean_Diarrhea	
egen Cluster_HH=group(country_cat HH1 HH2)
replace diarrhea=diarrhea/100

* local ControlsVar "i.country_cat i.urban i.WS1 i.Drinking_water_ladder i.Sanitation_ladder i.Hygeiene_ladder i.helevel"
global ControlsVar i.country_cat i.urban i.helevel i.WS1

gen      Wealth_categ_3=windex5
recode   Wealth_categ_3 1=1 2=2 3 4 5=3


label define WS9l 0 "No treatment" 1 "Water treatment", modify
label values WS9 WS9l

/*
"Poor" "Middle" "Rich"
foreach r in 1 2 3 {
    * Mediation analysis for child diarrhea 
    eststo: mediate (diarrhea $ControlsVar) (WQ26 $ControlsVar) (water_treatment) [pw=hhweight] if Wealth_categ_3 == `r', vce(cluster Cluster_HH)
    sum diarrhea if Wealth_categ_3 == `r'
    estadd scalar Mean = r(mean)
}
*/

*  VeryHighRiskHome WQ26
foreach o in RiskHome_0_12 {
/* 
eststo: mediate (diarrhea $ControlsVar) (`o' $ControlsVar) (WS9) [pw=hhweight] , vce(cluster Cluster_HH)
    sum diarrhea
    estadd scalar Mean = r(mean)
*/

eststo: mediate (diarrhea $ControlsVar) (`o' $ControlsVar) (water_treatment) [pw=hhweight] , vce(cluster Cluster_HH)
    sum diarrhea
    estadd scalar Mean = r(mean)
/* 
foreach r in 0 1 2 {
    * Mediation analysis for child diarrhea
    eststo: mediate (diarrhea $ControlsVar) (`o' $ControlsVar) (WS9) [pw=hhweight] if RiskSource== `r', vce(cluster Cluster_HH)
    sum diarrhea if RiskSource == `r'
    estadd scalar Mean = r(mean)
}
*/

foreach r in 0 1 2 {
    * Mediation analysis for child diarrhea
    eststo: mediate (diarrhea $ControlsVar) (`o' $ControlsVar) (water_treatment) [pw=hhweight] if RiskSource== `r', vce(cluster Cluster_HH)
    sum diarrhea if RiskSource == `r'
    estadd scalar Mean = r(mean)
}

esttab using "${Table}Result_Medi_`o'.csv", se ar2 nonotes b(3)  drop(_cons) replace
esttab using "${Table}Result_Medi_`o'.tex", se ar2 nonotes label b(3) drop(_cons) ///
			  nobase indicate("Control=*urban *helevel *WS1 *country_cat") ///
			  mtitle("Across" "\shortstack{No risk}" "\shortstack{Moderate to\\high risk}" "\shortstack{Very\\high risk}" ) ///
			  title("Direct and Indirect Effect of Water Treatment (`o')"\label{Medi}) ///
			  substitute("{l}{\footnotesize" "{p{0.99\linewidth}}{\footnotesize" ///
						 "main" "" "NDE" "\textbf{Direct Effect}" "TE" "\textbf{Total Effect}" "NIE" "\textbf{Indirect Effect}"  ///
						 ) ///
			  stats(Mean N, fmt(%9.2fc %9.0fc) labels(`"Outcome Mean"' `"Observations"' )) ///
			  note("Notes: Standard errors in parentheses, $\sym{*} p<.10,\sym{**} p<.05,\sym{***} p<.01$" "`ControlsVar_exp'") ///
              starlevels(\sym{*} 0.10 \sym{**} 0.05 \sym{***} 0.010) replace
eststo clear
}

END

start_clean
replace VeryHighRiskHome=VeryHighRiskHome*100
graph bar VeryHighRiskHome if WS1==21 | WS1==22 | WS1==31 | WS1==32, over(WQ15, sort(VeryHighRiskHome) label(angle(90) labsize(small))) ylab(0 (10) 60)  
* graph bar VeryHighRiskHome if WS1==21 | WS1==22 | WS1==31 | WS1==32, over(helevel) over(WQ15, sort(VeryHighRiskHome) label(angle(90) labsize(small))) ylab(0 (10) 60)  

graph bar VeryHighRiskHome if WS1==11 | WS1==12 | WS1==13 | WS1==14, over(WQ15, sort(VeryHighRiskHome) label(angle(90) labsize(small))) ylab(0 (5) 30) 
graph bar VeryHighRiskHome if WS1==21 | WS1==22 | WS1==31 | WS1==32, over(WQ15, sort(VeryHighRiskHome) label(angle(90) labsize(small))) ylab(0 (10) 60) 
 graph bar VeryHighRiskHome, over(WQ15, sort(VeryHighRiskHome) label(angle(90) labsize(small))) ylab(0 (20) 60)
 
 graph bar WQ26, over(WQ15, sort(WQ26) label(angle(90) labsize(small))) ylab(0 (20) 60)
 graph bar WQ27, over(WQ15, sort(WQ27) label(angle(90) labsize(small))) ylab(0 (20) 60)



	   
/*-----------------------------
         Main Figure
-----------------------------*/


END


start_clean			
replace ImprovedWaterSource=ImprovedWaterSource*100
collapse ImprovedWaterSource VeryHighRiskHome [pw=wqsweight], by(country_cat windex5)
twoway (connected ImprovedWaterSource windex5 if country_cat==1) (connected ImprovedWaterSource  windex5 if country_cat==2) ///
       (connected ImprovedWaterSource  windex5 if country_cat==3) (connected ImprovedWaterSource  windex5 if country_cat==4) ///
	   (connected ImprovedWaterSource  windex5 if country_cat==5) (connected ImprovedWaterSource  windex5 if country_cat==6) ///
       (connected ImprovedWaterSource  windex5 if country_cat==7) (connected ImprovedWaterSource  windex5 if country_cat==8) ///
	   , ytitle("ImprovedWaterSource") xtitle("Wealth quintile") xlabel(1(1)5) ///
	   legend(order(1 "Benin" 2 "CAR" 3 "Chad" 4 "DRC") pos(6) row(2)) 






global Main WQ26 WQ27 WQ29 urban ///
			Basic_water_service Limited_water_service Surface_water_service Unimproved_water_service ///
            WS1_11 WS1_12 WS1_13 WS1_14 WS1_21 WS1_31 WS1_32 WS1_41 WS1_42 WS1_51 WS1_61 WS1_71 WS1_81 WS1_91 WS1_96 ///
			WS9 WS10_0 WS10_1 WS10_2 WS10_3 WS10_6 WS10_7 WS10_98 WS10_99 ///
			water_treatment WQ15_0 WQ15_1 WQ15_2 WQ15_3 WQ15_6 WQ15_7 WQ15_98 WQ15_99 ///
			WS3_1 WS3_2 WS3_3 WS3_998 ///
			rainy_season

	   
start_clean
			
local Main "Desciptive statistics of the variables used in LASSO"
local LabelMain "Desc0"
					 
foreach k in Main {
* Mean
	eststo  model0: estpost summarize $`k'
* Median
	foreach i in $`k' {
	egen m_`i'=median(`i')
	replace `i'=m_`i'
	}
	eststo  model1: estpost summarize $`k'

* Min
	start_clean
	foreach i in $`k' {
	egen i_`i'=min(`i')
	replace `i'=i_`i'
	}

	eststo  model6: estpost summarize $`k'
* Max
	start_clean
	foreach i in $`k' {
	egen a_`i'=max(`i')
	replace `i'=a_`i'
	}
	eststo  model7: estpost summarize $`k'
* Missing 
	start_clean
	foreach i in $`k' {
	egen `i'_s=rowmiss(`i')
	egen s_`i'=sum(`i'_s)
	replace `i'=s_`i'
	}
	eststo  model8: estpost summarize $`k'

esttab model0 model1 model6 model7 model8 using "${Table}Descript_`k'.tex", title("``k''" \label{`Label`k''}) ///
	   cell("mean (fmt(2) label(_))") stats(N, fmt("%9.0fc") label(Observations) ) /// 
	   mtitles("Mean" "Median" "Min" "Max" "Number missing") nonum ///
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

 END
 
 use "${Final}temp_LASSO.dta", clear
 graph bar yhat_cv if WS1==81, over(WS10, sort(yhat_cv) label(angle(45) labsize(small)))
 
 
 cibar yhat_cv if WS1==21, over1(WS10) graphopts(ylab(0 (20) 80))
 graph bar yhat_cv if WS1==21, over(WS10, sort(yhat_cv) label(angle(45) labsize(small)))
 
 graph bar yhat_cv if WS1==21 |  WS1==32 |  WS1==14  |  WS1==81, over(WS10, label(angle(45) labsize(small))) by(WS1)
 collapse yhat_cv, by(WS1 WS10)
 * label var yhat_cv     "Computed probablity of wearing glasses"
 twoway (connected yhat_cv country_cat) , ///
	    legend(order(1 "First time visited by VerBien" 2 "Visited by VerBien" )) xline(0.2)
		graph export "${Figure}LASSO_WS1_WS10.eps", replace     

		
		END
		
 graph bar yhat_cv, over(WS10, label(angle(90) labsize(small)))
 graph bar yhat_cv, over(country_cat, label(angle(90) labsize(small)))
 use "${Final}temp_LASSO.dta", clear
 keep if WS1==14 | WS1==21  | WS1==31 | WS1==32 | WS1==41 | WS1==42 | WS1==81
 graph bar yhat_cv, over(WS1, sort(yhat_cv) label(angle(90) labsize(small))) ylab(0 (20) 80)
 graph bar WQ27, over(WS1, sort(yhat_cv) label(angle(90) labsize(small))) ylab(0 (20) 80)
 graph export "${Figure}LASSO_WS1.eps", replace     
 cibar WQ26, over1(WS1) graphopts(ylab(0 (20) 80))
 graph export "${Figure}Mean_WS1.eps", replace     
 graph bar yhat_cv, over(WS1, label(angle(90) labsize(small)))
 cibar yhat_cv, over(WS1, label(angle(90) labsize(small)))
 
 graph bar yhat_cv, over(WS3)
 
 
 
 
 /*----------------------------------------------------
  Water (at the point of consumption)
  ----------------------------------------------------*/
* Normal Lasso
 start_clean
* Variable construction
    global V_country   i.country_cat
	global V_urban     i.urban
	global V_source    i.WS1
	global V_location  i.WS3
	global V_treat     i.WS10
	global V_season    i.rainy_season

global V_simple   $V_country $V_urban $V_source $V_location $V_treat $V_season
global V_interact $V_country##$V_urban##$V_source##$V_location##$V_treat##$V_season

* Model creation
	splitsample, generate(sample) nsplit(3) rseed(1234)
 
 * foreach i in WQ26 VeryHighRiskHome {
 foreach i in WQ26 {
 lasso linear `i' $V_simple $V_interact if sample==1, selection(cv) rseed(1234)
 estimates store cv
 predict yhat_cv, xb
 * Estimate results 
 lassocoef cv, sort(coef, standardized) nofvlabel
 * lassocoef cv adaptive, sort(coef, standardized) nofvlabel
 esttab cv using "${Table}LASSO_`i'.csv", replace ///
       stats(k_allvars k_nonzero_sel lambda_sel N, fmt(%9.0fc %9.0fc %9.4fc %9.0fc) labels(`"\# of potential variables"' `"\# of selected variables"' `"Lambda"' `"Observations"')) ///
       label mtitles("CV" "Adaptive" "BIC") nonum ///
	   substitute("{l}{\footnotesize" "{p{0.9\linewidth}}{\scriptsize") ///
	   title("LASSO: Selection of variables" \label{`Label`k''}) nonotes
 
 esttab cv using "${Table}LASSO_`i'.tex", replace ///
       stats(k_allvars k_nonzero_sel lambda_sel N, fmt(%9.0fc %9.0fc %9.4fc %9.0fc) labels(`"\# of potential variables"' `"\# of selected variables"' `"Lambda"' `"Observations"')) ///
       label mtitles("CV" "Adaptive" "BIC") nonum ///
	   substitute("{l}{\footnotesize" "{p{0.9\linewidth}}{\scriptsize") ///
	   keep(_cons) ///
	   title("``i''" \label{`Label`k''}) nonotes
	   
 lassogof  cv, over(sample) postselection
 outtable  using "${Table}LASSO_`i'_R", mat(r(table)) replace
 save "${Final}temp_LASSO_`i'.dta", replace
 }
 END

 
 
 
 
****************
* Wate Quality *
****************
use "${data}Water_Final.dta", clear
global ListWQ CA1 WQ1 WQ4 WQ7_11 WQ7_12 WQ7_13 WQ7_14 WQ7_21 WQ7_31 WQ7_32 WQ7_41 WQ7_42 WQ7_51 WQ7_71 WQ7_81 WQ7_96 WQ7_99 ///
              WQ16_num WQ17 WQ9 WQ9_categ_0 WQ9_categ_10 WQ9_categ_25 WQ9_categ_50 WQ9_categ_200 WQ18_num WQ19D WQ20D ///
			  WQ19D_Bi WQ20D_Bi ///
			  WQ19D_Level_0 WQ19D_Level_1 WQ20D_Level_0 WQ20D_Level_1
			  
foreach i in $ListWQ {
replace `i'=0 if `i' ==. & Country==3 
}

* graph pie, over(WQ7)

local  ListWQ   "Descriptive of Water Quality"
local  LabelListProviderBias  "WHOPB"
local  noteListWQ "Notes: "
foreach k in ListWQ {
eststo summstats0: estpost summarize $`k' 
eststo summstats1: estpost summarize $`k' if Country==1
eststo summstats2: estpost summarize $`k' if Country==2 
eststo summstats3: estpost summarize $`k' if Country==3 
	* Min
	foreach i in $`k' {
	egen min_`i'=min(`i')
	replace `i'=min_`i'
	}

	eststo summstats4: estpost summarize $`k'
* Max
	use "${data}Water_Final.dta", clear
	foreach i in $`k' {
	egen max_`i'=max(`i')
	replace `i'=max_`i'
	}
	eststo summstats5: estpost summarize $`k'
* Missing 
	use "${data}Water_Final.dta", clear
	foreach i in $`k' {
	egen `i'_Miss=rowmiss(`i')
	egen max_`i'=sum(`i'_Miss)
	replace `i'=max_`i'
	}
	eststo summstats6: estpost summarize $`k'

esttab summstats0 summstats1 summstats2 summstats3 summstats4 summstats5 summstats6 using "${Table}Main_Table_`k'.tex", replace ///
       cell("mean (fmt(2) label(_))") label mtitles("Across" "Bangla" "Nepal" "Ghana" "Min" "Max" "Missing") nonum ///
	   substitute("{l}{\footnotesize" "{p{0.95\linewidth}}{\scriptsize" ///
	              ".00" " "  ///
				  "Permission to get drinking water sample for arsenic test" "\shortstack{Permission to get drinking\\water sample for arsenic test}" ///
				  "Arsenic Level:" "~~~" ///
				  "Blue colonies in 100 ml" "E-coli:" ///
				  "Surface water (river, stream, dam, lake, pond, canal, irrigation channel)" "Surface water"  ///
				  "Piped into dwelling" "\textbf{Water Source} \\\hline Piped into dwelling" ///
				  "Blank test for E-coli conducted" "\textbf{E-coli} \\\hline Blank test for E-coli conducted" ///
				  "WHO: Had a baby in the last 4 weeks?" "\multicolumn{2}{l}{\textbf{Panel A: WHO Check Question}} \\ \hline WHO: Had a baby in the last 4 weeks?" ///
	              "Using modern family planning (exclude condom)" "\multicolumn{2}{l}{\textbf{Panel B: Family Planning Usage}} \\ \hline Using modern family planning (exclude condom)") ///
	   title("``k''" \label{`Label`k''}) addnote("`note`k''") 
eststo clear
}

