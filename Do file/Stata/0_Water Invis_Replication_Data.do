*--------------------------------------------------------------------
* Project: Resolution of Uncertainty through Testing, The Impact of Pregnancy Tests on Reproductive and Maternal Health Beliefs and Behavior in Uganda
* File Name: Descriptive Statistics
* RA: Akito
* Last updated: Akito on 2017/04/14
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
		global box 		"/Users/akitokamei/Library/CloudStorage/Box-Box/MICS Water project/"                 
		global github	"/Users/akitokamei/GitHub/Mics_Water/"
		global Overleaf "/Users/akitokamei/Dropbox/Apps/Overleaf/"
		global DataRaw  "${box}01. 2_Pilot/Data/1_raw/"
		
	}
	
	else if c(username) == "sujey" {		
		global box 		"/Users/sujey/Box/MICS Water project/"                 
		global github	"/Users/sujey/Documents/GitHub/Mics_Water/"
		global Overleaf "/Users/sujey/Box/MICS Water project/Analysis/Docs"
		global DataRaw  "${box}01. 2_Pilot/Data/1_raw/"
		
	}
	
	global Table  "${Overleaf}MICS_Water/Table/"
	global Figure "${Overleaf}MICS_Water/Figure/"
	global Data   "${box}Data/"
	* global Temp   "${data}0_Temp/"

clear all               
set graph off
set graph on	

* Check what is happening to Nepal
use "/Users/akitokamei/Library/CloudStorage/Box-Box/MICS Water project/Analysis/Raw/Nepal/hl.dta", clear
keep if HL1==1
keep  HH1 HH2 helevel2
gen Country="Nepal"
gen helevel=helevel2

drop helevel2
fre helevel
rename helevel helevel_Nep
save "/Users/akitokamei/Library/CloudStorage/Box-Box/MICS Water project/Analysis/Raw/Nepal/hl_REVISED.dta", replace


* count: 61,252
use "${Data}Cleaned_Pooled_MICS6_Africa_Latam_Asia_2.dta", clear
	* use "${Data}Africa_Latam_Asia_Pooled_hh_data_2.dta", clear
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
	
	
	*--------------------------------------------------------------------
* Construct mutually exclusive WS10 and WQ15 variables
*
* Priority:
* Boil > chlorine/tablets > filter > solar >
* strain through cloth > settle > other > no treatment
*--------------------------------------------------------------------

foreach pair in "WS10 WS9" "WQ15 WQ14" {

    gettoken treatment mainq : pair

    capture drop `treatment'
    gen byte `treatment' = .

    * No household water treatment
    replace `treatment' = 0 if `mainq' == 2

    * Assign lower-priority affirmative methods first
    replace `treatment' = 98 if `treatment'X == "X"   // Other
    replace `treatment' = 6  if `treatment'F == "F"   // Let stand and settle
    replace `treatment' = 3  if `treatment'C == "C"   // Strain through cloth
    replace `treatment' = 5  if `treatment'E == "E"   // Solar disinfection
    replace `treatment' = 4  if `treatment'D == "D"   // Water filter

    * Chemical treatment
    replace `treatment' = 8 if `treatment'H == "H"    // Add water tablet
    replace `treatment' = 7 if `treatment'G == "G"    // Aquatab/PUR/etc.
    replace `treatment' = 2 if `treatment'B == "B"    // Bleach/chlorine

    * Highest-priority method assigned last
    replace `treatment' = 1 if `treatment'A == "A"    // Boil

    * Don't know or nonresponse
    capture replace `treatment' = 99 if ///
        `treatment'Z == "Z" & missing(`treatment')

    capture replace `treatment' = 99 if ///
        `treatment'Q == "?" & missing(`treatment')

    capture replace `treatment' = 99 if ///
        `treatment'NR == "?" & missing(`treatment')

    * Main question: don't know or no response
    replace `treatment' = 99 if inlist(`mainq', 8, 9)

    label variable `treatment' ///
        "Highest-priority household water-treatment method reported"
}

label define WS10l ///
    0  "Treat: Nothing" ///
    1  "Treat: Boil" ///
    2  "Treat: Bleach/Chlorine" ///
    3  "Treat: Strain with a cloth" ///
    4  "Treat: Filter" ///
    5  "Treat: Solar" ///
    6  "Treat: Let it settle" ///
    7  "Treat: Aquatabs/PUR" ///
    8  "Treat: Add tablet" ///
    98 "Treat: Other" ///
    99 "Treat: Do not know/missing", replace

label values WS10 WQ15 WS10l

fre WQ15
/*	
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
	
	* Count the number of water-treatment methods reported
* A-H and X are treatment responses; Z and NR are excluded

foreach i in WS10 WQ15 {

    gen `i'_n_methods = 0
    foreach j in A B C D E F G H X {
        replace `i'_n_methods = `i'_n_methods + (`i'`j' == "`j'")
    }

    * Indicator for more than one treatment method reported
    gen `i'_multiple = (`i'_n_methods > 1)

    label variable `i'_n_methods ///
        "Number of water-treatment methods reported"

    label variable `i'_multiple ///
        "More than one water-treatment method reported"

    label define multiple_lbl 0 "One or no method" 1 "More than one method", replace
    label values `i'_multiple multiple_lbl
}

tab WQ15_n_methods, missing
tab WQ15_multiple , missing
	
	* WQ15 (water treatment methods are missing if WQ14 is do not know or no response)
	tab     WQ14 water_treatment,m
	replace WQ15=. if WQ14==8 | WQ14==9
	
*/	
	
	label define WS10l 0 "Treat: Nothing" 1 "Treat: Boil" 2 "Treat: Bleach/Chlorine" 3 "Treat: Stain with a cloth" 4 "Treat: Filter" 5 "Treat: Soler" 6 "Treat: Let it settle" 7 "Treat: Aquatabs/PUR" 8 "Treat: Add tablet" 98 "Treat: Other" 99 "Treat: Do not know/missing", modify
	label values WS10 WQ15 WS10l
	
	* Grouping water treatment
	gen    WQ15_g=WQ15
	recode WQ15_g 2 7 8=2 3 6=3
	label define WQ15_gl 0 "Treat: Nothing" 1 "Treat: Boil" 2 "Treat: Chlorine/Aquatabs/PUR" 3 "Treat: Strain/Settle" 4 "Treat: Filter" 5 "Treat: Soler" 8 "Treat: Add tablet" 98 "Treat: Other" 99 "Treat: Do not know/missing", modify
	label values WQ15_g WQ15_gl
	fre WQ15_g
	
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

gen     windex_ur=windex5u
replace windex_ur=windex5r if windex_ur==.
recode  windex_ur 1 2=1 3 4 5=2
gen          windex5_categ=windex5
recode       windex5_categ 1 2=1 3=2 4 5=3
label define windex5_categl 1 "Poor" 2 "Middle" 3 "Rich/Richest", modify
label values windex5_categ windex5_categl

drop windex_ur

save "${Data}MASTER_MICS.dta", replace

use "${Data}MASTER_MICS.dta", clear

foreach c in "Bangladesh" "Benin" "Gambia" "Madagascar" ///
             "Mongolia" "Togo" "Viet Nam" "Zimbabwe" {

    di "------------------------------"
    di "`c'"
    tab helevel1 if Country=="`c'", missing
    tab helevel2 if Country=="`c'", missing
    tab helevelx if Country=="`c'", missing
}

tab     WQ15_g_99, missing
drop if WQ15_g_99 == 1
drop    WQ15_g_99

*--------------------------------------------------------------------
* Harmonize education level of household head
*
* Final coding:
* 0 = None
* 1 = Primary
* 2 = Lower secondary
* 3 = Upper secondary
* 4 = College/tertiary
*--------------------------------------------------------------------

*--------------------------------------------------------------------
* Harmonize education of household head for Nepal
*--------------------------------------------------------------------

capture drop helevel_Nep
gen byte helevel_Nep = .

replace helevel_Nep = 0 if Country == "Nepal" & helevel2 == 0
replace helevel_Nep = 1 if Country == "Nepal" & helevel2 == 3
replace helevel_Nep = 2 if Country == "Nepal" & helevel2 == 4
replace helevel_Nep = 3 if Country == "Nepal" & helevel2 == 6
replace helevel_Nep = 4 if Country == "Nepal" & helevel2 == 7

* Don't know and missing
replace helevel_Nep = . if Country == "Nepal" & inlist(helevel2, 8, 9)

* Verify the mapping before replacing the final variable
tab helevel2 helevel_Nep if Country == "Nepal", missing

* Insert the Nepal coding into the final harmonized variable
replace helevel = helevel_Nep if Country == "Nepal"

label define helevel_l ///
    0 "None" ///
    1 "Primary" ///
    2 "Lower secondary" ///
    3 "Upper secondary" ///
    4 "College/tertiary", replace

label values helevel helevel_l
label variable helevel "Education level of household head"

tab helevel if Country == "Nepal", missing

* Drop source variables after verification
capture drop helevel1 helevel2 helevelx helevel_Nep

save "${Data}MASTER_MICS_DDML_FINAL.dta", replace

use "${Data}MASTER_MICS_DDML_FINAL.dta", clear

