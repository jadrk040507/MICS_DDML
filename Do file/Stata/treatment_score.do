*--------------------------------------------------------------------*
* 0. ENTORNO Y RUTAS
*--------------------------------------------------------------------*

clear all
set more off
set varabbrev off

if c(username) == "akitokamei" {		
    global Dropbox "/Users/akitokamei/Library/CloudStorage/Dropbox/"
    global Overleaf "${Dropbox}Apps/Overleaf/"	
}
else if c(username) == "jadrk" {		
    global Dropbox "C:/Users/jadrk/Dropbox/"
    global Overleaf "C:/Users/jadrk/Dropbox/"
}

global Data_Final "${Dropbox}MICS_DDML/Data/3. Final/"
global Tables    "${Overleaf}MICS_DDML/Table/"

*--------------------------------------------------------------------*
* 1. CARGA DE DATOS
*--------------------------------------------------------------------*

use "${Data_Final}MASTER_MICS_FINAL.dta", clear

*--------------------------------------------------------------------*
* 2. CONTROLES (TUS MACROS ORIGINALES)
*--------------------------------------------------------------------*

* Base controls
global X_base i.windex5 helevel  i.country_cat i.WS1_g

* Extended controls
global X_ext ///
    $X_base Any_U5 Girls_less_than15 Boys_15or_less Pit_latrine Open_defecation i.water_carrier_edu


*--------------------------------------------------------------------*
* 3. LOGIT DE PROPENSIÓN + EFECTOS MARGINALES
*--------------------------------------------------------------------*

eststo clear

*--- (a) Modelo base ---*
logit water_treatment $X_base
margins, dydx(*) post
eststo Propensity_base

*--- (b) Modelo extended ---*
logit water_treatment $X_ext
margins, dydx(*) post
eststo Propensity_ext

*--------------------------------------------------------------------*
* 4. TABLAS LaTeX – BASE Y EXTENDED
*--------------------------------------------------------------------*

local outfile "${Tables}Propensity_WaterTreatment_BASE_EXT_dydx.tex"

esttab Propensity_base Propensity_ext using "`outfile'", replace ///
    booktabs fragment ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    mtitle("Base controls" "Extended controls") ///
    stats(N, fmt(%9.0fc %9.0fc) labels("Observations")) ///
    coeflabels( ///
        windex5            "Wealth index (quintile)" ///
        helevel            "HH head education (3 levels)" ///
        Any_U5             "Any child under 5" ///
        Girls_less_than15  "Girls under 15" ///
        Boys_15or_less     "Boys under 15" ///
        Pit_latrine        "Pit latrine" ///
        Open_defecation    "Open defecation" ///
        water_carrier_edu  "Water carrier education" ///
    ) ///
    prehead("\begin{table}[htbp]\centering" ///
            "\caption{Determinants of household water treatment (average marginal effects)}" ///
            "\begin{tabular}{l cc}" ///
            "\toprule") ///
    posthead("") ///
    postfoot("\bottomrule\end{tabular}\end{table}")


