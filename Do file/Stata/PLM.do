*--------------------------------------------------------------------*
* PROJECT: E-coli – DDML (PLM) with pystacked (logit + ML + Best)
*--------------------------------------------------------------------*

clear all
capture log close _all
set more off
set varabbrev off
set emptycells drop
set seed 12345
set linesize 135
set graph off

*--------------------------------------------------------------------*
* 1. PATHS
*--------------------------------------------------------------------*
if c(username) == "akitokamei" {		
    global Dropbox "/Users/akitokamei/Library/CloudStorage/Dropbox/"
    global Overleaf "${Dropbox}Apps/Overleaf/"	
}
else if c(username) == "jadrk" {		
    global Dropbox "C:/Users/jadrk/Dropbox/"
    global Overleaf "C:/Users/jadrk/Dropbox/"
}

global Tables     "${Overleaf}MICS_DDML/Table/"
global Figures    "${Overleaf}MICS_DDML/Figure/"
global Data_Clean "${Dropbox}MICS_DDML/Data/2. Clean/"
global Data_Final "${Dropbox}MICS_DDML/Data/3. Final/"

* Base controls
global X_base i.windex5 helevel  i.country_cat i.WS1_g

* Extended controls
global X_ext ///
    $X_base Any_U5 Girls_less_than15 Boys_15or_less Pit_latrine Open_defecation i.water_carrier_edu


*--------------------------------------------------------------------*
* 2. GLOBALS PARA MÉTODOS PYSTACKED (incluye gradboost)
*--------------------------------------------------------------------*

global PYSTACK_Y_METHODS ///
    logit lassocv ridgecv elasticcv rf gradboost

global PYSTACK_D_METHODS ///
    logit lassocv ridgecv elasticcv rf gradboost

* Macros por método
global M_logit     logit
global M_lasso     lassocv
global M_ridge     ridgecv
global M_elastic   elasticcv
global M_rf        rf
global M_gradboost gradboost
global M_best      logit lassocv ridgecv elasticcv rf gradboost   // stacked

global T water_treatment


*--------------------------------------------------------------------*
* 3. PROGRAM TO LOAD FINAL DATA
*--------------------------------------------------------------------*
cap program drop start_from_final
program define start_from_final
    use "${Data_Final}MASTER_MICS_FINAL.dta", clear
    * sample 1    // solo para debug
end


*--------------------------------------------------------------------*
* 4. PROGRAMA DDML-PLM PARA UN CONJUNTO DE MÉTODOS
*--------------------------------------------------------------------*
cap program drop do_ddml_plm_one
program define do_ddml_plm_one
    syntax , outcome(name) xset(name) mm(name) suffix(name)

    if ("`xset'" == "base") {
        local Xcontrols "$X_base"
    }
    else if ("`xset'" == "ext") {
        local Xcontrols "$X_ext"
    }
    else {
        di as error "xset must be base or ext"
        exit 198
    }

    ddml init partial, kfolds(5) reps(1)

    ddml E[Y|X]: pystacked `outcome' `Xcontrols', ///
        type(class) methods($`mm') njobs(-1)

    ddml E[D|X]: pystacked $T `Xcontrols', ///
        type(class) methods($`mm') njobs(-1)

    ddml crossfit
    ddml estimate

    eststo PLM_`suffix'

    if "`mm'" == "M_best" {
        ddml extract, show(stweights)

        matrix WY = r(Y1_pystacked_w_mn)
        matrix WD = r(D1_pystacked_w_mn)

        * quedarte con la columna mean_weight (2) y transponer a 1xK
        matrix WY = WY[.,"mean_weight"]
        matrix WD = WD[.,"mean_weight"]

        * guardar con nombres por sufijo
        matrix WY_`suffix' = WY'
        matrix WD_`suffix' = WD'
    }

end


*--------------------------------------------------------------------*
* 7. CORRER DDML PARA TODOS LOS OUTCOMES Y X-SETS (BENCHMARKS)
*--------------------------------------------------------------------*

start_from_final
eststo clear

*** SomeRiskHome (SR) ***

* OLS base / ext
reg SomeRiskHome $T $X_base
eststo OLS_SR_b

reg SomeRiskHome $T $X_ext
eststo OLS_SR_e

* Base
do_ddml_plm_one, outcome(SomeRiskHome) xset(base) mm(M_logit)     suffix(SR_b_logit)
do_ddml_plm_one, outcome(SomeRiskHome) xset(base) mm(M_lasso)     suffix(SR_b_las)
do_ddml_plm_one, outcome(SomeRiskHome) xset(base) mm(M_ridge)     suffix(SR_b_rid)
do_ddml_plm_one, outcome(SomeRiskHome) xset(base) mm(M_elastic)   suffix(SR_b_ela)
do_ddml_plm_one, outcome(SomeRiskHome) xset(base) mm(M_rf)        suffix(SR_b_rf)
do_ddml_plm_one, outcome(SomeRiskHome) xset(base) mm(M_gradboost) suffix(SR_b_gb)
do_ddml_plm_one, outcome(SomeRiskHome) xset(base) mm(M_best)      suffix(SR_b_bst)

* Extended
do_ddml_plm_one, outcome(SomeRiskHome) xset(ext) mm(M_logit)      suffix(SR_e_logit)
do_ddml_plm_one, outcome(SomeRiskHome) xset(ext) mm(M_lasso)      suffix(SR_e_las)
do_ddml_plm_one, outcome(SomeRiskHome) xset(ext) mm(M_ridge)      suffix(SR_e_rid)
do_ddml_plm_one, outcome(SomeRiskHome) xset(ext) mm(M_elastic)    suffix(SR_e_ela)
do_ddml_plm_one, outcome(SomeRiskHome) xset(ext) mm(M_rf)         suffix(SR_e_rf)
do_ddml_plm_one, outcome(SomeRiskHome) xset(ext) mm(M_gradboost)  suffix(SR_e_gb)
do_ddml_plm_one, outcome(SomeRiskHome) xset(ext) mm(M_best)       suffix(SR_e_bst)


*** VeryHighRiskHome (VH) ***

* OLS base / ext
reg VeryHighRiskHome $T $X_base
eststo OLS_VH_b

reg VeryHighRiskHome $T $X_ext
eststo OLS_VH_e

* Base
do_ddml_plm_one, outcome(VeryHighRiskHome) xset(base) mm(M_logit)     suffix(VH_b_logit)
do_ddml_plm_one, outcome(VeryHighRiskHome) xset(base) mm(M_lasso)     suffix(VH_b_las)
do_ddml_plm_one, outcome(VeryHighRiskHome) xset(base) mm(M_ridge)     suffix(VH_b_rid)
do_ddml_plm_one, outcome(VeryHighRiskHome) xset(base) mm(M_elastic)   suffix(VH_b_ela)
do_ddml_plm_one, outcome(VeryHighRiskHome) xset(base) mm(M_rf)        suffix(VH_b_rf)
do_ddml_plm_one, outcome(VeryHighRiskHome) xset(base) mm(M_gradboost) suffix(VH_b_gb)
do_ddml_plm_one, outcome(VeryHighRiskHome) xset(base) mm(M_best)      suffix(VH_b_bst)

* Extended
do_ddml_plm_one, outcome(VeryHighRiskHome) xset(ext) mm(M_logit)      suffix(VH_e_logit)
do_ddml_plm_one, outcome(VeryHighRiskHome) xset(ext) mm(M_lasso)      suffix(VH_e_las)
do_ddml_plm_one, outcome(VeryHighRiskHome) xset(ext) mm(M_ridge)      suffix(VH_e_rid)
do_ddml_plm_one, outcome(VeryHighRiskHome) xset(ext) mm(M_elastic)    suffix(VH_e_ela)
do_ddml_plm_one, outcome(VeryHighRiskHome) xset(ext) mm(M_rf)         suffix(VH_e_rf)
do_ddml_plm_one, outcome(VeryHighRiskHome) xset(ext) mm(M_gradboost)  suffix(VH_e_gb)
do_ddml_plm_one, outcome(VeryHighRiskHome) xset(ext) mm(M_best)       suffix(VH_e_bst)


*--------------------------------------------------------------------*
* 9. Tablas LaTeX – OLS + todos los métodos + Best (benchmark total)
*--------------------------------------------------------------------*

*** 9.1 SomeRiskHome (SR) ***

local outfile_SR "${Tables}SomeRiskHome_PLM_BASE_EXT_All.tex"

* Panel A: Base controls
esttab OLS_SR_b ///
       PLM_SR_b_logit ///
       PLM_SR_b_las ///
       PLM_SR_b_rid ///
       PLM_SR_b_ela ///
       PLM_SR_b_rf ///
       PLM_SR_b_gb ///
       PLM_SR_b_bst ///
    using "`outfile_SR'", replace ///
    booktabs fragment ///
    mtitle("OLS (LPM)" "Logit" "Lasso" "Ridge" "Elastic net" "Random forest" "Gradient boost" "Best (stacked)") ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    keep($T) ///
    stats(N, fmt(%9.0fc) labels("Observations")) ///
    prehead("\begin{table}[htbp]\centering" ///
            "\caption{Effect of Household Water Treatment on E.coli Risk (SomeRiskHome)}" ///
            "\begin{tabular}{l*{@M}{c}}" ///
            "\toprule") ///
    posthead("\multicolumn{@span}{l}{\textbf{Panel A: Base controls}}\\" ///
             "\midrule") ///
    postfoot("")

* Panel B: Extended controls
esttab OLS_SR_e ///
       PLM_SR_e_logit ///
       PLM_SR_e_las ///
       PLM_SR_e_rid ///
       PLM_SR_e_ela ///
       PLM_SR_e_rf ///
       PLM_SR_e_gb ///
       PLM_SR_e_bst ///
    using "`outfile_SR'", append ///
    booktabs fragment ///
    nomtitle nonumber ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    keep($T) ///
    stats(N, fmt(%9.0fc) labels("Observations")) ///
    prehead("\midrule" ///
            "\multicolumn{@span}{l}{\textbf{Panel B: Extended controls}}\\" ///
            "\midrule") ///
    posthead("") ///
    postfoot("\bottomrule\end{tabular}\end{table}")


*** 9.2 VeryHighRiskHome (VH) ***

local outfile_VH "${Tables}VeryHighRiskHome_PLM_BASE_EXT_All.tex"

* Panel A: Base controls
esttab OLS_VH_b ///
       PLM_VH_b_logit ///
       PLM_VH_b_las ///
       PLM_VH_b_rid ///
       PLM_VH_b_ela ///
       PLM_VH_b_rf ///
       PLM_VH_b_gb ///
       PLM_VH_b_bst ///
    using "`outfile_VH'", replace ///
    booktabs fragment ///
    mtitle("OLS (LPM)" "Logit" "Lasso" "Ridge" "Elastic net" "Random forest" "Gradient boost" "Best (stacked)") ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    keep($T) ///
    stats(N, fmt(%9.0fc) labels("Observations")) ///
    prehead("\begin{table}[htbp]\centering" ///
            "\caption{Effect of Household Water Treatment on Very High E.coli Risk (VeryHighRiskHome)}" ///
            "\begin{tabular}{l*{@M}{c}}" ///
            "\toprule") ///
    posthead("\multicolumn{@span}{l}{\textbf{Panel A: Base controls}}\\" ///
             "\midrule") ///
    postfoot("")

* Panel B: Extended controls
esttab OLS_VH_e ///
       PLM_VH_e_logit ///
       PLM_VH_e_las ///
       PLM_VH_e_rid ///
       PLM_VH_e_ela ///
       PLM_VH_e_rf ///
       PLM_VH_e_gb ///
       PLM_VH_e_bst ///
    using "`outfile_VH'", append ///
    booktabs fragment ///
    nomtitle nonumber ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    keep($T) ///
    stats(N, fmt(%9.0fc) labels("Observations")) ///
    prehead("\midrule" ///
            "\multicolumn{@span}{l}{\textbf{Panel B: Extended controls}}\\" ///
            "\midrule") ///
    posthead("") ///
    postfoot("\bottomrule\end{tabular}\end{table}")


*--------------------------------------------------------------------*
* 9.9 Construir matrices apiladas de pesos Best (Y y D) – Benchmarks
*--------------------------------------------------------------------*

matrix WY_all = ///
    WY_SR_b_bst \ ///
    WY_SR_e_bst \ ///
    WY_VH_b_bst \ ///
    WY_VH_e_bst

matrix WD_all = ///
    WD_SR_b_bst \ ///
    WD_SR_e_bst \ ///
    WD_VH_b_bst \ ///
    WD_VH_e_bst


*--------------------------------------------------------------------*
* 10. Tabla combinada de pesos Best (Y y D) – Benchmarks
*--------------------------------------------------------------------*

local K     : word count $M_best
local ncols = `K' + 1
local outfile "${Tables}stack_weights_Best_YD.tex"

cap file close sw
file open sw using "`outfile'", write replace

file write sw "\begin{table}[htbp]\centering" _n
file write sw "\caption{Stacking weights for Best (stacked) for \(E[Y \mid X]\) and \(E[D \mid X]\)}" _n

file write sw "\begin{tabular}{l"
forvalues j = 1/`K' {
    file write sw "c"
}
file write sw "}" _n

file write sw "\toprule" _n

* Panel A: E[Y|X]
local panelA "\multicolumn{`ncols'}{l}{\textbf{Panel A: Outcome equation \(E[Y \mid X]\)}}\\"
file write sw "`panelA'" _n
file write sw "\midrule" _n

file write sw "Specification"
forvalues j = 1/`K' {
    local m : word `j' of $M_best
    file write sw " & `m'"
}
file write sw " \\" _n
file write sw "\midrule" _n

file write sw "SomeRiskHome, base"
forvalues j = 1/`K' {
    local val  = WY_all[1,`j']
    local sval : display %6.3f `val'
    file write sw " & `sval'"
}
file write sw " \\" _n

file write sw "SomeRiskHome, extended"
forvalues j = 1/`K' {
    local val  = WY_all[2,`j']
    local sval : display %6.3f `val'
    file write sw " & `sval'"
}
file write sw " \\" _n

file write sw "VeryHighRiskHome, base"
forvalues j = 1/`K' {
    local val  = WY_all[3,`j']
    local sval : display %6.3f `val'
    file write sw " & `sval'"
}
file write sw " \\" _n

file write sw "VeryHighRiskHome, extended"
forvalues j = 1/`K' {
    local val  = WY_all[4,`j']
    local sval : display %6.3f `val'
    file write sw " & `sval'"
}
file write sw " \\" _n

file write sw "\midrule" _n

* Panel B: E[D|X]
local panelB "\multicolumn{`ncols'}{l}{\textbf{Panel B: Treatment equation \(E[D \mid X]\)}}\\"
file write sw "`panelB'" _n
file write sw "\midrule" _n

file write sw "Specification"
forvalues j = 1/`K' {
    local m : word `j' of $M_best
    file write sw " & `m'"
}
file write sw " \\" _n
file write sw "\midrule" _n

file write sw "SomeRiskHome, base"
forvalues j = 1/`K' {
    local val  = WD_all[1,`j']
    local sval : display %6.3f `val'
    file write sw " & `sval'"
}
file write sw " \\" _n

file write sw "SomeRiskHome, extended"
forvalues j = 1/`K' {
    local val  = WD_all[2,`j']
    local sval : display %6.3f `val'
    file write sw " & `sval'"
}
file write sw " \\" _n

file write sw "VeryHighRiskHome, base"
forvalues j = 1/`K' {
    local val  = WD_all[3,`j']
    local sval : display %6.3f `val'
    file write sw " & `sval'"
}
file write sw " \\" _n

file write sw "VeryHighRiskHome, extended"
forvalues j = 1/`K' {
    local val  = WD_all[4,`j']
    local sval : display %6.3f `val'
    file write sw " & `sval'"
}
file write sw " \\" _n

file write sw "\bottomrule" _n
file write sw "\end{tabular}" _n
file write sw "\end{table}" _n

file close sw


*--------------------------------------------------------------------*
* 11. SUBMUESTRAS POR RiskSource – DDML (PLM) con pystacked
*--------------------------------------------------------------------*

start_from_final
eststo clear

local outcomes  "SomeRiskHome VeryHighRiskHome"
local subsets   0 1 2

foreach y of local outcomes {

    local yshort = cond("`y'"=="SomeRiskHome","SR","VH")

    foreach s of local subsets {
        
        preserve
        keep if RiskSource == `s'

        local rshort = "r`s'"
        local S = "`yshort'_`rshort'"

        * OLS base / ext
        reg `y' $T $X_base
        eststo OLS_`S'_b

        reg `y' $T $X_ext
        eststo OLS_`S'_e

        * DDML-PLM: Base controls
        do_ddml_plm_one, outcome(`y') xset(base) mm(M_logit)     suffix(`S'_b_logit)
        do_ddml_plm_one, outcome(`y') xset(base) mm(M_lasso)     suffix(`S'_b_las)
        do_ddml_plm_one, outcome(`y') xset(base) mm(M_ridge)     suffix(`S'_b_rid)
        do_ddml_plm_one, outcome(`y') xset(base) mm(M_elastic)   suffix(`S'_b_ela)
        do_ddml_plm_one, outcome(`y') xset(base) mm(M_rf)        suffix(`S'_b_rf)
        do_ddml_plm_one, outcome(`y') xset(base) mm(M_gradboost) suffix(`S'_b_gb)
        do_ddml_plm_one, outcome(`y') xset(base) mm(M_best)      suffix(`S'_b_bst)

        * DDML-PLM: Extended controls
        do_ddml_plm_one, outcome(`y') xset(ext) mm(M_logit)      suffix(`S'_e_logit)
        do_ddml_plm_one, outcome(`y') xset(ext) mm(M_lasso)      suffix(`S'_e_las)
        do_ddml_plm_one, outcome(`y') xset(ext) mm(M_ridge)      suffix(`S'_e_rid)
        do_ddml_plm_one, outcome(`y') xset(ext) mm(M_elastic)    suffix(`S'_e_ela)
        do_ddml_plm_one, outcome(`y') xset(ext) mm(M_rf)         suffix(`S'_e_rf)
        do_ddml_plm_one, outcome(`y') xset(ext) mm(M_gradboost)  suffix(`S'_e_gb)
        do_ddml_plm_one, outcome(`y') xset(ext) mm(M_best)       suffix(`S'_e_bst)

        restore
    }
}


*--------------------------------------------------------------------*
* 12. Tablas combinadas por outcome (Benchmark + RiskSource)
*--------------------------------------------------------------------*
*--------------------------------------------------------------*
* Programa: tablas por outcome, base y extended por separado
*--------------------------------------------------------------*
cap program drop make_panel_tables_two
program define make_panel_tables_two
    syntax , outcome(string)

    * Abreviatura SR / VH
    local yshort = cond("`outcome'"=="SomeRiskHome","SR","VH")

    * Captions y archivos según outcome
    if "`outcome'" == "SomeRiskHome" {
        local capB "Effect of Household Water Treatment on E.coli Risk (SomeRiskHome), base controls"
        local capE "Effect of Household Water Treatment on E.coli Risk (SomeRiskHome), extended controls"
        local outfileB "${Tables}SomeRiskHome_PLM_BASE_RiskSource_All.tex"
        local outfileE "${Tables}SomeRiskHome_PLM_EXT_RiskSource_All.tex"
    }
    else if "`outcome'" == "VeryHighRiskHome" {
        local capB "Effect of Household Water Treatment on Very High E.coli Risk (VeryHighRiskHome), base controls"
        local capE "Effect of Household Water Treatment on Very High E.coli Risk (VeryHighRiskHome), extended controls"
        local outfileB "${Tables}VeryHighRiskHome_PLM_BASE_RiskSource_All.tex"
        local outfileE "${Tables}VeryHighRiskHome_PLM_EXT_RiskSource_All.tex"
    }

    *================ TABLA BASE (A–D) ================*

    * Panel A: Benchmark, base
    esttab OLS_`yshort'_b ///
           PLM_`yshort'_b_logit ///
           PLM_`yshort'_b_las ///
           PLM_`yshort'_b_rid ///
           PLM_`yshort'_b_ela ///
           PLM_`yshort'_b_rf ///
           PLM_`yshort'_b_gb ///
           PLM_`yshort'_b_bst ///
        using "`outfileB'", replace ///
        booktabs fragment ///
        mtitle("OLS (LPM)" "Logit" "Lasso" "Ridge" "Elastic net" "Random forest" "Gradient boost" "Best (stacked)") ///
        b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
        keep($T) ///
        stats(N, fmt(%9.0fc) labels("Observations")) ///
        prehead("\begin{table}[htbp]\centering" ///
                "\caption{`capB'}" ///
                "\begin{tabular}{l*{@M}{c}}" ///
                "\toprule") ///
        posthead("\multicolumn{@span}{l}{\textbf{Panel A: Benchmark (Total population)}}\\" ///
                 "\midrule") ///
        postfoot("")

    * Panel B: RS = 0, base
    esttab OLS_`yshort'_r0_b ///
           PLM_`yshort'_r0_b_logit ///
           PLM_`yshort'_r0_b_las ///
           PLM_`yshort'_r0_b_rid ///
           PLM_`yshort'_r0_b_ela ///
           PLM_`yshort'_r0_b_rf ///
           PLM_`yshort'_r0_b_gb ///
           PLM_`yshort'_r0_b_bst ///
        using "`outfileB'", append ///
        booktabs fragment nomtitle nonumber ///
        b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
        keep($T) ///
        stats(N, fmt(%9.0fc) labels("Observations")) ///
        prehead("\midrule" ///
                "\multicolumn{@span}{l}{\textbf{Panel B: RiskSource = 0}}\\" ///
                "\midrule") ///
        posthead("") ///
        postfoot("")

    * Panel C: RS = 1, base
    esttab OLS_`yshort'_r1_b ///
           PLM_`yshort'_r1_b_logit ///
           PLM_`yshort'_r1_b_las ///
           PLM_`yshort'_r1_b_rid ///
           PLM_`yshort'_r1_b_ela ///
           PLM_`yshort'_r1_b_rf ///
           PLM_`yshort'_r1_b_gb ///
           PLM_`yshort'_r1_b_bst ///
        using "`outfileB'", append ///
        booktabs fragment nomtitle nonumber ///
        b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
        keep($T) ///
        stats(N, fmt(%9.0fc) labels("Observations")) ///
        prehead("\midrule" ///
                "\multicolumn{@span}{l}{\textbf{Panel C: RiskSource = 1}}\\" ///
                "\midrule") ///
        posthead("") ///
        postfoot("")

    * Panel D: RS = 2, base
    esttab OLS_`yshort'_r2_b ///
           PLM_`yshort'_r2_b_logit ///
           PLM_`yshort'_r2_b_las ///
           PLM_`yshort'_r2_b_rid ///
           PLM_`yshort'_r2_b_ela ///
           PLM_`yshort'_r2_b_rf ///
           PLM_`yshort'_r2_b_gb ///
           PLM_`yshort'_r2_b_bst ///
        using "`outfileB'", append ///
        booktabs fragment nomtitle nonumber ///
        b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
        keep($T) ///
        stats(N, fmt(%9.0fc) labels("Observations")) ///
        prehead("\midrule" ///
                "\multicolumn{@span}{l}{\textbf{Panel D: RiskSource = 2}}\\" ///
                "\midrule") ///
        posthead("") ///
        postfoot("\bottomrule\end{tabular}\end{table}")

    *================ TABLA EXTENDED (A–D) ================*

    * Panel A: Benchmark, extended
    esttab OLS_`yshort'_e ///
           PLM_`yshort'_e_logit ///
           PLM_`yshort'_e_las ///
           PLM_`yshort'_e_rid ///
           PLM_`yshort'_e_ela ///
           PLM_`yshort'_e_rf ///
           PLM_`yshort'_e_gb ///
           PLM_`yshort'_e_bst ///
        using "`outfileE'", replace ///
        booktabs fragment ///
        mtitle("OLS (LPM)" "Logit" "Lasso" "Ridge" "Elastic net" "Random forest" "Gradient boost" "Best (stacked)") ///
        b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
        keep($T) ///
        stats(N, fmt(%9.0fc) labels("Observations")) ///
        prehead("\begin{table}[htbp]\centering" ///
                "\caption{`capE'}" ///
                "\begin{tabular}{l*{@M}{c}}" ///
                "\toprule") ///
        posthead("\multicolumn{@span}{l}{\textbf{Panel A: Benchmark (Total population)}}\\" ///
                 "\midrule") ///
        postfoot("")

    * Panel B: RS = 0, extended
    esttab OLS_`yshort'_r0_e ///
           PLM_`yshort'_r0_e_logit ///
           PLM_`yshort'_r0_e_las ///
           PLM_`yshort'_r0_e_rid ///
           PLM_`yshort'_r0_e_ela ///
           PLM_`yshort'_r0_e_rf ///
           PLM_`yshort'_r0_e_gb ///
           PLM_`yshort'_r0_e_bst ///
        using "`outfileE'", append ///
        booktabs fragment nomtitle nonumber ///
        b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
        keep($T) ///
        stats(N, fmt(%9.0fc) labels("Observations")) ///
        prehead("\midrule" ///
                "\multicolumn{@span}{l}{\textbf{Panel B: RiskSource = 0}}\\" ///
                "\midrule") ///
        posthead("") ///
        postfoot("")

    * Panel C: RS = 1, extended
    esttab OLS_`yshort'_r1_e ///
           PLM_`yshort'_r1_e_logit ///
           PLM_`yshort'_r1_e_las ///
           PLM_`yshort'_r1_e_rid ///
           PLM_`yshort'_r1_e_ela ///
           PLM_`yshort'_r1_e_rf ///
           PLM_`yshort'_r1_e_gb ///
           PLM_`yshort'_r1_e_bst ///
        using "`outfileE'", append ///
        booktabs fragment nomtitle nonumber ///
        b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
        keep($T) ///
        stats(N, fmt(%9.0fc) labels("Observations")) ///
        prehead("\midrule" ///
                "\multicolumn{@span}{l}{\textbf{Panel C: RiskSource = 1}}\\" ///
                "\midrule") ///
        posthead("") ///
        postfoot("")

    * Panel D: RS = 2, extended
    esttab OLS_`yshort'_r2_e ///
           PLM_`yshort'_r2_e_logit ///
           PLM_`yshort'_r2_e_las ///
           PLM_`yshort'_r2_e_rid ///
           PLM_`yshort'_r2_e_ela ///
           PLM_`yshort'_r2_e_rf ///
           PLM_`yshort'_r2_e_gb ///
           PLM_`yshort'_r2_e_bst ///
        using "`outfileE'", append ///
        booktabs fragment nomtitle nonumber ///
        b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
        keep($T) ///
        stats(N, fmt(%9.0fc) labels("Observations")) ///
        prehead("\midrule" ///
                "\multicolumn{@span}{l}{\textbf{Panel D: RiskSource = 2}}\\" ///
                "\midrule") ///
        posthead("") ///
        postfoot("\bottomrule\end{tabular}\end{table}")

end



* Para SomeRiskHome (SR)
make_panel_tables_two, outcome(SomeRiskHome)

* Para VeryHighRiskHome (VH)
make_panel_tables_two, outcome(VeryHighRiskHome)

*--------------------------------------------------------------------*
* 13. Matrices y tablas de pesos Best por RiskSource (BASE + EXT)
*--------------------------------------------------------------------*

*==================== MATRICES BASE ====================*

* SomeRiskHome – E[Y|X], base
matrix WY_SR_all_RS = ///
    WY_SR_b_bst    \ ///
    WY_SR_r0_b_bst \ ///
    WY_SR_r1_b_bst \ ///
    WY_SR_r2_b_bst

* SomeRiskHome – E[D|X], base
matrix WD_SR_all_RS = ///
    WD_SR_b_bst    \ ///
    WD_SR_r0_b_bst \ ///
    WD_SR_r1_b_bst \ ///
    WD_SR_r2_b_bst

* VeryHighRiskHome – E[Y|X], base
matrix WY_VH_all_RS = ///
    WY_VH_b_bst    \ ///
    WY_VH_r0_b_bst \ ///
    WY_VH_r1_b_bst \ ///
    WY_VH_r2_b_bst

* VeryHighRiskHome – E[D|X], base
matrix WD_VH_all_RS = ///
    WD_VH_b_bst    \ ///
    WD_VH_r0_b_bst \ ///
    WD_VH_r1_b_bst \ ///
    WD_VH_r2_b_bst

*==================== MATRICES EXTENDED ====================*

* SomeRiskHome – E[Y|X], extended
matrix WY_SR_all_RS_ext = ///
    WY_SR_e_bst    \ ///
    WY_SR_r0_e_bst \ ///
    WY_SR_r1_e_bst \ ///
    WY_SR_r2_e_bst

* SomeRiskHome – E[D|X], extended
matrix WD_SR_all_RS_ext = ///
    WD_SR_e_bst    \ ///
    WD_SR_r0_e_bst \ ///
    WD_SR_r1_e_bst \ ///
    WD_SR_r2_e_bst

* VeryHighRiskHome – E[Y|X], extended
matrix WY_VH_all_RS_ext = ///
    WY_VH_e_bst    \ ///
    WY_VH_r0_e_bst \ ///
    WY_VH_r1_e_bst \ ///
    WY_VH_r2_e_bst

* VeryHighRiskHome – E[D|X], extended
matrix WD_VH_all_RS_ext = ///
    WD_VH_e_bst    \ ///
    WD_VH_r0_e_bst \ ///
    WD_VH_r1_e_bst \ ///
    WD_VH_r2_e_bst


*--------------------------------------------------------------------*
* 13.2 Tabla LaTeX de pesos Best – SomeRiskHome (base + extended)
*--------------------------------------------------------------------*

local K     : word count $M_best
local ncols = `K' + 1
local outfile_SRw "${Tables}stack_weights_Best_YD_SomeRiskHome_RiskSource.tex"

cap file close swSR
file open swSR using "`outfile_SRw'", write replace

file write swSR "\begin{table}[htbp]\centering" _n
file write swSR "\caption{Stacking weights for Best (stacked) for \(E[Y \mid X]\) and \(E[D \mid X]\) – SomeRiskHome, by RiskSource}" _n

file write swSR "\begin{tabular}{l"
forvalues j = 1/`K' {
    file write swSR "c"
}
file write swSR "}" _n

file write swSR "\toprule" _n

* -------- Panel A: E[Y|X], base --------
local panelA2 "\multicolumn{`ncols'}{l}{\textbf{Panel A: Outcome equation \(E[Y \mid X]\), base controls}}\\"
file write swSR "`panelA2'" _n
file write swSR "\midrule" _n

file write swSR "Specification"
forvalues j = 1/`K' {
    local m : word `j' of $M_best
    file write swSR " & `m'"
}
file write swSR " \\" _n
file write swSR "\midrule" _n

file write swSR "SomeRiskHome, total"
forvalues j = 1/`K' {
    local val  = WY_SR_all_RS[1,`j']
    local sval : display %6.3f `val'
    file write swSR " & `sval'"
}
file write swSR " \\" _n

file write swSR "SomeRiskHome, RiskSource = 0"
forvalues j = 1/`K' {
    local val  = WY_SR_all_RS[2,`j']
    local sval : display %6.3f `val'
    file write swSR " & `sval'"
}
file write swSR " \\" _n

file write swSR "SomeRiskHome, RiskSource = 1"
forvalues j = 1/`K' {
    local val  = WY_SR_all_RS[3,`j']
    local sval : display %6.3f `val'
    file write swSR " & `sval'"
}
file write swSR " \\" _n

file write swSR "SomeRiskHome, RiskSource = 2"
forvalues j = 1/`K' {
    local val  = WY_SR_all_RS[4,`j']
    local sval : display %6.3f `val'
    file write swSR " & `sval'"
}
file write swSR " \\" _n

file write swSR "\midrule" _n

* -------- Panel B: E[D|X], base --------
local panelB2 "\multicolumn{`ncols'}{l}{\textbf{Panel B: Treatment equation \(E[D \mid X]\), base controls}}\\"
file write swSR "`panelB2'" _n
file write swSR "\midrule" _n

file write swSR "Specification"
forvalues j = 1/`K' {
    local m : word `j' of $M_best
    file write swSR " & `m'"
}
file write swSR " \\" _n
file write swSR "\midrule" _n

file write swSR "SomeRiskHome, total"
forvalues j = 1/`K' {
    local val  = WD_SR_all_RS[1,`j']
    local sval : display %6.3f `val'
    file write swSR " & `sval'"
}
file write swSR " \\" _n

file write swSR "SomeRiskHome, RiskSource = 0"
forvalues j = 1/`K' {
    local val  = WD_SR_all_RS[2,`j']
    local sval : display %6.3f `val'
    file write swSR " & `sval'"
}
file write swSR " \\" _n

file write swSR "SomeRiskHome, RiskSource = 1"
forvalues j = 1/`K' {
    local val  = WD_SR_all_RS[3,`j']
    local sval : display %6.3f `val'
    file write swSR " & `sval'"
}
file write swSR " \\" _n

file write swSR "SomeRiskHome, RiskSource = 2"
forvalues j = 1/`K' {
    local val  = WD_SR_all_RS[4,`j']
    local sval : display %6.3f `val'
    file write swSR " & `sval'"
}
file write swSR " \\" _n

file write swSR "\midrule" _n

* -------- Panel C: E[Y|X], extended --------
local panelC2 "\multicolumn{`ncols'}{l}{\textbf{Panel C: Outcome equation \(E[Y \mid X]\), extended controls}}\\"
file write swSR "`panelC2'" _n
file write swSR "\midrule" _n

file write swSR "Specification"
forvalues j = 1/`K' {
    local m : word `j' of $M_best
    file write swSR " & `m'"
}
file write swSR " \\" _n
file write swSR "\midrule" _n

file write swSR "SomeRiskHome, total"
forvalues j = 1/`K' {
    local val  = WY_SR_all_RS_ext[1,`j']
    local sval : display %6.3f `val'
    file write swSR " & `sval'"
}
file write swSR " \\" _n

file write swSR "SomeRiskHome, RiskSource = 0"
forvalues j = 1/`K' {
    local val  = WY_SR_all_RS_ext[2,`j']
    local sval : display %6.3f `val'
    file write swSR " & `sval'"
}
file write swSR " \\" _n

file write swSR "SomeRiskHome, RiskSource = 1"
forvalues j = 1/`K' {
    local val  = WY_SR_all_RS_ext[3,`j']
    local sval : display %6.3f `val'
    file write swSR " & `sval'"
}
file write swSR " \\" _n

file write swSR "SomeRiskHome, RiskSource = 2"
forvalues j = 1/`K' {
    local val  = WY_SR_all_RS_ext[4,`j']
    local sval : display %6.3f `val'
    file write swSR " & `sval'"
}
file write swSR " \\" _n

file write swSR "\midrule" _n

* -------- Panel D: E[D|X], extended --------
local panelD2 "\multicolumn{`ncols'}{l}{\textbf{Panel D: Treatment equation \(E[D \mid X]\), extended controls}}\\"
file write swSR "`panelD2'" _n
file write swSR "\midrule" _n

file write swSR "Specification"
forvalues j = 1/`K' {
    local m : word `j' of $M_best
    file write swSR " & `m'"
}
file write swSR " \\" _n
file write swSR "\midrule" _n

file write swSR "SomeRiskHome, total"
forvalues j = 1/`K' {
    local val  = WD_SR_all_RS_ext[1,`j']
    local sval : display %6.3f `val'
    file write swSR " & `sval'"
}
file write swSR " \\" _n

file write swSR "SomeRiskHome, RiskSource = 0"
forvalues j = 1/`K' {
    local val  = WD_SR_all_RS_ext[2,`j']
    local sval : display %6.3f `val'
    file write swSR " & `sval'"
}
file write swSR " \\" _n

file write swSR "SomeRiskHome, RiskSource = 1"
forvalues j = 1/`K' {
    local val  = WD_SR_all_RS_ext[3,`j']
    local sval : display %6.3f `val'
    file write swSR " & `sval'"
}
file write swSR " \\" _n

file write swSR "SomeRiskHome, RiskSource = 2"
forvalues j = 1/`K' {
    local val  = WD_SR_all_RS_ext[4,`j']
    local sval : display %6.3f `val'
    file write swSR " & `sval'"
}
file write swSR " \\" _n

file write swSR "\bottomrule" _n
file write swSR "\end{tabular}" _n
file write swSR "\end{table}" _n

file close swSR


*--------------------------------------------------------------------*
* 13.3 Tabla LaTeX de pesos Best – VeryHighRiskHome (base + extended)
*--------------------------------------------------------------------*

local outfile_VHw "${Tables}stack_weights_Best_YD_VeryHighRiskHome_RiskSource.tex"

cap file close swVH
file open swVH using "`outfile_VHw'", write replace

file write swVH "\begin{table}[htbp]\centering" _n
file write swVH "\caption{Stacking weights for Best (stacked) for \(E[Y \mid X]\) and \(E[D \mid X]\) – VeryHighRiskHome, by RiskSource}" _n

file write swVH "\begin{tabular}{l"
forvalues j = 1/`K' {
    file write swVH "c"
}
file write swVH "}" _n

file write swVH "\toprule" _n

* Panel A: E[Y|X], base
local panelA_vh "\multicolumn{`ncols'}{l}{\textbf{Panel A: Outcome equation \(E[Y \mid X]\), base controls}}\\"
file write swVH "`panelA_vh'" _n
file write swVH "\midrule" _n

file write swVH "Specification"
forvalues j = 1/`K' {
    local m : word `j' of $M_best
    file write swVH " & `m'"
}
file write swVH " \\" _n
file write swVH "\midrule" _n

file write swVH "VeryHighRiskHome, total"
forvalues j = 1/`K' {
    local val  = WY_VH_all_RS[1,`j']
    local sval : display %6.3f `val'
    file write swVH " & `sval'"
}
file write swVH " \\" _n

file write swVH "VeryHighRiskHome, RiskSource = 0"
forvalues j = 1/`K' {
    local val  = WY_VH_all_RS[2,`j']
    local sval : display %6.3f `val'
    file write swVH " & `sval'"
}
file write swVH " \\" _n

file write swVH "VeryHighRiskHome, RiskSource = 1"
forvalues j = 1/`K' {
    local val  = WY_VH_all_RS[3,`j']
    local sval : display %6.3f `val'
    file write swVH " & `sval'"
}
file write swVH " \\" _n

file write swVH "VeryHighRiskHome, RiskSource = 2"
forvalues j = 1/`K' {
    local val  = WY_VH_all_RS[4,`j']
    local sval : display %6.3f `val'
    file write swVH " & `sval'"
}
file write swVH " \\" _n

file write swVH "\midrule" _n

* Panel B: E[D|X], base
local panelB_vh "\multicolumn{`ncols'}{l}{\textbf{Panel B: Treatment equation \(E[D \mid X]\), base controls}}\\"
file write swVH "`panelB_vh'" _n
file write swVH "\midrule" _n

file write swVH "Specification"
forvalues j = 1/`K' {
    local m : word `j' of $M_best
    file write swVH " & `m'"
}
file write swVH " \\" _n
file write swVH "\midrule" _n

file write swVH "VeryHighRiskHome, total"
forvalues j = 1/`K' {
    local val  = WD_VH_all_RS[1,`j']
    local sval : display %6.3f `val'
    file write swVH " & `sval'"
}
file write swVH " \\" _n

file write swVH "VeryHighRiskHome, RiskSource = 0"
forvalues j = 1/`K' {
    local val  = WD_VH_all_RS[2,`j']
    local sval : display %6.3f `val'
    file write swVH " & `sval'"
}
file write swVH " \\" _n

file write swVH "VeryHighRiskHome, RiskSource = 1"
forvalues j = 1/`K' {
    local val  = WD_VH_all_RS[3,`j']
    local sval : display %6.3f `val'
    file write swVH " & `sval'"
}
file write swVH " \\" _n

file write swVH "VeryHighRiskHome, RiskSource = 2"
forvalues j = 1/`K' {
    local val  = WD_VH_all_RS[4,`j']
    local sval : display %6.3f `val'
    file write swVH " & `sval'"
}
file write swVH " \\" _n

file write swVH "\midrule" _n

* Panel C: E[Y|X], extended
local panelC_vh "\multicolumn{`ncols'}{l}{\textbf{Panel C: Outcome equation \(E[Y \mid X]\), extended controls}}\\"
file write swVH "`panelC_vh'" _n
file write swVH "\midrule" _n

file write swVH "Specification"
forvalues j = 1/`K' {
    local m : word `j' of $M_best
    file write swVH " & `m'"
}
file write swVH " \\" _n
file write swVH "\midrule" _n

file write swVH "VeryHighRiskHome, total"
forvalues j = 1/`K' {
    local val  = WY_VH_all_RS_ext[1,`j']
    local sval : display %6.3f `val'
    file write swVH " & `sval'"
}
file write swVH " \\" _n

file write swVH "VeryHighRiskHome, RiskSource = 0"
forvalues j = 1/`K' {
    local val  = WY_VH_all_RS_ext[2,`j']
    local sval : display %6.3f `val'
    file write swVH " & `sval'"
}
file write swVH " \\" _n

file write swVH "VeryHighRiskHome, RiskSource = 1"
forvalues j = 1/`K' {
    local val  = WY_VH_all_RS_ext[3,`j']
    local sval : display %6.3f `val'
    file write swVH " & `sval'"
}
file write swVH " \\" _n

file write swVH "VeryHighRiskHome, RiskSource = 2"
forvalues j = 1/`K' {
    local val  = WY_VH_all_RS_ext[4,`j']
    local sval : display %6.3f `val'
    file write swVH " & `sval'"
}
file write swVH " \\" _n

file write swVH "\midrule" _n

* Panel D: E[D|X], extended
local panelD_vh "\multicolumn{`ncols'}{l}{\textbf{Panel D: Treatment equation \(E[D \mid X]\), extended controls}}\\"
file write swVH "`panelD_vh'" _n
file write swVH "\midrule" _n

file write swVH "Specification"
forvalues j = 1/`K' {
    local m : word `j' of $M_best
    file write swVH " & `m'"
}
file write swVH " \\" _n
file write swVH "\midrule" _n

file write swVH "VeryHighRiskHome, total"
forvalues j = 1/`K' {
    local val  = WD_VH_all_RS_ext[1,`j']
    local sval : display %6.3f `val'
    file write swVH " & `sval'"
}
file write swVH " \\" _n

file write swVH "VeryHighRiskHome, RiskSource = 0"
forvalues j = 1/`K' {
    local val  = WD_VH_all_RS_ext[2,`j']
    local sval : display %6.3f `val'
    file write swVH " & `sval'"
}
file write swVH " \\" _n

file write swVH "VeryHighRiskHome, RiskSource = 1"
forvalues j = 1/`K' {
    local val  = WD_VH_all_RS_ext[3,`j']
    local sval : display %6.3f `val'
    file write swVH " & `sval'"
}
file write swVH " \\" _n

file write swVH "VeryHighRiskHome, RiskSource = 2"
forvalues j = 1/`K' {
    local val  = WD_VH_all_RS_ext[4,`j']
    local sval : display %6.3f `val'
    file write swVH " & `sval'"
}
file write swVH " \\" _n

file write swVH "\bottomrule" _n
file write swVH "\end{tabular}" _n
file write swVH "\end{table}" _n

file close swVH

