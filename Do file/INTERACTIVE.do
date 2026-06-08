*--------------------------------------------------------------------*
* PROJECT: E-coli – DDML (Interactive) with pystacked (logit + ML + Best)
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

*--------------------------------------------------------------------*
* 2. CONTROLES
*--------------------------------------------------------------------*

* Base controls
global X_base i.windex5 helevel  i.country_cat i.WS1_g

* Extended controls
global X_ext ///
    $X_base Any_U5 Girls_less_than15 Boys_15or_less Pit_latrine Open_defecation i.water_carrier_edu


*--------------------------------------------------------------------*
* 3. GLOBALS PARA MÉTODOS PYSTACKED
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
* 4. PROGRAM TO LOAD FINAL DATA
*--------------------------------------------------------------------*
cap program drop start_from_final
program define start_from_final
    use "${Data_Final}MASTER_MICS_FINAL.dta", clear
    * sample 10   // usar solo para debug
end

*--------------------------------------------------------------------*
* 5. PROGRAMA DDML-INTERACTIVE PARA UN CONJUNTO DE MÉTODOS (ATE + PESOS)
*--------------------------------------------------------------------*
cap program drop do_ddml_interactive_one
program define do_ddml_interactive_one
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

    ddml init interactive, kfolds(5) reps(1)

    * E[D|X]
    ddml E[D|X]: pystacked $T `Xcontrols', ///
        type(class) methods($`mm') njobs(-1)

    * E[Y|D,X]
    ddml E[Y|D,X]: pystacked `outcome' $T `Xcontrols', ///
        type(class) methods($`mm') njobs(-1)

    ddml crossfit
    ddml estimate    // ATE por defecto

    eststo INT_`suffix'

    * Pesos solo para Best (stacked)
    if "$`mm'" == "$M_best" {
        ddml extract, show(stweights)

        matrix WY = r(Y1_pystacked_w_mn)
        matrix WD = r(D1_pystacked_w_mn)

        matrix WY = WY[.,"mean_weight"]
        matrix WD = WD[.,"mean_weight"]

        matrix WY_`suffix' = WY'
        matrix WD_`suffix' = WD'
    }
end

*--------------------------------------------------------------------*
* 6. CORRER DDML-INTERACTIVE PARA TODOS LOS OUTCOMES Y X-SETS
*--------------------------------------------------------------------*

start_from_final
eststo clear

*** SomeRiskHome (SR) ***

reg SomeRiskHome $T $X_base
eststo OLS_SR_b

reg SomeRiskHome $T $X_ext
eststo OLS_SR_e

do_ddml_interactive_one, outcome(SomeRiskHome) xset(base) mm(M_logit)     suffix(SR_b_logit)
do_ddml_interactive_one, outcome(SomeRiskHome) xset(base) mm(M_lasso)     suffix(SR_b_las)
do_ddml_interactive_one, outcome(SomeRiskHome) xset(base) mm(M_ridge)     suffix(SR_b_rid)
do_ddml_interactive_one, outcome(SomeRiskHome) xset(base) mm(M_elastic)   suffix(SR_b_ela)
do_ddml_interactive_one, outcome(SomeRiskHome) xset(base) mm(M_rf)        suffix(SR_b_rf)
do_ddml_interactive_one, outcome(SomeRiskHome) xset(base) mm(M_gradboost) suffix(SR_b_gb)
do_ddml_interactive_one, outcome(SomeRiskHome) xset(base) mm(M_best)      suffix(SR_b_bst)

do_ddml_interactive_one, outcome(SomeRiskHome) xset(ext) mm(M_logit)      suffix(SR_e_logit)
do_ddml_interactive_one, outcome(SomeRiskHome) xset(ext) mm(M_lasso)      suffix(SR_e_las)
do_ddml_interactive_one, outcome(SomeRiskHome) xset(ext) mm(M_ridge)      suffix(SR_e_rid)
do_ddml_interactive_one, outcome(SomeRiskHome) xset(ext) mm(M_elastic)    suffix(SR_e_ela)
do_ddml_interactive_one, outcome(SomeRiskHome) xset(ext) mm(M_rf)         suffix(SR_e_rf)
do_ddml_interactive_one, outcome(SomeRiskHome) xset(ext) mm(M_gradboost)  suffix(SR_e_gb)
do_ddml_interactive_one, outcome(SomeRiskHome) xset(ext) mm(M_best)       suffix(SR_e_bst)

*** VeryHighRiskHome (VH) ***

reg VeryHighRiskHome $T $X_base
eststo OLS_VH_b

reg VeryHighRiskHome $T $X_ext
eststo OLS_VH_e

do_ddml_interactive_one, outcome(VeryHighRiskHome) xset(base) mm(M_logit)     suffix(VH_b_logit)
do_ddml_interactive_one, outcome(VeryHighRiskHome) xset(base) mm(M_lasso)     suffix(VH_b_las)
do_ddml_interactive_one, outcome(VeryHighRiskHome) xset(base) mm(M_ridge)     suffix(VH_b_rid)
do_ddml_interactive_one, outcome(VeryHighRiskHome) xset(base) mm(M_elastic)   suffix(VH_b_ela)
do_ddml_interactive_one, outcome(VeryHighRiskHome) xset(base) mm(M_rf)        suffix(VH_b_rf)
do_ddml_interactive_one, outcome(VeryHighRiskHome) xset(base) mm(M_gradboost) suffix(VH_b_gb)
do_ddml_interactive_one, outcome(VeryHighRiskHome) xset(base) mm(M_best)      suffix(VH_b_bst)

do_ddml_interactive_one, outcome(VeryHighRiskHome) xset(ext) mm(M_logit)      suffix(VH_e_logit)
do_ddml_interactive_one, outcome(VeryHighRiskHome) xset(ext) mm(M_lasso)      suffix(VH_e_las)
do_ddml_interactive_one, outcome(VeryHighRiskHome) xset(ext) mm(M_ridge)      suffix(VH_e_rid)
do_ddml_interactive_one, outcome(VeryHighRiskHome) xset(ext) mm(M_elastic)    suffix(VH_e_ela)
do_ddml_interactive_one, outcome(VeryHighRiskHome) xset(ext) mm(M_rf)         suffix(VH_e_rf)
do_ddml_interactive_one, outcome(VeryHighRiskHome) xset(ext) mm(M_gradboost)  suffix(VH_e_gb)
do_ddml_interactive_one, outcome(VeryHighRiskHome) xset(ext) mm(M_best)       suffix(VH_e_bst)

*--------------------------------------------------------------------*
* 7. Tablas LaTeX – OLS + Interactive (ATE) – Benchmark total
*--------------------------------------------------------------------*

*** 7.1 SomeRiskHome (SR) ***

local outfile_SR_int "${Tables}SomeRiskHome_INTERACTIVE_BASE_EXT_All.tex"

esttab OLS_SR_b ///
       INT_SR_b_logit ///
       INT_SR_b_las ///
       INT_SR_b_rid ///
       INT_SR_b_ela ///
       INT_SR_b_rf ///
       INT_SR_b_gb ///
       INT_SR_b_bst ///
    using "`outfile_SR_int'", replace ///
    booktabs fragment ///
    mtitle("OLS (LPM)" "Logit" "Lasso" "Ridge" "Elastic net" "Random forest" "Gradient boost" "Best (stacked)") ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    keep($T) ///
    stats(N, fmt(%9.0fc) labels("Observations")) ///
    prehead("\begin{table}[htbp]\centering" ///
            "\caption{Effect of Household Water Treatment on E.coli Risk (SomeRiskHome), interactive DDML}" ///
            "\begin{tabular}{l*{@M}{c}}" ///
            "\toprule") ///
    posthead("\multicolumn{@span}{l}{\textbf{Panel A: Base controls}}\\\\" ///
             "\midrule") ///
    postfoot("")

esttab OLS_SR_e ///
       INT_SR_e_logit ///
       INT_SR_e_las ///
       INT_SR_e_rid ///
       INT_SR_e_ela ///
       INT_SR_e_rf ///
       INT_SR_e_gb ///
       INT_SR_e_bst ///
    using "`outfile_SR_int'", append ///
    booktabs fragment ///
    nomtitle nonumber ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    keep($T) ///
    stats(N, fmt(%9.0fc) labels("Observations")) ///
    prehead("\midrule" ///
            "\multicolumn{@span}{l}{\textbf{Panel B: Extended controls}}\\\\" ///
            "\midrule") ///
    posthead("") ///
    postfoot("\bottomrule\end{tabular}\end{table}")

*** 7.2 VeryHighRiskHome (VH) ***

local outfile_VH_int "${Tables}VeryHighRiskHome_INTERACTIVE_BASE_EXT_All.tex"

esttab OLS_VH_b ///
       INT_VH_b_logit ///
       INT_VH_b_las ///
       INT_VH_b_rid ///
       INT_VH_b_ela ///
       INT_VH_b_rf ///
       INT_VH_b_gb ///
       INT_VH_b_bst ///
    using "`outfile_VH_int'", replace ///
    booktabs fragment ///
    mtitle("OLS (LPM)" "Logit" "Lasso" "Ridge" "Elastic net" "Random forest" "Gradient boost" "Best (stacked)") ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    keep($T) ///
    stats(N, fmt(%9.0fc) labels("Observations")) ///
    prehead("\begin{table}[htbp]\centering" ///
            "\caption{Effect of Household Water Treatment on Very High E.coli Risk (VeryHighRiskHome), interactive DDML}" ///
            "\begin{tabular}{l*{@M}{c}}" ///
            "\toprule") ///
    posthead("\multicolumn{@span}{l}{\textbf{Panel A: Base controls}}\\\\" ///
             "\midrule") ///
    postfoot("")

esttab OLS_VH_e ///
       INT_VH_e_logit ///
       INT_VH_e_las ///
       INT_VH_e_rid ///
       INT_VH_e_ela ///
       INT_VH_e_rf ///
       INT_VH_e_gb ///
       INT_VH_e_bst ///
    using "`outfile_VH_int'", append ///
    booktabs fragment ///
    nomtitle nonumber ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    keep($T) ///
    stats(N, fmt(%9.0fc) labels("Observations")) ///
    prehead("\midrule" ///
            "\multicolumn{@span}{l}{\textbf{Panel B: Extended controls}}\\\\" ///
            "\midrule") ///
    posthead("") ///
    postfoot("\bottomrule\end{tabular}\end{table}")

*--------------------------------------------------------------------*
* 8. SUBMUESTRAS POR RiskSource – DDML (Interactive)
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

        reg `y' $T $X_base
        eststo OLS_`S'_b

        reg `y' $T $X_ext
        eststo OLS_`S'_e

        do_ddml_interactive_one, outcome(`y') xset(base) mm(M_logit)     suffix(`S'_b_logit)
        do_ddml_interactive_one, outcome(`y') xset(base) mm(M_lasso)     suffix(`S'_b_las)
        do_ddml_interactive_one, outcome(`y') xset(base) mm(M_ridge)     suffix(`S'_b_rid)
        do_ddml_interactive_one, outcome(`y') xset(base) mm(M_elastic)   suffix(`S'_b_ela)
        do_ddml_interactive_one, outcome(`y') xset(base) mm(M_rf)        suffix(`S'_b_rf)
        do_ddml_interactive_one, outcome(`y') xset(base) mm(M_gradboost) suffix(`S'_b_gb)
        do_ddml_interactive_one, outcome(`y') xset(base) mm(M_best)      suffix(`S'_b_bst)

        do_ddml_interactive_one, outcome(`y') xset(ext) mm(M_logit)      suffix(`S'_e_logit)
        do_ddml_interactive_one, outcome(`y') xset(ext) mm(M_lasso)      suffix(`S'_e_las)
        do_ddml_interactive_one, outcome(`y') xset(ext) mm(M_ridge)      suffix(`S'_e_rid)
        do_ddml_interactive_one, outcome(`y') xset(ext) mm(M_elastic)    suffix(`S'_e_ela)
        do_ddml_interactive_one, outcome(`y') xset(ext) mm(M_rf)         suffix(`S'_e_rf)
        do_ddml_interactive_one, outcome(`y') xset(ext) mm(M_gradboost)  suffix(`S'_e_gb)
        do_ddml_interactive_one, outcome(`y') xset(ext) mm(M_best)       suffix(`S'_e_bst)

        restore
    }
}

*--------------------------------------------------------------------*
* 9. Tablas combinadas por outcome (Benchmark + RiskSource) – Interactive
*--------------------------------------------------------------------*

cap program drop make_panel_tables_two_int
program define make_panel_tables_two_int
    syntax , outcome(string)

    local yshort = cond("`outcome'"=="SomeRiskHome","SR","VH")

    if "`outcome'" == "SomeRiskHome" {
        local capB "Effect of Household Water Treatment on E.coli Risk (SomeRiskHome), interactive DDML, base controls"
        local capE "Effect of Household Water Treatment on E.coli Risk (SomeRiskHome), interactive DDML, extended controls"
        local outfileB "${Tables}SomeRiskHome_INTERACTIVE_BASE_RiskSource_All.tex"
        local outfileE "${Tables}SomeRiskHome_INTERACTIVE_EXT_RiskSource_All.tex"
    }
    else if "`outcome'" == "VeryHighRiskHome" {
        local capB "Effect of Household Water Treatment on Very High E.coli Risk (VeryHighRiskHome), interactive DDML, base controls"
        local capE "Effect of Household Water Treatment on Very High E.coli Risk (VeryHighRiskHome), interactive DDML, extended controls"
        local outfileB "${Tables}VeryHighRiskHome_INTERACTIVE_BASE_RiskSource_All.tex"
        local outfileE "${Tables}VeryHighRiskHome_INTERACTIVE_EXT_RiskSource_All.tex"
    }

    * BASE
    esttab OLS_`yshort'_b ///
           INT_`yshort'_b_logit ///
           INT_`yshort'_b_las ///
           INT_`yshort'_b_rid ///
           INT_`yshort'_b_ela ///
           INT_`yshort'_b_rf ///
           INT_`yshort'_b_gb ///
           INT_`yshort'_b_bst ///
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

    esttab OLS_`yshort'_r0_b ///
           INT_`yshort'_r0_b_logit ///
           INT_`yshort'_r0_b_las ///
           INT_`yshort'_r0_b_rid ///
           INT_`yshort'_r0_b_ela ///
           INT_`yshort'_r0_b_rf ///
           INT_`yshort'_r0_b_gb ///
           INT_`yshort'_r0_b_bst ///
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

    esttab OLS_`yshort'_r1_b ///
           INT_`yshort'_r1_b_logit ///
           INT_`yshort'_r1_b_las ///
           INT_`yshort'_r1_b_rid ///
           INT_`yshort'_r1_b_ela ///
           INT_`yshort'_r1_b_rf ///
           INT_`yshort'_r1_b_gb ///
           INT_`yshort'_r1_b_bst ///
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

    esttab OLS_`yshort'_r2_b ///
           INT_`yshort'_r2_b_logit ///
           INT_`yshort'_r2_b_las ///
           INT_`yshort'_r2_b_rid ///
           INT_`yshort'_r2_b_ela ///
           INT_`yshort'_r2_b_rf ///
           INT_`yshort'_r2_b_gb ///
           INT_`yshort'_r2_b_bst ///
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

    * EXTENDED
    esttab OLS_`yshort'_e ///
           INT_`yshort'_e_logit ///
           INT_`yshort'_e_las ///
           INT_`yshort'_e_rid ///
           INT_`yshort'_e_ela ///
           INT_`yshort'_e_rf ///
           INT_`yshort'_e_gb ///
           INT_`yshort'_e_bst ///
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

    esttab OLS_`yshort'_r0_e ///
           INT_`yshort'_r0_e_logit ///
           INT_`yshort'_r0_e_las ///
           INT_`yshort'_r0_e_rid ///
           INT_`yshort'_r0_e_ela ///
           INT_`yshort'_r0_e_rf ///
           INT_`yshort'_r0_e_gb ///
           INT_`yshort'_r0_e_bst ///
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

    esttab OLS_`yshort'_r1_e ///
           INT_`yshort'_r1_e_logit ///
           INT_`yshort'_r1_e_las ///
           INT_`yshort'_r1_e_rid ///
           INT_`yshort'_r1_e_ela ///
           INT_`yshort'_r1_e_rf ///
           INT_`yshort'_r1_e_gb ///
           INT_`yshort'_r1_e_bst ///
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

    esttab OLS_`yshort'_r2_e ///
           INT_`yshort'_r2_e_logit ///
           INT_`yshort'_r2_e_las ///
           INT_`yshort'_r2_e_rid ///
           INT_`yshort'_r2_e_ela ///
           INT_`yshort'_r2_e_rf ///
           INT_`yshort'_r2_e_gb ///
           INT_`yshort'_r2_e_bst ///
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

make_panel_tables_two_int, outcome(SomeRiskHome)
make_panel_tables_two_int, outcome(VeryHighRiskHome)

*--------------------------------------------------------------------*
* 10. MATRICES APILADAS DE PESOS Best – Benchmarks (INTERACTIVE)
*--------------------------------------------------------------------*

matrix WY_all_INT = ///
    WY_SR_b_bst \ ///
    WY_SR_e_bst \ ///
    WY_VH_b_bst \ ///
    WY_VH_e_bst

matrix WD_all_INT = ///
    WD_SR_b_bst \ ///
    WD_SR_e_bst \ ///
    WD_VH_b_bst \ ///
    WD_VH_e_bst

*--------------------------------------------------------------------*
* 11. Tabla combinada de pesos Best (Y y D) – Benchmarks (INTERACTIVE)
*--------------------------------------------------------------------*

local K     : word count $M_best
local ncols = `K' + 1
local outfile_INT "${Tables}stack_weights_Best_YD_INTERACTIVE.tex"

cap file close swI
file open swI using "`outfile_INT'", write replace

file write swI "\begin{table}[htbp]\centering" _n
file write swI "\caption{Stacking weights for Best (stacked) for \(E[Y \mid D,X]\) and \(E[D \mid X]\), interactive DDML}" _n

file write swI "\begin{tabular}{l"
forvalues j = 1/`K' {
    file write swI "c"
}
file write swI "}" _n

file write swI "\toprule" _n

* Panel A: E[Y|D,X]
local panelAI "\multicolumn{`ncols'}{l}{\textbf{Panel A: Outcome equation \(E[Y \mid D,X]\)}}\\"
file write swI "`panelAI'" _n
file write swI "\midrule" _n

file write swI "Specification"
forvalues j = 1/`K' {
    local m : word `j' of $M_best
    file write swI " & `m'"
}
file write swI " \\" _n
file write swI "\midrule" _n

file write swI "SomeRiskHome, base"
forvalues j = 1/`K' {
    local val  = WY_all_INT[1,`j']
    local sval : display %6.3f `val'
    file write swI " & `sval'"
}
file write swI " \\" _n

file write swI "SomeRiskHome, extended"
forvalues j = 1/`K' {
    local val  = WY_all_INT[2,`j']
    local sval : display %6.3f `val'
    file write swI " & `sval'"
}
file write swI " \\" _n

file write swI "VeryHighRiskHome, base"
forvalues j = 1/`K' {
    local val  = WY_all_INT[3,`j']
    local sval : display %6.3f `val'
    file write swI " & `sval'"
}
file write swI " \\" _n

file write swI "VeryHighRiskHome, extended"
forvalues j = 1/`K' {
    local val  = WY_all_INT[4,`j']
    local sval : display %6.3f `val'
    file write swI " & `sval'"
}
file write swI " \\" _n

file write swI "\midrule" _n

* Panel B: E[D|X]
local panelBI "\multicolumn{`ncols'}{l}{\textbf{Panel B: Treatment equation \(E[D \mid X]\)}}\\"
file write swI "`panelBI'" _n
file write swI "\midrule" _n

file write swI "Specification"
forvalues j = 1/`K' {
    local m : word `j' of $M_best
    file write swI " & `m'"
}
file write swI " \\" _n
file write swI "\midrule" _n

file write swI "SomeRiskHome, base"
forvalues j = 1/`K' {
    local val  = WD_all_INT[1,`j']
    local sval : display %6.3f `val'
    file write swI " & `sval'"
}
file write swI " \\" _n

file write swI "SomeRiskHome, extended"
forvalues j = 1/`K' {
    local val  = WD_all_INT[2,`j']
    local sval : display %6.3f `val'
    file write swI " & `sval'"
}
file write swI " \\" _n

file write swI "VeryHighRiskHome, base"
forvalues j = 1/`K' {
    local val  = WD_all_INT[3,`j']
    local sval : display %6.3f `val'
    file write swI " & `sval'"
}
file write swI " \\" _n

file write swI "VeryHighRiskHome, extended"
forvalues j = 1/`K' {
    local val  = WD_all_INT[4,`j']
    local sval : display %6.3f `val'
    file write swI " & `sval'"
}
file write swI " \\" _n

file write swI "\bottomrule" _n
file write swI "\end{tabular}" _n
file write swI "\end{table}" _n

file close swI

*--------------------------------------------------------------------*
* 12. MATRICES DE PESOS Best por RiskSource (BASE + EXT) – INTERACTIVE
*--------------------------------------------------------------------*

* BASE
matrix WY_SR_all_RS_INT = ///
    WY_SR_b_bst    \ ///
    WY_SR_r0_b_bst \ ///
    WY_SR_r1_b_bst \ ///
    WY_SR_r2_b_bst

matrix WD_SR_all_RS_INT = ///
    WD_SR_b_bst    \ ///
    WD_SR_r0_b_bst \ ///
    WD_SR_r1_b_bst \ ///
    WD_SR_r2_b_bst

matrix WY_VH_all_RS_INT = ///
    WY_VH_b_bst    \ ///
    WY_VH_r0_b_bst \ ///
    WY_VH_r1_b_bst \ ///
    WY_VH_r2_b_bst

matrix WD_VH_all_RS_INT = ///
    WD_VH_b_bst    \ ///
    WD_VH_r0_b_bst \ ///
    WD_VH_r1_b_bst \ ///
    WD_VH_r2_b_bst

* EXTENDED
matrix WY_SR_all_RS_ext_INT = ///
    WY_SR_e_bst    \ ///
    WY_SR_r0_e_bst \ ///
    WY_SR_r1_e_bst \ ///
    WY_SR_r2_e_bst

matrix WD_SR_all_RS_ext_INT = ///
    WD_SR_e_bst    \ ///
    WD_SR_r0_e_bst \ ///
    WD_SR_r1_e_bst \ ///
    WD_SR_r2_e_bst

matrix WY_VH_all_RS_ext_INT = ///
    WY_VH_e_bst    \ ///
    WY_VH_r0_e_bst \ ///
    WY_VH_r1_e_bst \ ///
    WY_VH_r2_e_bst

matrix WD_VH_all_RS_ext_INT = ///
    WD_VH_e_bst    \ ///
    WD_VH_r0_e_bst \ ///
    WD_VH_r1_e_bst \ ///
    WD_VH_r2_e_bst

*--------------------------------------------------------------------*
* 13. Tablas de pesos Best – SomeRiskHome por RiskSource (INTERACTIVE)
*--------------------------------------------------------------------*

local outfile_SRw_INT "${Tables}stack_weights_Best_YD_SomeRiskHome_RiskSource_INTERACTIVE.tex"

cap file close swSRI
file open swSRI using "`outfile_SRw_INT'", write replace

file write swSRI "\begin{table}[htbp]\centering" _n
file write swSRI "\caption{Stacking weights for Best (stacked) for \(E[Y \mid D,X]\) and \(E[D \mid X]\) – SomeRiskHome, by RiskSource, interactive DDML}" _n

file write swSRI "\begin{tabular}{l"
forvalues j = 1/`K' {
    file write swSRI "c"
}
file write swSRI "}" _n

file write swSRI "\toprule" _n

* Panel A: E[Y|D,X], base
local panelA2I "\multicolumn{`ncols'}{l}{\textbf{Panel A: Outcome equation \(E[Y \mid D,X]\), base controls}}\\"
file write swSRI "`panelA2I'" _n
file write swSRI "\midrule" _n

file write swSRI "Specification"
forvalues j = 1/`K' {
    local m : word `j' of $M_best
    file write swSRI " & `m'"
}
file write swSRI " \\" _n
file write swSRI "\midrule" _n

file write swSRI "SomeRiskHome, total"
forvalues j = 1/`K' {
    local val  = WY_SR_all_RS_INT[1,`j']
    local sval : display %6.3f `val'
    file write swSRI " & `sval'"
}
file write swSRI " \\" _n

file write swSRI "SomeRiskHome, RiskSource = 0"
forvalues j = 1/`K' {
    local val  = WY_SR_all_RS_INT[2,`j']
    local sval : display %6.3f `val'
    file write swSRI " & `sval'"
}
file write swSRI " \\" _n

file write swSRI "SomeRiskHome, RiskSource = 1"
forvalues j = 1/`K' {
    local val  = WY_SR_all_RS_INT[3,`j']
    local sval : display %6.3f `val'
    file write swSRI " & `sval'"
}
file write swSRI " \\" _n

file write swSRI "SomeRiskHome, RiskSource = 2"
forvalues j = 1/`K' {
    local val  = WY_SR_all_RS_INT[4,`j']
    local sval : display %6.3f `val'
    file write swSRI " & `sval'"
}
file write swSRI " \\" _n

file write swSRI "\midrule" _n

* Panel B: E[D|X], base
local panelB2I "\multicolumn{`ncols'}{l}{\textbf{Panel B: Treatment equation \(E[D \mid X]\), base controls}}\\"
file write swSRI "`panelB2I'" _n
file write swSRI "\midrule" _n

file write swSRI "Specification"
forvalues j = 1/`K' {
    local m : word `j' of $M_best
    file write swSRI " & `m'"
}
file write swSRI " \\" _n
file write swSRI "\midrule" _n

file write swSRI "SomeRiskHome, total"
forvalues j = 1/`K' {
    local val  = WD_SR_all_RS_INT[1,`j']
    local sval : display %6.3f `val'
    file write swSRI " & `sval'"
}
file write swSRI " \\" _n

file write swSRI "SomeRiskHome, RiskSource = 0"
forvalues j = 1/`K' {
    local val  = WD_SR_all_RS_INT[2,`j']
    local sval : display %6.3f `val'
    file write swSRI " & `sval'"
}
file write swSRI " \\" _n

file write swSRI "SomeRiskHome, RiskSource = 1"
forvalues j = 1/`K' {
    local val  = WD_SR_all_RS_INT[3,`j']
    local sval : display %6.3f `val'
    file write swSRI " & `sval'"
}
file write swSRI " \\" _n

file write swSRI "SomeRiskHome, RiskSource = 2"
forvalues j = 1/`K' {
    local val  = WD_SR_all_RS_INT[4,`j']
    local sval : display %6.3f `val'
    file write swSRI " & `sval'"
}
file write swSRI " \\" _n

file write swSRI "\midrule" _n

* Panel C: E[Y|D,X], extended
local panelC2I "\multicolumn{`ncols'}{l}{\textbf{Panel C: Outcome equation \(E[Y \mid D,X]\), extended controls}}\\"
file write swSRI "`panelC2I'" _n
file write swSRI "\midrule" _n

file write swSRI "Specification"
forvalues j = 1/`K' {
    local m : word `j' of $M_best
    file write swSRI " & `m'"
}
file write swSRI " \\" _n
file write swSRI "\midrule" _n

file write swSRI "SomeRiskHome, total"
forvalues j = 1/`K' {
    local val  = WY_SR_all_RS_ext_INT[1,`j']
    local sval : display %6.3f `val'
    file write swSRI " & `sval'"
}
file write swSRI " \\" _n

file write swSRI "SomeRiskHome, RiskSource = 0"
forvalues j = 1/`K' {
    local val  = WY_SR_all_RS_ext_INT[2,`j']
    local sval : display %6.3f `val'
    file write swSRI " & `sval'"
}
file write swSRI " \\" _n

file write swSRI "SomeRiskHome, RiskSource = 1"
forvalues j = 1/`K' {
    local val  = WY_SR_all_RS_ext_INT[3,`j']
    local sval : display %6.3f `val'
    file write swSRI " & `sval'"
}
file write swSRI " \\" _n

file write swSRI "SomeRiskHome, RiskSource = 2"
forvalues j = 1/`K' {
    local val  = WY_SR_all_RS_ext_INT[4,`j']
    local sval : display %6.3f `val'
    file write swSRI " & `sval'"
}
file write swSRI " \\" _n

file write swSRI "\midrule" _n

* Panel D: E[D|X], extended
local panelD2I "\multicolumn{`ncols'}{l}{\textbf{Panel D: Treatment equation \(E[D \mid X]\), extended controls}}\\"
file write swSRI "`panelD2I'" _n
file write swSRI "\midrule" _n

file write swSRI "Specification"
forvalues j = 1/`K' {
    local m : word `j' of $M_best
    file write swSRI " & `m'"
}
file write swSRI " \\" _n
file write swSRI "\midrule" _n

file write swSRI "SomeRiskHome, total"
forvalues j = 1/`K' {
    local val  = WD_SR_all_RS_ext_INT[1,`j']
    local sval : display %6.3f `val'
    file write swSRI " & `sval'"
}
file write swSRI " \\" _n

file write swSRI "SomeRiskHome, RiskSource = 0"
forvalues j = 1/`K' {
    local val  = WD_SR_all_RS_ext_INT[2,`j']
    local sval : display %6.3f `val'
    file write swSRI " & `sval'"
}
file write swSRI " \\" _n

file write swSRI "SomeRiskHome, RiskSource = 1"
forvalues j = 1/`K' {
    local val  = WD_SR_all_RS_ext_INT[3,`j']
    local sval : display %6.3f `val'
    file write swSRI " & `sval'"
}
file write swSRI " \\" _n

file write swSRI "SomeRiskHome, RiskSource = 2"
forvalues j = 1/`K' {
    local val  = WD_SR_all_RS_ext_INT[4,`j']
    local sval : display %6.3f `val'
    file write swSRI " & `sval'"
}
file write swSRI " \\" _n

file write swSRI "\bottomrule" _n
file write swSRI "\end{tabular}" _n
file write swSRI "\end{table}" _n

file close swSRI

*--------------------------------------------------------------------*
* 14. Tablas de pesos Best – VeryHighRiskHome por RiskSource (INTERACTIVE)
*--------------------------------------------------------------------*

local outfile_VHw_INT "${Tables}stack_weights_Best_YD_VeryHighRiskHome_RiskSource_INTERACTIVE.tex"

cap file close swVHI
file open swVHI using "`outfile_VHw_INT'", write replace

file write swVHI "\begin{table}[htbp]\centering" _n
file write swVHI "\caption{Stacking weights for Best (stacked) for \(E[Y \mid D,X]\) and \(E[D \mid X]\) – VeryHighRiskHome, by RiskSource, interactive DDML}" _n

file write swVHI "\begin{tabular}{l"
forvalues j = 1/`K' {
    file write swVHI "c"
}
file write swVHI "}" _n

file write swVHI "\toprule" _n

* Panel A: E[Y|D,X], base
local panelA_vhI "\multicolumn{`ncols'}{l}{\textbf{Panel A: Outcome equation \(E[Y \mid D,X]\), base controls}}\\"
file write swVHI "`panelA_vhI'" _n
file write swVHI "\midrule" _n

file write swVHI "Specification"
forvalues j = 1/`K' {
    local m : word `j' of $M_best
    file write swVHI " & `m'"
}
file write swVHI " \\" _n
file write swVHI "\midrule" _n

file write swVHI "VeryHighRiskHome, total"
forvalues j = 1/`K' {
    local val  = WY_VH_all_RS_INT[1,`j']
    local sval : display %6.3f `val'
    file write swVHI " & `sval'"
}
file write swVHI " \\" _n

file write swVHI "VeryHighRiskHome, RiskSource = 0"
forvalues j = 1/`K' {
    local val  = WY_VH_all_RS_INT[2,`j']
    local sval : display %6.3f `val'
    file write swVHI " & `sval'"
}
file write swVHI " \\" _n

file write swVHI "VeryHighRiskHome, RiskSource = 1"
forvalues j = 1/`K' {
    local val  = WY_VH_all_RS_INT[3,`j']
    local sval : display %6.3f `val'
    file write swVHI " & `sval'"
}
file write swVHI " \\" _n

file write swVHI "VeryHighRiskHome, RiskSource = 2"
forvalues j = 1/`K' {
    local val  = WY_VH_all_RS_INT[4,`j']
    local sval : display %6.3f `val'
    file write swVHI " & `sval'"
}
file write swVHI " \\" _n

file write swVHI "\midrule" _n

* Panel B: E[D|X], base
local panelB_vhI "\multicolumn{`ncols'}{l}{\textbf{Panel B: Treatment equation \(E[D \mid X]\), base controls}}\\"
file write swVHI "`panelB_vhI'" _n
file write swVHI "\midrule" _n

file write swVHI "Specification"
forvalues j = 1/`K' {
    local m : word `j' of $M_best
    file write swVHI " & `m'"
}
file write swVHI " \\" _n
file write swVHI "\midrule" _n

file write swVHI "VeryHighRiskHome, total"
forvalues j = 1/`K' {
    local val  = WD_VH_all_RS_INT[1,`j']
    local sval : display %6.3f `val'
    file write swVHI " & `sval'"
}
file write swVHI " \\" _n

file write swVHI "VeryHighRiskHome, RiskSource = 0"
forvalues j = 1/`K' {
    local val  = WD_VH_all_RS_INT[2,`j']
    local sval : display %6.3f `val'
    file write swVHI " & `sval'"
}
file write swVHI " \\" _n

file write swVHI "VeryHighRiskHome, RiskSource = 1"
forvalues j = 1/`K' {
    local val  = WD_VH_all_RS_INT[3,`j']
    local sval : display %6.3f `val'
    file write swVHI " & `sval'"
}
file write swVHI " \\" _n

file write swVHI "VeryHighRiskHome, RiskSource = 2"
forvalues j = 1/`K' {
    local val  = WD_VH_all_RS_INT[4,`j']
    local sval : display %6.3f `val'
    file write swVHI " & `sval'"
}
file write swVHI " \\" _n

file write swVHI "\midrule" _n

* Panel C: E[Y|D,X], extended
local panelC_vhI "\multicolumn{`ncols'}{l}{\textbf{Panel C: Outcome equation \(E[Y \mid D,X]\), extended controls}}\\"
file write swVHI "`panelC_vhI'" _n
file write swVHI "\midrule" _n

file write swVHI "Specification"
forvalues j = 1/`K' {
    local m : word `j' of $M_best
    file write swVHI " & `m'"
}
file write swVHI " \\" _n
file write swVHI "\midrule" _n

file write swVHI "VeryHighRiskHome, total"
forvalues j = 1/`K' {
    local val  = WY_VH_all_RS_ext_INT[1,`j']
    local sval : display %6.3f `val'
    file write swVHI " & `sval'"
}
file write swVHI " \\" _n

file write swVHI "VeryHighRiskHome, RiskSource = 0"
forvalues j = 1/`K' {
    local val  = WY_VH_all_RS_ext_INT[2,`j']
    local sval : display %6.3f `val'
    file write swVHI " & `sval'"
}
file write swVHI " \\" _n

file write swVHI "VeryHighRiskHome, RiskSource = 1"
forvalues j = 1/`K' {
    local val  = WY_VH_all_RS_ext_INT[3,`j']
    local sval : display %6.3f `val'
    file write swVHI " & `sval'"
}
file write swVHI " \\" _n

file write swVHI "VeryHighRiskHome, RiskSource = 2"
forvalues j = 1/`K' {
    local val  = WY_VH_all_RS_ext_INT[4,`j']
    local sval : display %6.3f `val'
    file write swVHI " & `sval'"
}
file write swVHI " \\" _n

file write swVHI "\midrule" _n

* Panel D: E[D|X], extended
local panelD_vhI "\multicolumn{`ncols'}{l}{\textbf{Panel D: Treatment equation \(E[D \mid X]\), extended controls}}\\"
file write swVHI "`panelD_vhI'" _n
file write swVHI "\midrule" _n

file write swVHI "Specification"
forvalues j = 1/`K' {
    local m : word `j' of $M_best
    file write swVHI " & `m'"
}
file write swVHI " \\" _n
file write swVHI "\midrule" _n

file write swVHI "VeryHighRiskHome, total"
forvalues j = 1/`K' {
    local val  = WD_VH_all_RS_ext_INT[1,`j']
    local sval : display %6.3f `val'
    file write swVHI " & `sval'"
}
file write swVHI " \\" _n

file write swVHI "VeryHighRiskHome, RiskSource = 0"
forvalues j = 1/`K' {
    local val  = WD_VH_all_RS_ext_INT[2,`j']
    local sval : display %6.3f `val'
    file write swVHI " & `sval'"
}
file write swVHI " \\" _n

file write swVHI "VeryHighRiskHome, RiskSource = 1"
forvalues j = 1/`K' {
    local val  = WD_VH_all_RS_ext_INT[3,`j']
    local sval : display %6.3f `val'
    file write swVHI " & `sval'"
}
file write swVHI " \\" _n

file write swVHI "VeryHighRiskHome, RiskSource = 2"
forvalues j = 1/`K' {
    local val  = WD_VH_all_RS_ext_INT[4,`j']
    local sval : display %6.3f `val'
    file write swVHI " & `sval'"
}
file write swVHI " \\" _n

file write swVHI "\bottomrule" _n
file write swVHI "\end{tabular}" _n
file write swVHI "\end{table}" _n

file close swVHI
