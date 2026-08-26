{smcl}
{com}{sf}{ul off}{txt}{.-}
      name:  {res}<unnamed>
       {txt}log:  {res}/Users/akitokamei/Dropbox/Apps/Overleaf/MICS_Water/Table/MICS_WaterRandom forest.md
  {txt}log type:  {res}smcl
 {txt}opened on:  {res} 5 Nov 2024, 09:30:59
{txt}
{com}. 
. * global outcome VeryHighRiskHome
. global outcome WQ26
{txt}
{com}. 
. ********************************************************************************
. **# Configure models
. ********************************************************************************
. * use  "${c -(}Data{c )-}Cleaned_Pooled_MICS6_Africa_2.dta", clear
. use "${c -(}Data{c )-}MASTER_MICS_RF.dta", clear
{txt}
{com}. 
. gen id=_n
{txt}
{com}. set seed 1381261
{txt}
{com}. gen Random=runiform(0,1)
{txt}
{com}. gen split=Random
{txt}
{com}. recode split 0/0.5=1 0.5/1=2
{txt}(62,179 changes made to {bf:split})

{com}. tab split, m

      {txt}split {c |}      Freq.     Percent        Cum.
{hline 12}{c +}{hline 35}
          1 {c |}{res}     31,173       50.13       50.13
{txt}          2 {c |}{res}     31,006       49.87      100.00
{txt}{hline 12}{c +}{hline 35}
      Total {c |}{res}     62,179      100.00
{txt}
{com}. 
. * recode        
. gen G_WS1=WS1
{txt}(4 missing values generated)

{com}. recode G_WS1 11 12 13 14=11 31 32=31 41 42=41 92=91  51 61 62 71 72=96
{txt}(21,785 changes made to {bf:G_WS1})

{com}. 
.         label define G_WS1l 11 "WS: Piped Water" 21 "WS: Tube well/borehole"  31 "WS: Dug well" 41 "WS: Spring" 81 "WS: Surface Water" 91 "WS: Packaged water" 96 "WS: Other", modify
{txt}
{com}.         label values G_WS1 G_WS1l
{txt}
{com}. 
. gen G_WS10=WS10
{txt}
{com}. recode G_WS10 6=98
{txt}(626 changes made to {bf:G_WS10})

{com}.         label define G_WS10l 0 "Treat: Nothing" 1 "Treat: Boil" 2 "Treat: Bleach/Chlorine" 3 "Treat: Stain with a cloth" 4 "Treat: Filter" 5 "Treat: Soler" 7 "Treat: Aquatabs/PUR" 8 "Treat: Add tablet" 98 "Treat: Other" 99 "Treat: Do not know/missing", modify
{txt}
{com}.         label values G_WS10 G_WS10l
{txt}
{com}.         
.         * WS1_11 WS1_12 WS1_13 WS1_14 WS1_21 WS1_31 WS1_32 WS1_41 WS1_42 WS1_51 WS1_61 WS1_62 WS1_71 WS1_72 WS1_81 WS1_91 WS1_92 WS1_96
.         * WS10_0 WS10_1 WS10_2 WS10_3 WS10_6 WS10_7 WS10_98 WS10_99
.         
. * Variable construction
. global V_source   G_WS1_11 G_WS1_21 G_WS1_31 G_WS1_41 G_WS1_81 G_WS1_91 G_WS1_96
{txt}
{com}. global V_treat    G_WS10_0 G_WS10_1 G_WS10_2 G_WS10_3 G_WS10_7 G_WS10_98 G_WS10_99
{txt}
{com}. global V_country  country_cat_2 country_cat_3 country_cat_4 country_cat_5 country_cat_8 country_cat_9 country_cat_10 country_cat_11 country_cat_12 country_cat_13 country_cat_14 country_cat_15 country_cat_16 country_cat_17 country_cat_19 country_cat_20 country_cat_21 country_cat_22 country_cat_23 country_cat_24 country_cat_25 country_cat_26 country_cat_27 country_cat_29 country_cat_32 country_cat_33
{txt}
{com}. global V_simple   $V_source $V_treat $V_country  urban Open_defecation
{txt}
{com}. 
.         * Create Dummy:  WS1,  WS10,  WS3
.         foreach v in G_WS1 G_WS10 country_cat {c -(}
{txt}  2{com}.         levelsof `v'
{txt}  3{com}.         foreach value in `r(levels)' {c -(}
{txt}  4{com}.                 gen     `v'_`value'=0
{txt}  5{com}.                 replace `v'_`value'=1 if `v'==`value'
{txt}  6{com}.                 replace `v'_`value'=. if `v'==.
{txt}  7{com}.                 label var `v'_`value' "`: label (`v') `value''"
{txt}  8{com}.         {c )-}
{txt}  9{com}.         {c )-}
{res}{txt}11 21 31 41 81 91 96
(18,228 real changes made)
(4 real changes made, 4 to missing)
(14,864 real changes made)
(4 real changes made, 4 to missing)
(7,419 real changes made)
(4 real changes made, 4 to missing)
(4,935 real changes made)
(4 real changes made, 4 to missing)
(3,183 real changes made)
(4 real changes made, 4 to missing)
(10,657 real changes made)
(4 real changes made, 4 to missing)
(2,889 real changes made)
(4 real changes made, 4 to missing)
{res}{txt}0 1 2 3 7 98 99
(47,072 real changes made)
(0 real changes made)
(5,322 real changes made)
(0 real changes made)
(2,156 real changes made)
(0 real changes made)
(2,457 real changes made)
(0 real changes made)
(2,070 real changes made)
(0 real changes made)
(3,080 real changes made)
(0 real changes made)
(22 real changes made)
(0 real changes made)
{res}{txt}2 3 4 5 8 9 10 11 12 13 14 15 16 17 19 20 21 22 23 24 25 26 27 29 32 33
(6,051 real changes made)
(0 real changes made)
(3,649 real changes made)
(0 real changes made)
(1,034 real changes made)
(0 real changes made)
(2,111 real changes made)
(0 real changes made)
(2,704 real changes made)
(0 real changes made)
(2,538 real changes made)
(0 real changes made)
(1,154 real changes made)
(0 real changes made)
(1,084 real changes made)
(0 real changes made)
(1,749 real changes made)
(0 real changes made)
(3,134 real changes made)
(0 real changes made)
(1,821 real changes made)
(0 real changes made)
(1,395 real changes made)
(0 real changes made)
(4,020 real changes made)
(0 real changes made)
(2,538 real changes made)
(0 real changes made)
(3,247 real changes made)
(0 real changes made)
(1,331 real changes made)
(0 real changes made)
(3,264 real changes made)
(0 real changes made)
(3,119 real changes made)
(0 real changes made)
(2,576 real changes made)
(0 real changes made)
(2,392 real changes made)
(0 real changes made)
(1,739 real changes made)
(0 real changes made)
(1,617 real changes made)
(0 real changes made)
(1,084 real changes made)
(0 real changes made)
(1,603 real changes made)
(0 real changes made)
(3,232 real changes made)
(0 real changes made)
(1,993 real changes made)
(0 real changes made)

{com}. 
. /*
> tab RiskHome RiskSource,m
> hexplot RiskSource  RiskHome, values(format(%9.1f)) aspectratio(1) legend(off) ///
>                               color(HCL reds, intensity(.6) reverse ) p(lc(black) lalign(center)) bins(5) ///
>                                                           xlabel(0 "No risk" 1 "Moderate risk" 2 "Very high risk") xtitle("Point of use") ///
>                                                           ylabel(0 "No risk" 1 "Moderate risk" 2 "Very high risk") ytitle("Source") ///
>                                                           sizeprop
> graph export "${c -(}Figure{c )-}TabSourceHome.eps", replace
> */
. 
. save           "${c -(}Data{c )-}MASTER_MICS_RF_Home.dta", replace
{txt}{p 0 4 2}
file {bf}
/Users/akitokamei/Box Sync/MICS Water project/Data/MASTER_MICS_RF_Home.dta{rm}
saved
{p_end}

{com}. savesome using "${c -(}Data{c )-}MASTER_MICS_RF_Home0.dta" if RiskSource==0, replace
{txt}{p 0 4 2}
file {bf}
/Users/akitokamei/Box Sync/MICS Water project/Data/MASTER_MICS_RF_Home0.dta{rm}
saved
{p_end}

{com}. savesome using "${c -(}Data{c )-}MASTER_MICS_RF_Home1.dta" if RiskSource==1, replace
{txt}{p 0 4 2}
file {bf}
/Users/akitokamei/Box Sync/MICS Water project/Data/MASTER_MICS_RF_Home1.dta{rm}
saved
{p_end}

{com}. 
. * graph bar VeryHighRiskHome, over(G_WS10, label(angle(45)))
. * graph bar VeryHighRiskHome, over(G_WS1)
. 
. global RFcontrols G_WS1_11 G_WS1_21 G_WS1_31 G_WS1_41 G_WS1_81 G_WS1_91 G_WS1_96 ///
>                   water_treatment ///
>                                   G_WS10_0 G_WS10_1 G_WS10_2 G_WS10_3 G_WS10_7 G_WS10_98 G_WS10_99 ///
>                                   WS3_1 WS3_2 WS3_3 ///
>                                   urban Open_defecation ///
>                                   
{txt}
{com}.                                 local Main "Desciptive statistics by the level of source water contamination"
{txt}
{com}. local LabelMain "Desc1"
{txt}
{com}. local noteMain "Notes: WQ29: Ask Jeremy how to control this. Clean Primary Water Source wit Sujey. Discuss the variable after location. Clean more and decide what to include"
{txt}
{com}.                                          
. foreach k in RFcontrols {c -(}
{txt}  2{com}. * Mean
.         eststo  model0: estpost summarize $`k' if RiskHome==0
{txt}  3{com}.         eststo  model1: estpost summarize $`k' if RiskHome==1
{txt}  4{com}.         eststo  model2: estpost summarize $`k' if RiskHome==2
{txt}  5{com}. 
. esttab model0 model1 model2 using "${c -(}Table{c )-}Descript_`k'_Risk.tex", title("``k''" \label{c -(}`Label`k''{c )-}) ///
>            cell("mean (fmt(2) label(_))") stats(N, fmt("%9.0fc") label(Observations) ) /// 
>            mtitles("No risk" "Moderate risk" "High risk") nonum ///
>            substitute( ".00" "" "{c -(}l{c )-}{c -(}\footnotesize" "{c -(}p{c -(}0.87\linewidth{c )-}{c )-}{c -(}\footnotesize" ///
>                                    "&           _&           _&           _&           _&           _\\" "" ///
>                                    "Piped water (Dwelling)" "\textbf{c -(}Primary Water Source{c )-} \\\hline Piped water (Dwelling)" ///
>                                    "Location: In own dwelling" "\textbf{c -(}Location{c )-} \\\hline Location: In own dwelling" ///
>                    "Any water treatment for tested" "\textbf{c -(}Water that is tested{c )-} \\\hline Any water treatment" ///
>                                    "Any water treatment for primary" "\textbf{c -(}Primary Water Source{c )-} \\\hline Any water treatment" ///
>                                    "-0 " "0" ///
>                                    "Treat:"  "~~~" "Location:"  "~~~" ///
>                                    ) ///
>            label  note("`note`k''")  ///
>            replace 
{txt}  6{com}.            {c )-}

{txt}{space 0}{space 0}{ralign 12:}{space 1}{c |}{space 1}{ralign 9:e(count)}{space 1}{space 1}{ralign 9:e(sum_w)}{space 1}{space 1}{ralign 9:e(mean)}{space 1}{space 1}{ralign 9:e(Var)}{space 1}{space 1}{ralign 9:e(sd)}{space 1}{space 1}{ralign 9:e(min)}{space 1}{space 1}{ralign 9:e(max)}{space 1}{space 1}{ralign 9:e(sum)}{space 1}
{space 0}{hline 13}{c   +}{hline 11}{hline 11}{hline 11}{hline 11}{hline 11}{hline 11}{hline 11}{hline 11}
{space 0}{space 0}{ralign 12:G_WS1_11}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    16280}}}{space 1}{space 1}{ralign 9:{res:{sf:    16280}}}{space 1}{space 1}{ralign 9:{res:{sf: .3980958}}}{space 1}{space 1}{ralign 9:{res:{sf: .2396303}}}{space 1}{space 1}{ralign 9:{res:{sf: .4895204}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     6481}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS1_21}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    16280}}}{space 1}{space 1}{ralign 9:{res:{sf:    16280}}}{space 1}{space 1}{ralign 9:{res:{sf: .1535627}}}{space 1}{space 1}{ralign 9:{res:{sf: .1299891}}}{space 1}{space 1}{ralign 9:{res:{sf: .3605401}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     2500}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS1_31}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    16280}}}{space 1}{space 1}{ralign 9:{res:{sf:    16280}}}{space 1}{space 1}{ralign 9:{res:{sf: .0447174}}}{space 1}{space 1}{ralign 9:{res:{sf: .0427204}}}{space 1}{space 1}{ralign 9:{res:{sf: .2066892}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:      728}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS1_41}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    16280}}}{space 1}{space 1}{ralign 9:{res:{sf:    16280}}}{space 1}{space 1}{ralign 9:{res:{sf: .0535627}}}{space 1}{space 1}{ralign 9:{res:{sf: .0506968}}}{space 1}{space 1}{ralign 9:{res:{sf: .2251595}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:      872}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS1_81}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    16280}}}{space 1}{space 1}{ralign 9:{res:{sf:    16280}}}{space 1}{space 1}{ralign 9:{res:{sf: .0184275}}}{space 1}{space 1}{ralign 9:{res:{sf: .0180891}}}{space 1}{space 1}{ralign 9:{res:{sf: .1344956}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:      300}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS1_91}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    16280}}}{space 1}{space 1}{ralign 9:{res:{sf:    16280}}}{space 1}{space 1}{ralign 9:{res:{sf: .2705774}}}{space 1}{space 1}{ralign 9:{res:{sf: .1973774}}}{space 1}{space 1}{ralign 9:{res:{sf: .4442718}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     4405}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS1_96}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    16280}}}{space 1}{space 1}{ralign 9:{res:{sf:    16280}}}{space 1}{space 1}{ralign 9:{res:{sf: .0610565}}}{space 1}{space 1}{ralign 9:{res:{sf: .0573321}}}{space 1}{space 1}{ralign 9:{res:{sf: .2394413}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:      994}}}{space 1}
{space 0}{space 0}{ralign 12:water_trea~t}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    16258}}}{space 1}{space 1}{ralign 9:{res:{sf:    16258}}}{space 1}{space 1}{ralign 9:{res:{sf: .3252553}}}{space 1}{space 1}{ralign 9:{res:{sf: .2194778}}}{space 1}{space 1}{ralign 9:{res:{sf: .4684846}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     5288}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS10_0}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    16282}}}{space 1}{space 1}{ralign 9:{res:{sf:    16282}}}{space 1}{space 1}{ralign 9:{res:{sf:  .740511}}}{space 1}{space 1}{ralign 9:{res:{sf: .1921663}}}{space 1}{space 1}{ralign 9:{res:{sf: .4383677}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:    12057}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS10_1}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    16282}}}{space 1}{space 1}{ralign 9:{res:{sf:    16282}}}{space 1}{space 1}{ralign 9:{res:{sf: .1212382}}}{space 1}{space 1}{ralign 9:{res:{sf:  .106546}}}{space 1}{space 1}{ralign 9:{res:{sf: .3264139}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     1974}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS10_2}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    16282}}}{space 1}{space 1}{ralign 9:{res:{sf:    16282}}}{space 1}{space 1}{ralign 9:{res:{sf: .0249969}}}{space 1}{space 1}{ralign 9:{res:{sf: .0243736}}}{space 1}{space 1}{ralign 9:{res:{sf: .1561204}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:      407}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS10_3}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    16282}}}{space 1}{space 1}{ralign 9:{res:{sf:    16282}}}{space 1}{space 1}{ralign 9:{res:{sf: .0207591}}}{space 1}{space 1}{ralign 9:{res:{sf: .0203294}}}{space 1}{space 1}{ralign 9:{res:{sf: .1425813}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:      338}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS10_7}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    16282}}}{space 1}{space 1}{ralign 9:{res:{sf:    16282}}}{space 1}{space 1}{ralign 9:{res:{sf: .0130819}}}{space 1}{space 1}{ralign 9:{res:{sf: .0129116}}}{space 1}{space 1}{ralign 9:{res:{sf: .1136292}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:      213}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS10_98}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    16282}}}{space 1}{space 1}{ralign 9:{res:{sf:    16282}}}{space 1}{space 1}{ralign 9:{res:{sf:   .07929}}}{space 1}{space 1}{ralign 9:{res:{sf: .0730076}}}{space 1}{space 1}{ralign 9:{res:{sf: .2701992}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     1291}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS10_99}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    16282}}}{space 1}{space 1}{ralign 9:{res:{sf:    16282}}}{space 1}{space 1}{ralign 9:{res:{sf: .0001228}}}{space 1}{space 1}{ralign 9:{res:{sf: .0001228}}}{space 1}{space 1}{ralign 9:{res:{sf: .0110828}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:        2}}}{space 1}
{space 0}{space 0}{ralign 12:WS3_1}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    16282}}}{space 1}{space 1}{ralign 9:{res:{sf:    16282}}}{space 1}{space 1}{ralign 9:{res:{sf: .0254269}}}{space 1}{space 1}{ralign 9:{res:{sf: .0247818}}}{space 1}{space 1}{ralign 9:{res:{sf: .1574225}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:      414}}}{space 1}
{space 0}{space 0}{ralign 12:WS3_2}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    16282}}}{space 1}{space 1}{ralign 9:{res:{sf:    16282}}}{space 1}{space 1}{ralign 9:{res:{sf: .1545265}}}{space 1}{space 1}{ralign 9:{res:{sf: .1306561}}}{space 1}{space 1}{ralign 9:{res:{sf: .3614638}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     2516}}}{space 1}
{space 0}{space 0}{ralign 12:WS3_3}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    16282}}}{space 1}{space 1}{ralign 9:{res:{sf:    16282}}}{space 1}{space 1}{ralign 9:{res:{sf: .2436433}}}{space 1}{space 1}{ralign 9:{res:{sf: .1842926}}}{space 1}{space 1}{ralign 9:{res:{sf: .4292931}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     3967}}}{space 1}
{space 0}{space 0}{ralign 12:urban}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    16282}}}{space 1}{space 1}{ralign 9:{res:{sf:    16282}}}{space 1}{space 1}{ralign 9:{res:{sf: .5527576}}}{space 1}{space 1}{ralign 9:{res:{sf: .2472318}}}{space 1}{space 1}{ralign 9:{res:{sf: .4972241}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     9000}}}{space 1}
{space 0}{space 0}{ralign 12:Open_defec~n}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    16123}}}{space 1}{space 1}{ralign 9:{res:{sf:    16123}}}{space 1}{space 1}{ralign 9:{res:{sf: .0545184}}}{space 1}{space 1}{ralign 9:{res:{sf: .0515493}}}{space 1}{space 1}{ralign 9:{res:{sf: .2270448}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:      879}}}{space 1}

{space 0}{space 0}{ralign 12:}{space 1}{c |}{space 1}{ralign 9:e(count)}{space 1}{space 1}{ralign 9:e(sum_w)}{space 1}{space 1}{ralign 9:e(mean)}{space 1}{space 1}{ralign 9:e(Var)}{space 1}{space 1}{ralign 9:e(sd)}{space 1}{space 1}{ralign 9:e(min)}{space 1}{space 1}{ralign 9:e(max)}{space 1}{space 1}{ralign 9:e(sum)}{space 1}
{space 0}{hline 13}{c   +}{hline 11}{hline 11}{hline 11}{hline 11}{hline 11}{hline 11}{hline 11}{hline 11}
{space 0}{space 0}{ralign 12:G_WS1_11}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf: .3012618}}}{space 1}{space 1}{ralign 9:{res:{sf:  .210511}}}{space 1}{space 1}{ralign 9:{res:{sf: .4588147}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     8094}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS1_21}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf: .2608032}}}{space 1}{space 1}{ralign 9:{res:{sf: .1927921}}}{space 1}{space 1}{ralign 9:{res:{sf: .4390809}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     7007}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS1_31}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf: .1041054}}}{space 1}{space 1}{ralign 9:{res:{sf: .0932709}}}{space 1}{space 1}{ralign 9:{res:{sf: .3054029}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     2797}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS1_41}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf: .0889195}}}{space 1}{space 1}{ralign 9:{res:{sf: .0810158}}}{space 1}{space 1}{ralign 9:{res:{sf: .2846328}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     2389}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS1_81}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf: .0347266}}}{space 1}{space 1}{ralign 9:{res:{sf: .0335219}}}{space 1}{space 1}{ralign 9:{res:{sf: .1830899}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:      933}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS1_91}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf: .1670823}}}{space 1}{space 1}{ralign 9:{res:{sf:  .139171}}}{space 1}{space 1}{ralign 9:{res:{sf: .3730563}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     4489}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS1_96}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf: .0431012}}}{space 1}{space 1}{ralign 9:{res:{sf:  .041245}}}{space 1}{space 1}{ralign 9:{res:{sf: .2030887}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     1158}}}{space 1}
{space 0}{space 0}{ralign 12:water_trea~t}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    26845}}}{space 1}{space 1}{ralign 9:{res:{sf:    26845}}}{space 1}{space 1}{ralign 9:{res:{sf: .1871484}}}{space 1}{space 1}{ralign 9:{res:{sf: .1521296}}}{space 1}{space 1}{ralign 9:{res:{sf: .3900379}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     5024}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS10_0}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf: .7622362}}}{space 1}{space 1}{ralign 9:{res:{sf: .1812389}}}{space 1}{space 1}{ralign 9:{res:{sf: .4257216}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:    20479}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS10_1}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf: .0745152}}}{space 1}{space 1}{ralign 9:{res:{sf: .0689653}}}{space 1}{space 1}{ralign 9:{res:{sf: .2626124}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     2002}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS10_2}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf: .0347638}}}{space 1}{space 1}{ralign 9:{res:{sf: .0335566}}}{space 1}{space 1}{ralign 9:{res:{sf: .1831845}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:      934}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS10_3}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf: .0525552}}}{space 1}{space 1}{ralign 9:{res:{sf:  .049795}}}{space 1}{space 1}{ralign 9:{res:{sf: .2231479}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     1412}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS10_7}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf: .0300368}}}{space 1}{space 1}{ralign 9:{res:{sf: .0291357}}}{space 1}{space 1}{ralign 9:{res:{sf: .1706919}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:      807}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS10_98}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf: .0456322}}}{space 1}{space 1}{ralign 9:{res:{sf: .0435515}}}{space 1}{space 1}{ralign 9:{res:{sf:   .20869}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     1226}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS10_99}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf: .0002605}}}{space 1}{space 1}{ralign 9:{res:{sf: .0002605}}}{space 1}{space 1}{ralign 9:{res:{sf: .0161395}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:        7}}}{space 1}
{space 0}{space 0}{ralign 12:WS3_1}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf: .0212156}}}{space 1}{space 1}{ralign 9:{res:{sf: .0207663}}}{space 1}{space 1}{ralign 9:{res:{sf: .1441051}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:      570}}}{space 1}
{space 0}{space 0}{ralign 12:WS3_2}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf: .2160271}}}{space 1}{space 1}{ralign 9:{res:{sf: .1693657}}}{space 1}{space 1}{ralign 9:{res:{sf: .4115406}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     5804}}}{space 1}
{space 0}{space 0}{ralign 12:WS3_3}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf: .4490267}}}{space 1}{space 1}{ralign 9:{res:{sf: .2474109}}}{space 1}{space 1}{ralign 9:{res:{sf: .4974042}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:    12064}}}{space 1}
{space 0}{space 0}{ralign 12:urban}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf:    26867}}}{space 1}{space 1}{ralign 9:{res:{sf: .3685934}}}{space 1}{space 1}{ralign 9:{res:{sf:  .232741}}}{space 1}{space 1}{ralign 9:{res:{sf: .4824324}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     9903}}}{space 1}
{space 0}{space 0}{ralign 12:Open_defec~n}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    26407}}}{space 1}{space 1}{ralign 9:{res:{sf:    26407}}}{space 1}{space 1}{ralign 9:{res:{sf: .1465142}}}{space 1}{space 1}{ralign 9:{res:{sf: .1250525}}}{space 1}{space 1}{ralign 9:{res:{sf: .3536276}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     3869}}}{space 1}

{space 0}{space 0}{ralign 12:}{space 1}{c |}{space 1}{ralign 9:e(count)}{space 1}{space 1}{ralign 9:e(sum_w)}{space 1}{space 1}{ralign 9:e(mean)}{space 1}{space 1}{ralign 9:e(Var)}{space 1}{space 1}{ralign 9:e(sd)}{space 1}{space 1}{ralign 9:e(min)}{space 1}{space 1}{ralign 9:e(max)}{space 1}{space 1}{ralign 9:e(sum)}{space 1}
{space 0}{hline 13}{c   +}{hline 11}{hline 11}{hline 11}{hline 11}{hline 11}{hline 11}{hline 11}{hline 11}
{space 0}{space 0}{ralign 12:G_WS1_11}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    19028}}}{space 1}{space 1}{ralign 9:{res:{sf:    19028}}}{space 1}{space 1}{ralign 9:{res:{sf: .1919802}}}{space 1}{space 1}{ralign 9:{res:{sf:  .155132}}}{space 1}{space 1}{ralign 9:{res:{sf:  .393868}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     3653}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS1_21}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    19028}}}{space 1}{space 1}{ralign 9:{res:{sf:    19028}}}{space 1}{space 1}{ralign 9:{res:{sf: .2815325}}}{space 1}{space 1}{ralign 9:{res:{sf: .2022826}}}{space 1}{space 1}{ralign 9:{res:{sf: .4497583}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     5357}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS1_31}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    19028}}}{space 1}{space 1}{ralign 9:{res:{sf:    19028}}}{space 1}{space 1}{ralign 9:{res:{sf: .2046458}}}{space 1}{space 1}{ralign 9:{res:{sf: .1627744}}}{space 1}{space 1}{ralign 9:{res:{sf: .4034531}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     3894}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS1_41}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    19028}}}{space 1}{space 1}{ralign 9:{res:{sf:    19028}}}{space 1}{space 1}{ralign 9:{res:{sf: .0879756}}}{space 1}{space 1}{ralign 9:{res:{sf: .0802401}}}{space 1}{space 1}{ralign 9:{res:{sf: .2832669}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     1674}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS1_81}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    19028}}}{space 1}{space 1}{ralign 9:{res:{sf:    19028}}}{space 1}{space 1}{ralign 9:{res:{sf: .1024806}}}{space 1}{space 1}{ralign 9:{res:{sf: .0919831}}}{space 1}{space 1}{ralign 9:{res:{sf: .3032872}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     1950}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS1_91}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    19028}}}{space 1}{space 1}{ralign 9:{res:{sf:    19028}}}{space 1}{space 1}{ralign 9:{res:{sf: .0926529}}}{space 1}{space 1}{ralign 9:{res:{sf: .0840728}}}{space 1}{space 1}{ralign 9:{res:{sf: .2899531}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     1763}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS1_96}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    19028}}}{space 1}{space 1}{ralign 9:{res:{sf:    19028}}}{space 1}{space 1}{ralign 9:{res:{sf: .0387324}}}{space 1}{space 1}{ralign 9:{res:{sf: .0372342}}}{space 1}{space 1}{ralign 9:{res:{sf: .1929615}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:      737}}}{space 1}
{space 0}{space 0}{ralign 12:water_trea~t}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    19003}}}{space 1}{space 1}{ralign 9:{res:{sf:    19003}}}{space 1}{space 1}{ralign 9:{res:{sf: .1341893}}}{space 1}{space 1}{ralign 9:{res:{sf: .1161887}}}{space 1}{space 1}{ralign 9:{res:{sf: .3408646}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     2550}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS10_0}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    19030}}}{space 1}{space 1}{ralign 9:{res:{sf:    19030}}}{space 1}{space 1}{ralign 9:{res:{sf: .7638466}}}{space 1}{space 1}{ralign 9:{res:{sf: .1803945}}}{space 1}{space 1}{ralign 9:{res:{sf: .4247287}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:    14536}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS10_1}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    19030}}}{space 1}{space 1}{ralign 9:{res:{sf:    19030}}}{space 1}{space 1}{ralign 9:{res:{sf: .0707304}}}{space 1}{space 1}{ralign 9:{res:{sf: .0657311}}}{space 1}{space 1}{ralign 9:{res:{sf: .2563807}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     1346}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS10_2}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    19030}}}{space 1}{space 1}{ralign 9:{res:{sf:    19030}}}{space 1}{space 1}{ralign 9:{res:{sf: .0428271}}}{space 1}{space 1}{ralign 9:{res:{sf: .0409951}}}{space 1}{space 1}{ralign 9:{res:{sf: .2024725}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:      815}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS10_3}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    19030}}}{space 1}{space 1}{ralign 9:{res:{sf:    19030}}}{space 1}{space 1}{ralign 9:{res:{sf: .0371519}}}{space 1}{space 1}{ralign 9:{res:{sf: .0357735}}}{space 1}{space 1}{ralign 9:{res:{sf: .1891388}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:      707}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS10_7}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    19030}}}{space 1}{space 1}{ralign 9:{res:{sf:    19030}}}{space 1}{space 1}{ralign 9:{res:{sf:  .055176}}}{space 1}{space 1}{ralign 9:{res:{sf: .0521344}}}{space 1}{space 1}{ralign 9:{res:{sf: .2283295}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     1050}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS10_98}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    19030}}}{space 1}{space 1}{ralign 9:{res:{sf:    19030}}}{space 1}{space 1}{ralign 9:{res:{sf: .0295849}}}{space 1}{space 1}{ralign 9:{res:{sf: .0287111}}}{space 1}{space 1}{ralign 9:{res:{sf: .1694435}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:      563}}}{space 1}
{space 0}{space 0}{ralign 12:G_WS10_99}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    19030}}}{space 1}{space 1}{ralign 9:{res:{sf:    19030}}}{space 1}{space 1}{ralign 9:{res:{sf: .0006831}}}{space 1}{space 1}{ralign 9:{res:{sf: .0006827}}}{space 1}{space 1}{ralign 9:{res:{sf: .0261285}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:       13}}}{space 1}
{space 0}{space 0}{ralign 12:WS3_1}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    19030}}}{space 1}{space 1}{ralign 9:{res:{sf:    19030}}}{space 1}{space 1}{ralign 9:{res:{sf: .0176563}}}{space 1}{space 1}{ralign 9:{res:{sf: .0173455}}}{space 1}{space 1}{ralign 9:{res:{sf: .1317023}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:      336}}}{space 1}
{space 0}{space 0}{ralign 12:WS3_2}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    19030}}}{space 1}{space 1}{ralign 9:{res:{sf:    19030}}}{space 1}{space 1}{ralign 9:{res:{sf: .2021019}}}{space 1}{space 1}{ralign 9:{res:{sf: .1612652}}}{space 1}{space 1}{ralign 9:{res:{sf: .4015784}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     3846}}}{space 1}
{space 0}{space 0}{ralign 12:WS3_3}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    19030}}}{space 1}{space 1}{ralign 9:{res:{sf:    19030}}}{space 1}{space 1}{ralign 9:{res:{sf: .6113505}}}{space 1}{space 1}{ralign 9:{res:{sf: .2376136}}}{space 1}{space 1}{ralign 9:{res:{sf: .4874562}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:    11634}}}{space 1}
{space 0}{space 0}{ralign 12:urban}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    19030}}}{space 1}{space 1}{ralign 9:{res:{sf:    19030}}}{space 1}{space 1}{ralign 9:{res:{sf: .2515502}}}{space 1}{space 1}{ralign 9:{res:{sf: .1882826}}}{space 1}{space 1}{ralign 9:{res:{sf: .4339154}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     4787}}}{space 1}
{space 0}{space 0}{ralign 12:Open_defec~n}{space 1}{c |}{space 1}{ralign 9:{res:{sf:    18406}}}{space 1}{space 1}{ralign 9:{res:{sf:    18406}}}{space 1}{space 1}{ralign 9:{res:{sf: .3242964}}}{space 1}{space 1}{ralign 9:{res:{sf: .2191402}}}{space 1}{space 1}{ralign 9:{res:{sf: .4681241}}}{space 1}{space 1}{ralign 9:{res:{sf:        0}}}{space 1}{space 1}{ralign 9:{res:{sf:        1}}}{space 1}{space 1}{ralign 9:{res:{sf:     5969}}}{space 1}
{res}{txt}(output written to {browse  `"/Users/akitokamei/Dropbox/Apps/Overleaf/MICS_Water/Table/Descript_RFcontrols_Risk.tex"'})

{com}.           
. 
.                                                                                 ********************************************************************************
.                                                                                 **                      # Some risk at the source (Random Forest)
.                                                                                 ********************************************************************************
. 
. 
. use "${c -(}Data{c )-}MASTER_MICS_RF_Home.dta", clear
{txt}
{com}. keep RiskSource_0_12 $V_simple id split
{txt}
{com}. 
. *** Run random forest estimate ***
. 
. * Parameters: 
. * Number of variables = controls/3 = 119/3 = 40
. * Depth = 5 for regressions
. 
. rforest RiskSource_0_12 $V_simple if split == 1, type(reg) iter(500) seed(1666994) 
{res}
{txt}
{com}. 
. *** Also compute R2 ***
.          
. *** Create a copy of the variable-importance matrix stored in e()
. 
.         matrix importance = e(importance)
{txt}
{com}.         svmat importance
{txt}
{com}.         gen oob_error = e(OOB_Error)
{txt}
{com}.         gen features = e(features) 
{txt}
{com}.         gen obs =  e(Observations)
{txt}
{com}. 
. *** Error on the test set ***
. 
. predict p if split == 2
{txt}
{com}. 
. gen validation_rmse1 = `e(RMSE)'
{txt}
{com}. gen validation_mae1 = `e(MAE)'
{txt}
{com}. 
. label variable  validation_rmse1 "RMSE"
{txt}
{com}. label variable  validation_mae1 "MAE"
{txt}
{com}. label variable  oob_error "Out-of-bag Error"
{txt}
{com}. label variable  features "Number of predictors"
{txt}
{com}. label variable  obs "Number of observations"
{txt}
{com}. 
. *** Full sample prediction ***
. 
. predict p_all 
{txt}
{com}. 
. * hist p_all, by(split)
. * hist needall, by(split)
. 
. drop p 
{txt}
{com}. 
. save "${c -(}Data{c )-}rf-output.dta", replace
{txt}{p 0 4 2}
file {bf}
/Users/akitokamei/Box Sync/MICS Water project/Data/rf-output.dta{rm}
saved
{p_end}

{com}. 
. eststo clear
{txt}
{com}. use "${c -(}Data{c )-}rf-output.dta", clear
{txt}
{com}. eststo temp1: reg RiskSource_0_12 p_all if split==1

{txt}      Source {c |}       SS           df       MS      Number of obs   ={res}    31,173
{txt}{hline 13}{c +}{hline 34}   F(1, 31171)     = {res} 15760.66
{txt}       Model {c |} {res} 2563.49663         1  2563.49663   {txt}Prob > F        ={res}    0.0000
{txt}    Residual {c |} {res} 5070.01282    31,171  .162651594   {txt}R-squared       ={res}    0.3358
{txt}{hline 13}{c +}{hline 34}   Adj R-squared   ={res}    0.3358
{txt}       Total {c |} {res} 7633.50945    31,172  .244883532   {txt}Root MSE        =   {res}  .4033

{txt}{hline 13}{c TT}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{col 1}RiskSourc~12{col 14}{c |} Coefficient{col 26}  Std. err.{col 38}      t{col 46}   P>|t|{col 54}     [95% con{col 67}f. interval]
{hline 13}{c +}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{space 7}p_all {c |}{col 14}{res}{space 2} 1.008199{col 26}{space 2} .0080308{col 37}{space 1}  125.54{col 46}{space 3}0.000{col 54}{space 4} .9924583{col 67}{space 3}  1.02394
{txt}{space 7}_cons {c |}{col 14}{res}{space 2}-.0047478{col 26}{space 2} .0051277{col 37}{space 1}   -0.93{col 46}{space 3}0.354{col 54}{space 4}-.0147982{col 67}{space 3} .0053026
{txt}{hline 13}{c BT}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{res}{txt}
{com}. sum RiskSource_0_12 if split==1

{txt}    Variable {c |}        Obs        Mean    Std. dev.       Min        Max
{hline 13}{c +}{hline 57}
RiskSourc~12 {c |}{res}     31,173    .5715844    .4948571          0          1
{txt}
{com}. estadd scalar Min = r(min) : temp1
{txt}
{com}. estadd scalar Max = r(max) : temp1
{txt}
{com}. eststo temp2: reg RiskSource_0_12 p_all if split==2

{txt}      Source {c |}       SS           df       MS      Number of obs   ={res}    31,006
{txt}{hline 13}{c +}{hline 34}   F(1, 31004)     = {res} 12851.67
{txt}       Model {c |} {res} 2225.41629         1  2225.41629   {txt}Prob > F        ={res}    0.0000
{txt}    Residual {c |} {res}  5368.7051    31,004  .173161692   {txt}R-squared       ={res}    0.2930
{txt}{hline 13}{c +}{hline 34}   Adj R-squared   ={res}    0.2930
{txt}       Total {c |} {res}  7594.1214    31,005  .244932153   {txt}Root MSE        =   {res} .41613

{txt}{hline 13}{c TT}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{col 1}RiskSourc~12{col 14}{c |} Coefficient{col 26}  Std. err.{col 38}      t{col 46}   P>|t|{col 54}     [95% con{col 67}f. interval]
{hline 13}{c +}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{space 7}p_all {c |}{col 14}{res}{space 2} .9462354{col 26}{space 2} .0083468{col 37}{space 1}  113.37{col 46}{space 3}0.000{col 54}{space 4} .9298754{col 67}{space 3} .9625955
{txt}{space 7}_cons {c |}{col 14}{res}{space 2} .0294294{col 26}{space 2} .0053317{col 37}{space 1}    5.52{col 46}{space 3}0.000{col 54}{space 4}  .018979{col 67}{space 3} .0398798
{txt}{hline 13}{c BT}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{res}{txt}
{com}. sum RiskSource_0_12 if split==2

{txt}    Variable {c |}        Obs        Mean    Std. dev.       Min        Max
{hline 13}{c +}{hline 57}
RiskSourc~12 {c |}{res}     31,006    .5712443    .4949062          0          1
{txt}
{com}. estadd scalar Min = r(min) : temp2
{txt}
{com}. estadd scalar Max = r(max) : temp2
{txt}
{com}. esttab using "${c -(}Table{c )-}RFP.tex",label se ar2 title("The performance of the model for the training and testing sample" \label{c -(}RF{c )-}) nonotes nobase ///
>                          mtitle("Training" "Testing") drop(p_all _cons) ///
>                          stats(r2_a rmse Min Max   N, fmt(%9.2fc %9.2fc %9.2fc %9.2fc %9.0fc) labels(`"Adjusted \(R^{c -(}2{c )-}\)"' `"RMSE"' `"Min"' `"Max"'  `"Observations"')) ///
>                          starlevels(\sym{c -(}*{c )-} 0.10 \sym{c -(}**{c )-} 0.05 \sym{c -(}***{c )-} 0.010) b(2) ///
>                          substitute("{c -(}l{c )-}{c -(}\footnotesize" "{c -(}p{c -(}0.5\linewidth{c )-}{c )-}{c -(}\footnotesize" ///
>                          "=1" "" ///
>                          ) ///
>                          addnote("Note: ") ///  
>                          replace
{res}{txt}(output written to {browse  `"/Users/akitokamei/Dropbox/Apps/Overleaf/MICS_Water/Table/RFP.tex"'})

{com}. eststo clear
{txt}
{com}. 
. 
. 
. reg RiskSource_0_12 p_all

{txt}      Source {c |}       SS           df       MS      Number of obs   ={res}    62,179
{txt}{hline 13}{c +}{hline 34}   F(1, 62177)     = {res} 28482.65
{txt}       Model {c |} {res} 4784.08323         1  4784.08323   {txt}Prob > F        ={res}    0.0000
{txt}    Residual {c |} {res} 10443.5494    62,177  .167964833   {txt}R-squared       ={res}    0.3142
{txt}{hline 13}{c +}{hline 34}   Adj R-squared   ={res}    0.3142
{txt}       Total {c |} {res} 15227.6326    62,178  .244903867   {txt}Root MSE        =   {res} .40984

{txt}{hline 13}{c TT}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{col 1}RiskSourc~12{col 14}{c |} Coefficient{col 26}  Std. err.{col 38}      t{col 46}   P>|t|{col 54}     [95% con{col 67}f. interval]
{hline 13}{c +}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{space 7}p_all {c |}{col 14}{res}{space 2} .9774391{col 26}{space 2} .0057916{col 37}{space 1}  168.77{col 46}{space 3}0.000{col 54}{space 4} .9660875{col 67}{space 3} .9887907
{txt}{space 7}_cons {c |}{col 14}{res}{space 2} .0122007{col 26}{space 2} .0036987{col 37}{space 1}    3.30{col 46}{space 3}0.001{col 54}{space 4} .0049512{col 67}{space 3} .0194503
{txt}{hline 13}{c BT}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{res}{txt}
{com}. twoway (lowess RiskSource_0_12 p_all if split==1) ///
>        (lowess RiskSource_0_12 p_all if split==2, msize(tiny)) ///
>        (lfit   RiskSource_0_12 RiskSource_0_12 if split==2, lpattern(dot) lcolor(black)) , ///
>            legend(order(1 "Training sample" 2 "Test sample" 3 "45 degree line")) ///
>            xtitle("Actual risk") ///
>            ytitle("Predicted value")
{res}{txt}
{com}. graph export "${c -(}Figure{c )-}RFP.eps", replace
{txt}{p 0 4 2}
file {bf}
/Users/akitokamei/Dropbox/Apps/Overleaf/MICS_Water/Figure/RFP.eps{rm}
saved as
EPS
format
{p_end}

{com}.            
. ********************************************************************************
. **# Variable Importance
. ********************************************************************************
. use "${c -(}Data{c )-}rf-output.dta", clear
{txt}
{com}. keep RiskSource_0_12 $V_simple id split importance1
{txt}
{com}. 
. *** Generate new variable id to be used for labeling ***
.         gen names=""
{txt}(62,179 missing values generated)

{com}. 
. *** Attach unique labels to individual columns in the chart ***
.         local mynames : rownames importance
{txt}
{com}.         local k : word count `mynames'
{txt}
{com}.             // If there are more variables than observations
.             if `k'>_N {c -(}
.                 set obs `k'
.             {c )-}
{txt}
{com}.             forvalues i = 1(1)`k' {c -(}
{txt}  2{com}.                 local aword : word `i' of `mynames'
{txt}  3{com}.                 local alabel : variable label `aword'
{txt}  4{com}.                 if ("`alabel'"!="") quietly replace names= "`alabel'" in `i'
{txt}  5{com}.                 else quietly replace names= "`aword'" in `i'
{txt}  6{com}.             {c )-}
{txt}
{com}.                         
. 
. sort importance1
{txt}
{com}. 
. * Drop rows with missing information
. drop if importance1 ==. | names == ""
{txt}(62,137 observations deleted)

{com}. 
. * Split into 4 panels
. gen row = _n
{txt}
{com}. gen group = 1
{txt}
{com}. * replace group = 2 if row >= 29 & row < 58
. * replace group = 3 if row >= 58 & row < 87
. * replace group = 4 if row >= 87
. 
. graph hbar importance1 if group == 1, over(names, sort(1) label(labsize(vsmall))) ytitle("") ///
>         nofill noext dots(mcolor(gs10)) ylab(0(0.1)1, glcolor(gs15) glstyle(solid)) plotregion(lcolor(black) lwidth(.2) )  ///
>         graphregion(color(white))
{res}{txt}
{com}. graph export "${c -(}Figure{c )-}RFI.eps", replace
{txt}{p 0 4 2}
file {bf}
/Users/akitokamei/Dropbox/Apps/Overleaf/MICS_Water/Figure/RFI.eps{rm}
saved as
EPS
format
{p_end}

{com}. 
.                                                                                 ********************************************************************************
.                                                                                 **                      # Some risk at the source (Random Forest)
.                                                                                 ********************************************************************************
. 
. 
. use "${c -(}Data{c )-}MASTER_MICS_RF_Home1.dta", clear
{txt}
{com}. drop if RiskHome==0
{txt}(2,052 observations deleted)

{com}. keep NoRiskHome_01_2 $V_simple id split
{txt}
{com}. 
. *** Run random forest estimate ***
. 
. * Parameters: 
. * Number of variables = controls/3 = 119/3 = 40
. * Depth = 5 for regressions
. 
. rforest NoRiskHome_01_2 $V_simple if split == 1, type(reg) iter(500) seed(1666994) 
{res}
{txt}
{com}. 
. *** Also compute R2 ***
.          
. *** Create a copy of the variable-importance matrix stored in e()
. 
.         matrix importance = e(importance)
{txt}
{com}.         svmat importance
{txt}
{com}.         gen oob_error = e(OOB_Error)
{txt}
{com}.         gen features = e(features) 
{txt}
{com}.         gen obs =  e(Observations)
{txt}
{com}. 
. *** Error on the test set ***
. 
. predict p if split == 2
{txt}
{com}. 
. gen validation_rmse1 = `e(RMSE)'
{txt}
{com}. gen validation_mae1 = `e(MAE)'
{txt}
{com}. 
. label variable  validation_rmse1 "RMSE"
{txt}
{com}. label variable  validation_mae1 "MAE"
{txt}
{com}. label variable  oob_error "Out-of-bag Error"
{txt}
{com}. label variable  features "Number of predictors"
{txt}
{com}. label variable  obs "Number of observations"
{txt}
{com}. 
. *** Full sample prediction ***
. 
. predict p_all 
{txt}
{com}. 
. * hist p_all, by(split)
. * hist needall, by(split)
. 
. drop p 
{txt}
{com}. 
. save "${c -(}Data{c )-}rf-output.dta", replace
{txt}{p 0 4 2}
file {bf}
/Users/akitokamei/Box Sync/MICS Water project/Data/rf-output.dta{rm}
saved
{p_end}

{com}. 
. eststo clear
{txt}
{com}. use "${c -(}Data{c )-}rf-output.dta", clear
{txt}
{com}. eststo temp1: reg NoRiskHome_01_2 p_all if split==1

{txt}      Source {c |}       SS           df       MS      Number of obs   ={res}    10,512
{txt}{hline 13}{c +}{hline 34}   F(1, 10510)     = {res}  2510.70
{txt}       Model {c |} {res} 409.621328         1  409.621328   {txt}Prob > F        ={res}    0.0000
{txt}    Residual {c |} {res} 1714.70658    10,510  .163150008   {txt}R-squared       ={res}    0.1928
{txt}{hline 13}{c +}{hline 34}   Adj R-squared   ={res}    0.1927
{txt}       Total {c |} {res} 2124.32791    10,511  .202105215   {txt}Root MSE        =   {res} .40392

{txt}{hline 13}{c TT}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{col 1}NoRiskHome~2{col 14}{c |} Coefficient{col 26}  Std. err.{col 38}      t{col 46}   P>|t|{col 54}     [95% con{col 67}f. interval]
{hline 13}{c +}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{space 7}p_all {c |}{col 14}{res}{space 2} 1.039756{col 26}{space 2} .0207507{col 37}{space 1}   50.11{col 46}{space 3}0.000{col 54}{space 4} .9990804{col 67}{space 3} 1.080431
{txt}{space 7}_cons {c |}{col 14}{res}{space 2}-.0282979{col 26}{space 2} .0154235{col 37}{space 1}   -1.83{col 46}{space 3}0.067{col 54}{space 4} -.058531{col 67}{space 3} .0019352
{txt}{hline 13}{c BT}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{res}{txt}
{com}. sum NoRiskHome_01_2 if split==1

{txt}    Variable {c |}        Obs        Mean    Std. dev.       Min        Max
{hline 13}{c +}{hline 57}
NoRiskHome~2 {c |}{res}     10,512    .7188927    .4495611          0          1
{txt}
{com}. estadd scalar Min = r(min) : temp1
{txt}
{com}. estadd scalar Max = r(max) : temp1
{txt}
{com}. eststo temp2: reg NoRiskHome_01_2 p_all if split==2

{txt}      Source {c |}       SS           df       MS      Number of obs   ={res}    10,310
{txt}{hline 13}{c +}{hline 34}   F(1, 10308)     = {res}  1214.43
{txt}       Model {c |} {res} 217.911494         1  217.911494   {txt}Prob > F        ={res}    0.0000
{txt}    Residual {c |} {res} 1849.61285    10,308  .179434697   {txt}R-squared       ={res}    0.1054
{txt}{hline 13}{c +}{hline 34}   Adj R-squared   ={res}    0.1053
{txt}       Total {c |} {res} 2067.52435    10,309  .200555276   {txt}Root MSE        =   {res}  .4236

{txt}{hline 13}{c TT}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{col 1}NoRiskHome~2{col 14}{c |} Coefficient{col 26}  Std. err.{col 38}      t{col 46}   P>|t|{col 54}     [95% con{col 67}f. interval]
{hline 13}{c +}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{space 7}p_all {c |}{col 14}{res}{space 2} .7616949{col 26}{space 2} .0218572{col 37}{space 1}   34.85{col 46}{space 3}0.000{col 54}{space 4} .7188506{col 67}{space 3} .8045393
{txt}{space 7}_cons {c |}{col 14}{res}{space 2} .1767683{col 26}{space 2} .0162036{col 37}{space 1}   10.91{col 46}{space 3}0.000{col 54}{space 4} .1450062{col 67}{space 3} .2085305
{txt}{hline 13}{c BT}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{res}{txt}
{com}. sum NoRiskHome_01_2 if split==2

{txt}    Variable {c |}        Obs        Mean    Std. dev.       Min        Max
{hline 13}{c +}{hline 57}
NoRiskHome~2 {c |}{res}     10,310    .7224054     .447834          0          1
{txt}
{com}. estadd scalar Min = r(min) : temp2
{txt}
{com}. estadd scalar Max = r(max) : temp2
{txt}
{com}. esttab using "${c -(}Table{c )-}RFP1.tex",label se ar2 title("The performance of the model for the training and testing sample: Determinants of having very high risk drinking water from the housheolds with some contamination" \label{c -(}ETR1{c )-}) nonotes nobase ///
>                          mtitle("Training" "Testing") drop(p_all _cons) ///
>                          stats(r2_a rmse Min Max   N, fmt(%9.2fc %9.2fc %9.2fc %9.2fc %9.0fc) labels(`"Adjusted \(R^{c -(}2{c )-}\)"' `"RMSE"' `"Min"' `"Max"'  `"Observations"')) ///
>                          starlevels(\sym{c -(}*{c )-} 0.10 \sym{c -(}**{c )-} 0.05 \sym{c -(}***{c )-} 0.010) b(2) ///
>                          substitute("{c -(}l{c )-}{c -(}\footnotesize" "{c -(}p{c -(}1\linewidth{c )-}{c )-}{c -(}\footnotesize" ///
>                          "=1" "" ///
>                          ) ///
>                          addnote("Note: The base of the socio-economic level is the two lowest quintile poor and very poor. Standard errors clustered at the primary sampling unit in parentheses, $\sym{c -(}*{c )-} p<.10,\sym{c -(}**{c )-} p<.05,\sym{c -(}***{c )-} p<.01$") /// 
>                          replace
{res}{txt}(output written to {browse  `"/Users/akitokamei/Dropbox/Apps/Overleaf/MICS_Water/Table/RFP1.tex"'})

{com}. eststo clear
{txt}
{com}. 
. 
. 
. reg NoRiskHome_01_2 p_all

{txt}      Source {c |}       SS           df       MS      Number of obs   ={res}    20,822
{txt}{hline 13}{c +}{hline 34}   F(1, 20820)     = {res}  3565.06
{txt}       Model {c |} {res} 612.852355         1  612.852355   {txt}Prob > F        ={res}    0.0000
{txt}    Residual {c |} {res} 3579.06413    20,820  .171905097   {txt}R-squared       ={res}    0.1462
{txt}{hline 13}{c +}{hline 34}   Adj R-squared   ={res}    0.1462
{txt}       Total {c |} {res} 4191.91648    20,821  .201331179   {txt}Root MSE        =   {res} .41461

{txt}{hline 13}{c TT}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{col 1}NoRiskHome~2{col 14}{c |} Coefficient{col 26}  Std. err.{col 38}      t{col 46}   P>|t|{col 54}     [95% con{col 67}f. interval]
{hline 13}{c +}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{space 7}p_all {c |}{col 14}{res}{space 2} .9012466{col 26}{space 2} .0150942{col 37}{space 1}   59.71{col 46}{space 3}0.000{col 54}{space 4} .8716608{col 67}{space 3} .9308324
{txt}{space 7}_cons {c |}{col 14}{res}{space 2} .0739924{col 26}{space 2} .0112047{col 37}{space 1}    6.60{col 46}{space 3}0.000{col 54}{space 4} .0520304{col 67}{space 3} .0959545
{txt}{hline 13}{c BT}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{res}{txt}
{com}. twoway (lowess NoRiskHome_01_2 p_all if split==1) ///
>        (lowess NoRiskHome_01_2 p_all if split==2, msize(tiny)) ///
>        (lfit   NoRiskHome_01_2 NoRiskHome_01_2 if split==2, lpattern(dot) lcolor(black)) , ///
>            legend(order(1 "Training sample" 2 "Test sample" 3 "45 degree line")) ///
>            xtitle("Actual risk") ///
>            ytitle("Predicted value")
{res}{txt}
{com}. graph export "${c -(}Figure{c )-}RFP1.eps", replace
{txt}{p 0 4 2}
file {bf}
/Users/akitokamei/Dropbox/Apps/Overleaf/MICS_Water/Figure/RFP1.eps{rm}
saved as
EPS
format
{p_end}

{com}.            
. ********************************************************************************
. **# Variable Importance
. ********************************************************************************
. use "${c -(}Data{c )-}rf-output.dta", clear
{txt}
{com}. keep NoRiskHome_01_2 $V_simple id split importance1
{txt}
{com}. 
. *** Generate new variable id to be used for labeling ***
.         gen names=""
{txt}(20,822 missing values generated)

{com}. 
. *** Attach unique labels to individual columns in the chart ***
.         local mynames : rownames importance
{txt}
{com}.         local k : word count `mynames'
{txt}
{com}.             // If there are more variables than observations
.             if `k'>_N {c -(}
.                 set obs `k'
.             {c )-}
{txt}
{com}.             forvalues i = 1(1)`k' {c -(}
{txt}  2{com}.                 local aword : word `i' of `mynames'
{txt}  3{com}.                 local alabel : variable label `aword'
{txt}  4{com}.                 if ("`alabel'"!="") quietly replace names= "`alabel'" in `i'
{txt}  5{com}.                 else quietly replace names= "`aword'" in `i'
{txt}  6{com}.             {c )-}
{txt}
{com}.                         
. 
. sort importance1
{txt}
{com}. 
. * Drop rows with missing information
. drop if importance1 ==. | names == ""
{txt}(20,780 observations deleted)

{com}. 
. * Split into 4 panels
. gen row = _n
{txt}
{com}. gen group = 1
{txt}
{com}. * replace group = 2 if row >= 29 & row < 58
. * replace group = 3 if row >= 58 & row < 87
. * replace group = 4 if row >= 87
. 
. graph hbar importance1 if group == 1, over(names, sort(1) label(labsize(vsmall))) ytitle("") ///
>         nofill noext dots(mcolor(gs10)) ylab(0(0.1)1, glcolor(gs15) glstyle(solid)) plotregion(lcolor(black) lwidth(.2) )  ///
>         graphregion(color(white))
{res}{txt}
{com}. graph export "${c -(}Figure{c )-}RFI1.eps", replace
{txt}{p 0 4 2}
file {bf}
/Users/akitokamei/Dropbox/Apps/Overleaf/MICS_Water/Figure/RFI1.eps{rm}
saved as
EPS
format
{p_end}

{com}. 
. 
.                                                                                 ********************************************************************************
.                                                                                 **                      # No risk at the source (Random Forest)
.                                                                                 ********************************************************************************
. 
. 
. use "${c -(}Data{c )-}MASTER_MICS_RF_Home0.dta", clear
{txt}
{com}. keep RiskHome_0_12 $V_simple id split
{txt}
{com}. 
. *** Run random forest estimate ***
. 
. * Parameters: 
. * Number of variables = controls/3 = 119/3 = 40
. * Depth = 5 for regressions
. 
. rforest  RiskHome_0_12 $V_simple if split == 1, type(reg) iter(500) seed(1666994) 
{res}
{txt}
{com}. 
. *** Also compute R2 ***
.          
. *** Create a copy of the variable-importance matrix stored in e()
. 
.         matrix importance = e(importance)
{txt}
{com}.         svmat importance
{txt}
{com}.         gen oob_error = e(OOB_Error)
{txt}
{com}.         gen features = e(features) 
{txt}
{com}.         gen obs =  e(Observations)
{txt}
{com}. 
. *** Error on the test set ***
. 
. predict p if split == 2
{txt}
{com}. 
. gen validation_rmse1 = `e(RMSE)'
{txt}
{com}. gen validation_mae1 = `e(MAE)'
{txt}
{com}. 
. label variable  validation_rmse1 "RMSE"
{txt}
{com}. label variable  validation_mae1 "MAE"
{txt}
{com}. label variable  oob_error "Out-of-bag Error"
{txt}
{com}. label variable  features "Number of predictors"
{txt}
{com}. label variable  obs "Number of observations"
{txt}
{com}. 
. *** Full sample prediction ***
. 
. predict p_all 
{txt}
{com}. 
. * hist p_all, by(split)
. * hist needall, by(split)
. 
. drop p 
{txt}
{com}. 
. save "${c -(}Data{c )-}rf-output.dta", replace
{txt}{p 0 4 2}
file {bf}
/Users/akitokamei/Box Sync/MICS Water project/Data/rf-output.dta{rm}
saved
{p_end}

{com}. 
. use "${c -(}Data{c )-}rf-output.dta", clear
{txt}
{com}. 
. eststo temp1: reg RiskHome_0_12 p_all if split==1

{txt}      Source {c |}       SS           df       MS      Number of obs   ={res}    13,355
{txt}{hline 13}{c +}{hline 34}   F(1, 13353)     = {res}  6224.88
{txt}       Model {c |} {res}  1060.2514         1   1060.2514   {txt}Prob > F        ={res}    0.0000
{txt}    Residual {c |} {res} 2274.34583    13,353  .170324708   {txt}R-squared       ={res}    0.3180
{txt}{hline 13}{c +}{hline 34}   Adj R-squared   ={res}    0.3179
{txt}       Total {c |} {res} 3334.59723    13,354  .249707745   {txt}Root MSE        =   {res}  .4127

{txt}{hline 13}{c TT}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{col 1}RiskHome_~12{col 14}{c |} Coefficient{col 26}  Std. err.{col 38}      t{col 46}   P>|t|{col 54}     [95% con{col 67}f. interval]
{hline 13}{c +}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{space 7}p_all {c |}{col 14}{res}{space 2} 1.016653{col 26}{space 2} .0128857{col 37}{space 1}   78.90{col 46}{space 3}0.000{col 54}{space 4} .9913952{col 67}{space 3} 1.041911
{txt}{space 7}_cons {c |}{col 14}{res}{space 2}-.0080231{col 26}{space 2} .0071684{col 37}{space 1}   -1.12{col 46}{space 3}0.263{col 54}{space 4}-.0220742{col 67}{space 3} .0060279
{txt}{hline 13}{c BT}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{res}{txt}
{com}. sum RiskHome_0_12 if split==1

{txt}    Variable {c |}        Obs        Mean    Std. dev.       Min        Max
{hline 13}{c +}{hline 57}
RiskHome_~12 {c |}{res}     13,355    .4823662    .4997077          0          1
{txt}
{com}. estadd scalar Min = r(min) : temp1
{txt}
{com}. estadd scalar Max = r(max) : temp1
{txt}
{com}. eststo temp2: reg RiskHome_0_12 p_all if split==2

{txt}      Source {c |}       SS           df       MS      Number of obs   ={res}    13,294
{txt}{hline 13}{c +}{hline 34}   F(1, 13292)     = {res}  4408.99
{txt}       Model {c |} {res} 826.971956         1  826.971956   {txt}Prob > F        ={res}    0.0000
{txt}    Residual {c |} {res}  2493.1153    13,292  .187565099   {txt}R-squared       ={res}    0.2491
{txt}{hline 13}{c +}{hline 34}   Adj R-squared   ={res}    0.2490
{txt}       Total {c |} {res} 3320.08726    13,293  .249762075   {txt}Root MSE        =   {res} .43309

{txt}{hline 13}{c TT}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{col 1}RiskHome_~12{col 14}{c |} Coefficient{col 26}  Std. err.{col 38}      t{col 46}   P>|t|{col 54}     [95% con{col 67}f. interval]
{hline 13}{c +}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{space 7}p_all {c |}{col 14}{res}{space 2} .9051422{col 26}{space 2} .0136316{col 37}{space 1}   66.40{col 46}{space 3}0.000{col 54}{space 4} .8784222{col 67}{space 3} .9318621
{txt}{space 7}_cons {c |}{col 14}{res}{space 2} .0441991{col 26}{space 2} .0076141{col 37}{space 1}    5.80{col 46}{space 3}0.000{col 54}{space 4} .0292743{col 67}{space 3} .0591239
{txt}{hline 13}{c BT}{hline 11}{hline 11}{hline 9}{hline 8}{hline 13}{hline 12}
{res}{txt}
{com}. sum RiskHome_0_12 if split==2

{txt}    Variable {c |}        Obs        Mean    Std. dev.       Min        Max
{hline 13}{c +}{hline 57}
RiskHome_~12 {c |}{res}     13,294    .4839777     .499762          0          1
{txt}
{com}. estadd scalar Min = r(min) : temp2
{txt}
{com}. estadd scalar Max = r(max) : temp2
{txt}
{com}. esttab using "${c -(}Table{c )-}RFP0.tex",label se ar2 title("The performance of the model for the training and testing sample: Determinants of having some E.Coli in the drinking water from the free from contamination water source" \label{c -(}ETR1{c )-}) nonotes nobase ///
>                          mtitle("Training" "Testing") drop(p_all _cons) ///
>                          stats(r2_a rmse Min Max   N, fmt(%9.2fc %9.2fc %9.2fc %9.2fc %9.0fc) labels(`"Adjusted \(R^{c -(}2{c )-}\)"' `"RMSE"' `"Min"' `"Max"'  `"Observations"')) ///
>                          starlevels(\sym{c -(}*{c )-} 0.10 \sym{c -(}**{c )-} 0.05 \sym{c -(}***{c )-} 0.010) b(2) ///
>                          substitute("{c -(}l{c )-}{c -(}\footnotesize" "{c -(}p{c -(}1\linewidth{c )-}{c )-}{c -(}\footnotesize" ///
>                          "=1" "" ///
>                          ) ///
>                          addnote("Note: The base of the socio-economic level is the two lowest quintile poor and very poor. Standard errors clustered at the primary sampling unit in parentheses, $\sym{c -(}*{c )-} p<.10,\sym{c -(}**{c )-} p<.05,\sym{c -(}***{c )-} p<.01$") /// 
>                          replace
{res}{txt}(output written to {browse  `"/Users/akitokamei/Dropbox/Apps/Overleaf/MICS_Water/Table/RFP0.tex"'})

{com}. eststo clear
{txt}
{com}. 
. 
. 
. twoway (lowess RiskHome_0_12 p_all if split==1, msize(tiny) msymbol(Oh)) ///
>        (lowess RiskHome_0_12 p_all if split==2, msize(tiny)) ///
>        (lfit RiskHome_0_12 RiskHome_0_12 if split==2, lpattern(dot) lcolor(black)) , ///
>            legend(order(1 "Training sample" 2 "Test sample" 3 "45 degree line")) ///
>            xtitle("Actual risk") ///
>            ytitle("Predicted value")
{res}{txt}
{com}. graph export "${c -(}Figure{c )-}RFP0.eps", replace
{txt}{p 0 4 2}
file {bf}
/Users/akitokamei/Dropbox/Apps/Overleaf/MICS_Water/Figure/RFP0.eps{rm}
saved as
EPS
format
{p_end}

{com}. 
. 
. ********************************************************************************
. **# Variable Importance
. ********************************************************************************
. use "${c -(}Data{c )-}rf-output.dta", clear
{txt}
{com}. keep RiskHome_0_12 $V_simple id split importance1
{txt}
{com}. 
. *** Generate new variable id to be used for labeling ***
.         gen names=""
{txt}(26,649 missing values generated)

{com}. 
. *** Attach unique labels to individual columns in the chart ***
.         local mynames : rownames importance
{txt}
{com}.         local k : word count `mynames'
{txt}
{com}.             // If there are more variables than observations
.             if `k'>_N {c -(}
.                 set obs `k'
.             {c )-}
{txt}
{com}.             forvalues i = 1(1)`k' {c -(}
{txt}  2{com}.                 local aword : word `i' of `mynames'
{txt}  3{com}.                 local alabel : variable label `aword'
{txt}  4{com}.                 if ("`alabel'"!="") quietly replace names= "`alabel'" in `i'
{txt}  5{com}.                 else quietly replace names= "`aword'" in `i'
{txt}  6{com}.             {c )-}
{txt}
{com}.                         
. 
. sort importance1
{txt}
{com}. 
. * Drop rows with missing information
. drop if importance1 ==. | names == ""
{txt}(26,607 observations deleted)

{com}. 
. * Split into 4 panels
. gen row = _n
{txt}
{com}. gen group = 1
{txt}
{com}. * replace group = 2 if row >= 29 & row < 58
. * replace group = 3 if row >= 58 & row < 87
. * replace group = 4 if row >= 87
. 
. graph hbar importance1 if group == 1, over(names, sort(1) label(labsize(vsmall))) ytitle("") ///
>         nofill noext dots(mcolor(gs10)) ylab(0(0.1)1, glcolor(gs15) glstyle(solid)) plotregion(lcolor(black) lwidth(.2) )  ///
>         graphregion(color(white))
{res}{txt}
{com}. graph export "${c -(}Figure{c )-}RFI0.eps", replace
{txt}{p 0 4 2}
file {bf}
/Users/akitokamei/Dropbox/Apps/Overleaf/MICS_Water/Figure/RFI0.eps{rm}
saved as
EPS
format
{p_end}

{com}. 
. 
. #del ;
{txt}delimiter now ;
{com}.         graph hbar importance1 if group == 1, ///
>         over(names, sort(1) label(labsize(vsmall))) ///
>         ytitle("") ///
>         ysize(1) nofill noext dots(mcolor(gs10)) ///
>         ylab(0(0.1)1, glcolor(gs15) glstyle(solid)) ///
>         plotregion(lcolor(black) lwidth(.2) )  ///
>         graphregion(color(white)) ///
>         name(A,replace) nodraw ///
> ;
{res}{txt}
{com}. #del cr
{txt}delimiter now cr
{com}. 
. /*
> #del ;
>         graph hbar importance1 if group == 2, ///
>         over(names, sort(1) label(labsize(vsmall))) ///
>         ytitle("") ///
>         ysize(1) nofill noext dots(mcolor(gs10)) ///
>         ylab(0(0.1)1, glcolor(gs15) glstyle(solid)) ///
>         plotregion(lcolor(black) lwidth(.2) )  ///
>         graphregion(color(white)) ///
>         name(B,replace) nodraw ///
> ;
> #del cr
> 
> graph combine A B, col(2)  b1(Importance) iscale(*0.75) graphregion(color(white))
> graph export "${c -(}Figure{c )-}Random Forest Importance_more.png", as(png) height(1200) replace
> */
. 
. END
{err}command {bf}END{sf} is unrecognized
{txt}{search r(199), local:r(199);}

end of do-file

{search r(199), local:r(199);}

{com}. clear all
{res}
{com}. exit
