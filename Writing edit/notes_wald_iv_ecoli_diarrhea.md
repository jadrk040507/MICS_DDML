# Research memo — Treatment-as-instrument (Wald/IV) for the effect of point-of-use *E. coli* on diarrhea

*Status: identification analysis only (no estimation run). Companion to the DDML pipeline.*

## 1. The question

The paper documents a **disconnect**: household water treatment robustly lowers measured point-of-use *E. coli*, yet has no credibly identified effect on child diarrhea. A natural follow-up is the *structural* link in between — **does point-of-use *E. coli* contamination cause diarrhea?** A Wald/IV design proposes to answer this by using water treatment as an instrument for contamination.

## 2. Setup and estimand

- $Z$ = household water treatment (binary, 0/1) — the **instrument**
- $D$ = point-of-use *E. coli* (binary, e.g. `SomeRiskHome` or `VeryHighRiskHome`) — the **endogenous regressor** (here a mediator of $Z$)
- $Y$ = child diarrhea (binary) — the **outcome**
- $X$ = the headline confounder vector (`BASE_CONFOUNDERS` + `U5_ADDITIONAL_CONFOUNDERS`)

Wald ratio (conditional on $X$):

$$\beta_{\text{Wald}} = \frac{\text{ITT}_{Z\to Y}}{\text{first stage}_{Z\to D}} = \frac{\mathbb{E}[Y\mid Z=1,X]-\mathbb{E}[Y\mid Z=0,X]}{\mathbb{E}[D\mid Z=1,X]-\mathbb{E}[D\mid Z=0,X]}.$$

With a binary instrument and binary treatment, this is the **LATE** of *E. coli* on diarrhea among **compliers** — households whose stored-water *E. coli* status is actually changed by treating.

## 3. The estimator and statistical correctness

This is exactly **`DoubleMLIIVM`** (interactive IV model) in `doubleml` (already installed; `irm/iivm.py`), score `'LATE'`. It estimates the LATE with three cross-fitted ML nuisances:

- $\hat g(Z,X) = \mathbb{E}[Y\mid Z,X]$ (outcome),
- $\hat m(X) = P(Z=1\mid X)$ (instrument propensity),
- $\hat r(Z,X) = \mathbb{E}[D\mid Z,X]$ (compliance / first stage).

The score is Neyman-orthogonal and doubly robust; cross-fitting removes regularization bias. **Statistically the estimator is recognized and correct** — same DML machinery as the rest of the paper. So the question is not statistical validity of the estimator but the **economic validity of the identifying assumptions**.

## 4. Identifying assumptions (LATE / Imbens–Angrist)

| Assumption | Status here |
|---|---|
| **Relevance** ($Z$ moves $D$) | ✅ **Strong.** Treatment robustly reduces *E. coli* — this is the paper's headline first stage. No weak-instrument problem. |
| **Independence / ignorability** ($Z \perp$ potential outcomes $\mid X$) | ⚠️ Assumed. This is the **same CIA** the IRM/APOS already invoke. No *new* assumption beyond the main design. |
| **Exclusion** ($Z$ affects $Y$ only through $D$) | ❌ **The binding threat — likely violated** (see §5). |
| **Monotonicity** (no defiers) | ✅ Plausible: treatment weakly reduces contamination for everyone. |
| **SUTVA** | Assumed, as throughout. |

## 5. Why exclusion is the binding threat

Exclusion requires that the *only* channel from treatment to diarrhea is the measured *E. coli* indicator. Two reasons it likely fails:

1. **Treatment acts on pathogens the indicator misses.** Boiling and chlorination inactivate viruses (rotavirus) and protozoa (*Giardia*, *Cryptosporidium*) that the *E. coli* test does not capture. If treatment lowers diarrhea by killing these, there is a **direct $Z\to Y$ path** not running through measured $D$ → exclusion violated, $\beta_{\text{Wald}}$ biased for the structural *E. coli*→diarrhea effect.
2. **\*E. coli\* is a proxy, not the causal agent.** The binary *E. coli* indicator is a noisy marker of fecal contamination, not the pathogen load itself. Instrumenting a proxy yields a **proxy-IV** estimand, attenuated/biased relative to the true contamination→diarrhea effect.

(Behavioral correlates of treating — storage, handwashing — threaten ignorability *and* exclusion if omitted from $X$; the rich control vector mitigates but cannot rule this out.)

## 6. Interpretation given the existing results

The reduced form $\text{ITT}_{Z\to Y}$ (treatment → diarrhea) is **near zero and fragile** (RV ≈ 0); the first stage $Z\to D$ is **large and robust**. Hence

$$\beta_{\text{Wald}} \approx \frac{\approx 0}{\text{large}} \approx 0 \quad\text{with wide CIs.}$$

Two readings, which the data cannot separate:

- **(a) Substantive:** point-of-use *E. coli* contamination has little causal effect on *measured* diarrhea in this population — a clean restatement of the paper's "disconnect."
- **(b) Exclusion failure:** any true *E. coli*→diarrhea effect is masked/rescaled because treatment's diarrhea effect (if any) flows through non-*E. coli* channels.

Either way, **the IV cannot rescue a diarrhea effect**: a null reduced form rescaled by a strong first stage stays null, only noisier. It will not produce a significant structural estimate.

## 7. Relation to mediation

This is the $T \to M \to Y$ chain with $T=Z$, $M=D$. The Wald ratio equals the **total effect of $T$ on $Y$ divided by the effect of $T$ on $M$** — i.e. the portion of $T$'s effect on $Y$ flowing through $M$ **only if** there is no direct $T\to Y$ path (exclusion). Under exclusion failure it conflates the indirect ($M$) and direct channels. So the Wald/IV here is best understood as a **structured restatement of the reduced-form disconnect**, not an independent identification of the contamination→diarrhea mechanism.

## 8. Recommendation

- **Worth reporting as an exploratory robustness exhibit**, with explicit exclusion-restriction caveats — *never* a headline.
- Frame the estimand precisely: "LATE of a treatment-induced change in *measured* point-of-use *E. coli* status on diarrhea, among compliers," not "the causal effect of contamination on diarrhea."
- Expect $\approx 0$ with wide CIs; report it as corroborating the disconnect.

### If pursued later (implementation sketch — not built)

- Model: `DoubleMLIIVM` (binary $Z$, binary $D$), one spec per $D \in \{$`SomeRiskHome`, `VeryHighRiskHome`$\}$, outcome `diarrhea`, on the U5 sample.
- Nuisances: reuse the project's binary-outcome learners (`ProbaRegressor`-wrapped classifiers) for $\hat g$, $\hat r$; classifier for $\hat m$. Stacked ensemble as the headline.
- Cross-fitting: `N_FOLDS=5`, `N_REP=3`; cluster on `HHID` (U5) if supported, else report i.i.d. with the singleton-cluster justification used for APOS.
- Diagnostics: instrument-propensity overlap; report the first-stage compliance share; sensitivity to the exclusion restriction is *not* testable — state it as an assumption.
- Cross-check: $\beta_{\text{Wald}}$ should ≈ (APOS/IRM diarrhea ATE) / (APOS/IRM *E. coli* ATE) — a useful internal consistency check.

---

## 9. Is treatment a valid instrument? (verdict)

$Z$ = household water treatment (the binary `water_treatment` any-treatment indicator).

A valid instrument needs **relevance + exogeneity + exclusion** (+ monotonicity for LATE). Conceptually:

- **Relevance** ✅ — treatment strongly moves contamination.
- **Exogeneity** ⚠️ — *only conditionally*. Treatment is a **choice, not randomized**; "instrument exogeneity" here is literally the paper's CIA (selection on observables), not design-based randomization. There is no lottery/discontinuity/natural experiment generating $Z$.
- **Exclusion** ❌ — likely fails: boiling/chlorination kill non-*E. coli* pathogens (rotavirus, *Giardia*, *Crypto*), and *E. coli* is a proxy → direct $Z\to Y$ paths.

**Key point:** re-casting treatment as an instrument **adds assumptions without adding identifying variation**. The direct analysis needs only the CIA; the IV needs CIA **plus** exclusion (untestable, likely false) **plus** monotonicity. So it is *strictly weaker* than the direct design. A genuine instrument would require exogenous variation in contamination (source-water/weather shocks, or a randomized treatment-promotion encouragement) — which this data does not have. The Wald/IV is therefore a **structured restatement of the reduced-form disconnect**, not independent identification of contamination→diarrhea.

## 10. Empirical instrument tests (results)

Run with [`42_run_iv_validity.py`](../Do%20file/Python/42_run_iv_validity.py) on the U5 sample ($n=36{,}121$), conditioning on the headline confounder vector, cluster-robust on `HHID`. Outputs: `Output/iv_validity_report.txt`, `Output/iv_validity_summary.csv`.

| Test | `SomeRiskHome` | `VeryHighRiskHome` | Reading |
|---|---|---|---|
| Relevance — first stage $\pi(Z\to D)$ | $-0.089$, partial $F=139$ | $-0.077$, $F=64$ | ✅ strong (no weak-IV) |
| Methods as 4 instruments — joint $F$ | $177.7$ | $114.1$ | ✅ strong |
| Reduced form ITT$(Z\to Y)$ | $+0.0002$ ($p=0.98$) | $+0.0002$ ($p=0.98$) | ≈ **exactly zero** |
| Just-identified Wald LATE$(D\to Y)$ | $-0.002$ | $-0.002$ | ≈ 0 |
| Overidentification — Sargan $J$ (df 3) | $3.12$ ($p=0.37$) | $3.07$ ($p=0.38$) | fail to reject |
| Monotonicity — subgroup first-stage signs | all $-$ | all $-$ | ✅ no defiers |

**Interpretation.**

- **Relevance: confirmed.** Treatment robustly lowers *E. coli* overall and method-by-method; first-stage $F\gg10$. By method, boiling has the largest first stage ($-0.16$), then "other", chlorine, filter — consistent with the APOS ranking.
- **Monotonicity: holds.** Every subgroup first stage is negative (treatment weakly reduces contamination); no sign flips → no obvious defiers. The one near-zero cell (very-high contamination among no-risk *source* households) is mechanical — those households almost never reach very-high point-of-use contamination, so there is nothing to reduce.
- **Exclusion: NOT validated.** The Sargan test fails to reject, but this is **near-powerless here**: the reduced form is $\approx0$ for *every* method, so the four instruments trivially "agree" on a null $D\to Y$. A non-rejection in a null-reduced-form setting is *not* evidence the exclusion restriction holds. The substantive threat (treatment acting on non-*E. coli* pathogens) is untouched by this test.
- **Exogeneity: not testable** — it is the CIA, assumed throughout.

**Bottom line (empirical).** The instrument is **relevant and monotone**, but its validity for identifying contamination→diarrhea rests entirely on the **untestable exogeneity (CIA)** and the **non-validated, likely-violated exclusion** restriction. Because the reduced form is essentially exactly zero, the Wald/IV LATE is $\approx0$ and uninformative: it corroborates the *E. coli*↔diarrhea disconnect but cannot establish (or refute) a structural mechanism. Report only as a caveated robustness exhibit.
