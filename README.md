# Bayesian false-positive tests on real ERW data

A Jupyter notebook that runs the same statistical tests as the [ERW false-positive explorer](https://github.com/mati-carbon/erw-fpr-explorer) on real paired (or unpaired) soil-chemistry data, end-to-end. Choose a Mati dataset (or supply your own), pick an immobile and mobile tracer, and the notebook produces gate tests, informational diagnostics, and a final color-coded verdict.

## What the notebook produces

| Test | Role | What it asks |
|---|---|---|
| Paired (or Welch) *t*-test on Δ*I* and Δ*X* | **Gate** | Is the observed change in concentration distinguishable from zero? |
| Statistical power (paired or Welch) | **Gate** | Is the test adequately powered to detect the observed effect? |
| Bayesian FPR via permutation inversion (p_r, p_rt) | **Gate** | Is the inferred CDR larger than what the soil-heterogeneity null produces? |
| Welch *t*-test (when data are paired) | Info | What's the cost of treating pre/post as independent? |
| Posterior shrinkage on τ | Info, with warning | How much weight does the data carry against the uniform prior on τ? |
| Joint posterior plot | Visual | The 2-D r–τ posterior, MAP, median, and the τ-indistinguishable region |

The final summary table color-codes every row (green = pass, yellow = caution, red = fail, grey = info) and the overall verdict is the worst-of among the three gates.

## How this differs from the in-browser explorer

The explorer is optimized for in-browser calculation speed. This notebook is optimized for robustness:

| | Explorer (HTML/JS) | This notebook |
|---|---|---|
| Likelihood for Δ*E* | Gaussian: σ = CV · mean(Δ) / √n | KDE over a bootstrap of (mean_post − mean_pre) |
| Posterior grid | 80 × 80 with cell-smearing variance correction | 160 × 160, no smearing |
| Null construction | Bivariate-Gaussian sampler with hardcoded ρ for paired vs unpaired | Permute the actual data: sign-flip per row (paired) or pool-and-reshuffle per element (unpaired) |

The notebook uses the empirical Δ*E* distribution and a real permutation null, which is more representative than the Gaussian approximations the explorer uses. Numerical agreement on individual *p*-values is typically within ±0.005 to ±0.02 for the Mati datasets.

## Files

```
fpr_explorer_on_real_data.ipynb   the notebook
utils.py                          minimal helper module (bootstrap, KDE, posterior inversion)
Mati_nchg.csv                     North Chhattisgarh 2024 — 182 paired samples
Mati_chg.csv                      Chhattisgarh 2024     — 128 paired samples
Mati_seoni.csv                    Seoni 2024            — 171 paired samples
README.md                         this file
```

Drop all of the above into the same folder and open the notebook. Outputs (PNG of the joint posterior plot) land in `outputs/` alongside.

## Requirements

Python 3.9+ with: `numpy`, `pandas`, `scipy`, `matplotlib`, `jupyter`. No other dependencies.

## Notebook structure

The notebook is organized so that everything a user might need to edit appears in the first few cells. The "Run me first" cell and Section 2 contain code that typically needs no changes; the analysis sections (3–10) run end-to-end with no further input.

### Run me first
Imports and module reload. No user configuration.

### Section 1a — Dataset selection
Pick one of three bundled Mati ERW datasets by uncommenting a single line. Each dataset entry in the `DATASETS` registry comes with its own hardcoded basalt endmember dict (in ppm), derived from aggregated source-fingerprint measurements of the actual basalt dispatched to that deployment.

The same section also documents how to add a custom dataset: prepare a CSV with `{Immobile}_pre`, `{Immobile}_post`, `{Mobile}_pre`, `{Mobile}_post` columns (all ppm), aggregate your basalt characterization to a per-element ppm dict, and add an entry to the `DATASETS` registry. Works for both paired data (each row is one location sampled pre and post) and unpaired data (each row is two independent observations).

### Section 1b — Tracer choice and paired/unpaired toggle
Three user choices: `IMMOBILE` (default `'Ti'`), `MOBILE` (default `'Ca'`), and `PAIRED` (`True` or `False`). The `PAIRED` flag selects both the null construction (sign-flip per row vs pool-and-reshuffle per element) and which *t*-test acts as the gate (paired vs Welch). The notebook handles one mobile tracer at a time — re-run with `MOBILE = 'Mg'` to see the alternative.

### Section 2 — Program setup
Constants the user generally doesn't need to touch: grid resolution (160 × 160), prior bounds (r ∈ [0, 0.156], τ ∈ [0, 1]), null and bootstrap sample sizes (`N_NULL = 1000`, `B_OBSERVED = 10000`, `B_NULL = 1000`), CDR-display parameters (`E_POT`, depth, bulk density), and the output directory. Edit only if you need tighter null resolution or a different prior range.

### Section 3 — Data ingestion and summary
Prints sample size, pre-treatment soil mean (= Cs), basalt endmember (= Cb), and bootstrap-free Δ*E* mean / std / CV for both tracers. Runs two structural sanity checks: the negative-Δ*E* screen (mean Δ*I* must be ≥ 0) and the basalt–soil contrast screen (Cb − Cs must be positive). A failure flags the chosen tracer as unusable for this framework.

### Section 4 — Forward statistical tests
Runs paired and Welch *t*-tests on each tracer, plus the normal-approximation statistical power and the *n* needed for 80% power. When `PAIRED = True` the paired test is the gate and Welch is informational; when `PAIRED = False` the roles swap and the paired test is skipped. Status thresholds match the explorer: *p* ≤ 0.05 pass / ≤ 0.10 caution / else fail; power ≥ 0.80 pass / ≥ 0.50 caution / else fail.

### Section 5 — Bayesian posterior inversion (observed data)
Runs the joint inversion `p(r, τ | Δ*I*, Δ*X*)` once on the chosen `(IMMOBILE, MOBILE)`. The likelihood is a KDE over the Δ*E* bootstrap distribution for each tracer; the forward model uses the soil-cation-dilution form `ΔX_pred = r · ((1 − τ) · Cb_X − Cs_X)`. Reports the joint MAP, marginal medians, 68% / 95% CIs, and the integrated E[r·τ] — the latter is the explorer's `p_rt` statistic (chosen over `r_MAP · τ_MAP` because it correctly accounts for the joint posterior's ridge correlation).

### Section 6 — Bayesian FPR via permutation null
Generates 1,000 null draws and re-runs the full inversion for each. Computes **p_r** = fraction where `r_median_null ≥ r_median_observed` and **p_rt** = fraction where `E[r·τ]_null ≥ E[r·τ]_observed`. The null construction follows `PAIRED`: paired data get a sign-flip null (per-row label swap with a sign vector shared across tracers, preserving multivariate correlation); unpaired data get a population-level permutation (pool baseline + reporting-period values per element, reshuffle, split). Decision rule: pass iff both *p* ≤ 0.05.

### Section 7 — Posterior shrinkage on r and τ
Computes `shrinkage = 1 − Var(posterior marginal) / Var(prior marginal)` and the equivalent data : prior odds ratio. Shrinkage on τ below 0.80 (< 4:1 data : prior odds) triggers a warning — this is the parameter that the framework most often fails to determine, because τ is unidentifiable when r is small.

### Section 8 — Joint posterior plot
The 2-D joint posterior with 68% and 95% HPD contours, a cross-hatched τ-indistinguishable region (where Δ*X* predicted from `(r, τ)` is within noise of zero), an orange-square MAP marker, and a blue-circle marginal-median marker. The τ-indistinguishable threshold uses `σ_X = std of the Δ*X* bootstrap distribution`, consistent with the KDE-of-bootstrap likelihood used in the inversion.

### Section 9 — τ marginal outputs
Print-only summary of the τ marginal (MAP, median, mean, 68% / 95% CIs) plus a check for whether the joint 95% HPD region overlaps the τ-indistinguishable zone. If it does, the τ estimate is not robust — the data alone cannot rule out values where the mobile-tracer constraint is degenerate.

### Section 10 — Final summary table
Color-coded pandas Styler output of every gate and informational test in one place. Overall verdict at the bottom is the worst-of among the three gates (*t*-test, power, Bayesian FPR). Informational rows (Welch when paired, posterior shrinkage) appear in grey and don't drive the verdict but flag follow-up considerations.

### Appendix
Notes on what each test is sensitive to (and what each one *misses* relative to the others), the rationale for using E[r·τ] rather than `r_MAP · τ_MAP`, and where outputs are written.

## How to run

1. Drop the files listed above into one folder.
2. Open `fpr_explorer_on_real_data.ipynb` in Jupyter.
3. In **Section 1a**, uncomment one of the three `DATASET = ...` lines (or add and select a `'custom'` entry).
4. In **Section 1b**, edit `IMMOBILE`, `MOBILE`, and `PAIRED` if you want non-default choices.
5. Run all cells. The full pipeline (including the 1,000-draw null loop) finishes in roughly 20–30 seconds on a recent laptop.
