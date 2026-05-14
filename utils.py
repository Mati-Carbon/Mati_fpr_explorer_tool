"""
Helpers for the Bayesian false-positive framework on paired or unpaired
soil-chemistry data.

This module is the minimal set of functions exercised by
`fpr_explorer_on_real_data.ipynb`:

    bootstrap_delta_E(...)   bootstrap distribution of mean(eoy) − mean(pre)
    FastKDE(...)             gaussian_kde wrapper with precomputed grid lookup
    compute_posterior(...)   joint posterior p(r, τ | ΔI, ΔX) on a 2-D grid
    marginalize(...)         1-D marginals p(r) and p(τ)
    summarize_marginal(...)  MAP / median / 68% / 95% CIs from a 1-D marginal
    joint_map(...)           (r, τ) at the 2-D posterior peak

Self-contained: depends only on numpy and scipy.stats.gaussian_kde. No file
paths, basalt source dictionaries, deployment-specific constants, spatial
joins, or sensitivity-scenario plumbing — those lived in a larger version
of this module used for the original GPM pipeline.
"""

import numpy as np
from scipy.stats import gaussian_kde


# ============================================================================
# ΔE bootstrap
# ============================================================================

def bootstrap_delta_E(x_pre, x_eoy, B=10000, seed=None, sigma_analytical=None):
    """Bootstrap pre- and eoy-population means independently, then form the
    outer-difference distribution ΔE = mean(eoy_boot) − mean(pre_boot).

    Treats pre and eoy as independent populations, which is valid for both
    paired and unpaired data. (When data are paired, the row-permutation
    upstream — sign-flip per row — is what carries the pairing structure into
    the null; this function just bootstraps the resulting two columns.)

    Args:
        x_pre, x_eoy:    1-D arrays. NaNs ignored.
        B:               number of bootstrap resamples per side.
        seed:            optional RNG seed (int) for reproducibility.
        sigma_analytical: optional per-measurement analytical SD (float).
                         If given, each resampled observation gets an
                         independent N(0, σ) perturbation before averaging.
                         Propagates measurement noise through the bootstrap.

    Returns:
        dict with keys 'delta_boot' (1-D array of length B), 'mean_pre_boot',
        'mean_eoy_boot', 'n_pre', 'n_eoy'. Returns None if either side is empty.
    """
    rng = np.random.default_rng(seed)
    x_pre = np.asarray(x_pre, dtype=float)
    x_eoy = np.asarray(x_eoy, dtype=float)
    x_pre_c = x_pre[~np.isnan(x_pre)]
    x_eoy_c = x_eoy[~np.isnan(x_eoy)]
    n_pre, n_eoy = len(x_pre_c), len(x_eoy_c)
    if n_pre == 0 or n_eoy == 0:
        return None

    # Bootstrap the pre-population mean B times
    idx_pre = rng.integers(0, n_pre, size=(B, n_pre))
    pre_resamples = x_pre_c[idx_pre]                       # (B, n_pre)
    if sigma_analytical is not None and sigma_analytical > 0:
        pre_resamples = pre_resamples + rng.normal(0.0, sigma_analytical,
                                                   size=pre_resamples.shape)
    mean_pre_boot = pre_resamples.mean(axis=1)             # (B,)

    # Bootstrap the eoy-population mean B times
    idx_eoy = rng.integers(0, n_eoy, size=(B, n_eoy))
    eoy_resamples = x_eoy_c[idx_eoy]
    if sigma_analytical is not None and sigma_analytical > 0:
        eoy_resamples = eoy_resamples + rng.normal(0.0, sigma_analytical,
                                                   size=eoy_resamples.shape)
    mean_eoy_boot = eoy_resamples.mean(axis=1)

    # Form ΔE by randomly re-pairing the two bootstrap distributions. This
    # samples B values from the convolution of the two sampling distributions,
    # which is the correct distribution for ΔE under the independent-population
    # assumption.
    b_idx = rng.integers(0, B, size=B)
    c_idx = rng.integers(0, B, size=B)
    delta_boot = mean_eoy_boot[c_idx] - mean_pre_boot[b_idx]

    return {
        'mean_pre_boot': mean_pre_boot,
        'mean_eoy_boot': mean_eoy_boot,
        'delta_boot':    delta_boot,
        'n_pre':         n_pre,
        'n_eoy':         n_eoy,
    }


# ============================================================================
# Fast KDE with precomputed grid
# ============================================================================

class FastKDE:
    """A scipy.stats.gaussian_kde wrapper with precomputed density on a 1-D
    grid. Queries via np.interp, which is ~100× faster than repeated
    kde.logpdf calls — relevant when the same KDE is evaluated at every cell
    of a 2-D parameter grid.

    Args:
        samples:   1-D array of samples to fit.
        n_grid:    number of points in the precomputed grid.
        pad:       how far past min/max to extend the grid, in units of
                   (max - min). Wider pad → safer extrapolation, slightly
                   slower interp.
        bw_method: passed through to gaussian_kde (default: Silverman).

    Use:
        kde = FastKDE(samples)
        kde.pdf(x)      # interpolated density at x (scalar or array)
        kde.logpdf(x)   # log-density; floor at log(1e-300) outside the grid
    """

    def __init__(self, samples, n_grid=500, pad=5.0, bw_method=None):
        samples = np.asarray(samples)
        self.kde = gaussian_kde(samples, bw_method=bw_method)
        lo = samples.min()
        hi = samples.max()
        rng = hi - lo
        if rng == 0:
            rng = abs(lo) + 1.0
        self.grid = np.linspace(lo - pad * rng, hi + pad * rng, n_grid)
        pdf = self.kde(self.grid)
        # Floor the density to avoid log(0) downstream. 1e-300 is just above
        # the underflow boundary for double precision.
        self._pdf_floor = max(float(pdf.max()) * 1e-300, 1e-300)
        pdf = np.maximum(pdf, self._pdf_floor)
        self.pdf_grid = pdf
        self.log_pdf_grid = np.log(pdf)
        self._log_floor = float(np.log(self._pdf_floor))

    def pdf(self, x):
        x = np.asarray(x)
        return np.interp(x, self.grid, self.pdf_grid,
                         left=self._pdf_floor, right=self._pdf_floor)

    def logpdf(self, x):
        x = np.asarray(x)
        return np.interp(x, self.grid, self.log_pdf_grid,
                         left=self._log_floor, right=self._log_floor)


# ============================================================================
# Bayesian inversion: joint posterior p(r, τ | ΔI, ΔX)
# ============================================================================

def compute_posterior(samples, bootstraps, immobile, mobile,
                      r_range, tau_range, n_grid,
                      aggregated_basalt, precomputed_kdes=None):
    """Joint posterior p(r, τ | data) on an n_grid × n_grid lattice.

    Forward model (soil-cation-dilution form):
        ΔI_pred = r · (I_b − I_s)
        ΔX_pred = r · ((1 − τ) · X_b − X_s)

    Likelihood = product over tracers of FastKDE log-densities evaluated at
    the predicted ΔE values, using each tracer's bootstrap distribution as
    the empirical observation model.

    Priors on r and τ are uniform over `r_range` and `tau_range` respectively.

    Args:
        samples:           DataFrame (or any dict-like indexed by column name
                           whose accessor returns something with .mean()).
                           Must contain f'{immobile}_pre' and f'{el}_pre' for
                           each `el` in `mobile`. Only these column means are
                           read — they set the pre-treatment soil concentrations
                           I_s and X_s. The '_eoy' columns were already used
                           upstream to build `bootstraps` and are not read here.
        bootstraps:        dict {element: result_dict_from_bootstrap_delta_E}.
                           Must contain entries for `immobile` and each member
                           of `mobile`.
        immobile:          element name (str), e.g. 'Ti'.
        mobile:            tuple of mobile element names, e.g. ('Ca',) or
                           ('Ca', 'Mg'). A single mobile tracer is the typical
                           credit-grade case.
        r_range:           (lo, hi) tuple — uniform prior bounds on r.
        tau_range:         (lo, hi) tuple — uniform prior bounds on τ.
        n_grid:            grid resolution per axis (160 is plenty for most
                           datasets; the explorer uses 80 with a cell-smearing
                           correction).
        aggregated_basalt: pd.Series or dict-like indexed by element name with
                           the basalt-endmember concentration (Cb) per element.
                           Required (no fallback) so the basalt-endmember
                           choice is always explicit in the caller.
        precomputed_kdes:  optional dict {element: FastKDE}, reuses a pre-built
                           KDE rather than rebuilding it from
                           bootstraps[el]['delta_boot']. Useful in tight null
                           loops where the same KDEs serve many inversions.

    Returns:
        (r_grid, tau_grid, post) where post is a normalized 2-D ndarray of
        shape (n_grid, n_grid) such that the trapezoidal integral over both
        axes equals 1.
    """
    # Pre-treatment soil concentrations (Cs)
    I_s = samples[f'{immobile}_pre'].mean()
    X_s_list = [samples[f'{el}_pre'].mean() for el in mobile]

    # Basalt endmember concentrations (Cb)
    I_b = aggregated_basalt[immobile]
    X_b_list = [aggregated_basalt[el] for el in mobile]

    # Get (or build) the per-element KDE that defines the observation likelihood
    if precomputed_kdes is None:
        precomputed_kdes = {}
    def get_kde(el):
        if el in precomputed_kdes:
            return precomputed_kdes[el]
        return FastKDE(bootstraps[el]['delta_boot'])
    kde_I = get_kde(immobile)
    kdes_X = [get_kde(el) for el in mobile]

    # Build the (r, τ) grid and the forward-model predictions
    r_grid = np.linspace(r_range[0], r_range[1], n_grid)
    tau_grid = np.linspace(tau_range[0], tau_range[1], n_grid)
    R, T = np.meshgrid(r_grid, tau_grid, indexing='ij')

    # Log-likelihood: immobile + each mobile tracer
    dI_pred = R * (I_b - I_s)
    log_lik = kde_I.logpdf(dI_pred)
    for kde_X_i, X_b, X_s in zip(kdes_X, X_b_list, X_s_list):
        dX_pred = R * ((1 - T) * X_b - X_s)
        log_lik = log_lik + kde_X_i.logpdf(dX_pred)

    # Normalize the exponentiated log-likelihood (uniform priors → drops out)
    log_post = log_lik - log_lik.max()
    post = np.exp(log_post)
    norm = np.trapezoid(np.trapezoid(post, x=tau_grid, axis=1), x=r_grid)
    post /= norm
    return r_grid, tau_grid, post


# ============================================================================
# Marginal extraction and summary statistics
# ============================================================================

def marginalize(r_grid, tau_grid, post):
    """Integrate the joint posterior over one axis to get each 1-D marginal,
    p(r) and p(τ). Both are renormalized to integrate to 1."""
    p_r = np.trapezoid(post, x=tau_grid, axis=1)
    p_tau = np.trapezoid(post, x=r_grid, axis=0)
    p_r = p_r / np.trapezoid(p_r, x=r_grid)
    p_tau = p_tau / np.trapezoid(p_tau, x=tau_grid)
    return p_r, p_tau


def summarize_marginal(grid, pdf):
    """MAP, median, 68% and 95% CIs for a 1-D marginal.

    MAP is the grid value at the highest-density cell. Percentiles are
    obtained by interpolating the CDF (cumulative sum of pdf, normalized
    to end at 1).
    """
    i_map = int(np.argmax(pdf))
    map_val = float(grid[i_map])
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    return {
        'MAP':     map_val,
        'median':  float(np.interp(0.5,   cdf, grid)),
        'ci68_lo': float(np.interp(0.16,  cdf, grid)),
        'ci68_hi': float(np.interp(0.84,  cdf, grid)),
        'ci95_lo': float(np.interp(0.025, cdf, grid)),
        'ci95_hi': float(np.interp(0.975, cdf, grid)),
    }


def joint_map(r_grid, tau_grid, post):
    """Location of the joint (2-D) maximum a posteriori. Returns (r, τ)."""
    i, j = np.unravel_index(np.argmax(post), post.shape)
    return float(r_grid[i]), float(tau_grid[j])
