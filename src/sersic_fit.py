"""
Part 3: Sersic profile fitting.

F(lambda) = 1 - A * exp( -(|lambda - lambda0| / b)^n )

Q5: fit to the observed Hbeta profile.
Q6: fit to every model spectrum in the PHOENIX grid, storing results in a
    table for use in Part 4 (chi-squared grid search) and the Q6b heatmap.
"""
import os
import sys
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gamma
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from model_loader import get_model_hbeta

HBETA = 4861.33

TEFF_VALUES = [5000, 5200, 5500, 5700, 6000, 6200, 6500, 6700, 7000]  # matches download_phoenix.py
LOGG_VALUES = [3.5, 4.0, 4.5, 5.0]
FEH_VALUES = [-0.5, 0.0, 0.5]


def sersic(wavelength, A, lambda0, b, n):
    """Sersic-type absorption line profile (Eq. 1 in the assignment)."""
    return 1.0 - A * np.exp(-(np.abs(wavelength - lambda0) / b) ** n)


def fit_sersic(wave, flux, sigma=None, p0=None, bounds=None):
    """
    Fit the Sersic profile to a normalized absorption line.

    Parameters
    ----------
    wave, flux : ndarray, normalized profile to fit
    sigma : ndarray or None, per-pixel uncertainty (for weighted fit / proper
            covariance scaling). If None, an unweighted fit is performed.
    p0 : tuple or None, initial guess (A, lambda0, b, n). Defaults chosen to
         be reasonable for a Balmer line: depth ~0.4, centered near Hbeta,
         width ~3 A, shape ~2 (Gaussian-like).
    bounds : tuple or None, (lower, upper) bounds for (A, lambda0, b, n).
             n is bounded away from 0 to avoid singular/blow-up behaviour.

    Returns
    -------
    popt : ndarray, best-fit (A, lambda0, b, n)
    perr : ndarray, 1-sigma uncertainties (sqrt of covariance diagonal)
    pcov : ndarray, full covariance matrix
    """
    if p0 is None:
        p0 = (0.5, HBETA, 1.5, 1.0)
    if bounds is None:
        bounds = ([0.01, HBETA - 5, 0.05, 0.1], [1.0, HBETA + 5, 12.0, 8.0])

    popt, pcov = curve_fit(
        sersic, wave, flux, p0=p0, sigma=sigma,
        bounds=bounds, absolute_sigma=(sigma is not None), maxfev=20000,
    )
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, pcov


def equivalent_width_numeric(wave, flux):
    """Numerical integration of (1 - F) over the profile. Returns EW in Angstrom."""
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.0 renamed trapz
    return trapz_fn(1.0 - flux, wave)


def equivalent_width_analytic(A, b, n):
    """Analytic EW from fitted Sersic parameters (Eq. given in assignment)."""
    return A * b * gamma(1.0 + 1.0 / n) * 2.0


def reduced_chi_square(flux_obs, flux_model, sigma, n_params=4):
    resid = (flux_obs - flux_model) / sigma
    chi2 = np.sum(resid ** 2)
    dof = len(flux_obs) - n_params
    return chi2 / dof


def fit_full_grid(R_obs=36800):
    """
    Q6a: fit the Sersic profile to every PHOENIX grid spectrum.
    Requires the grid to already be downloaded (download_phoenix.py).

    Returns
    -------
    pandas.DataFrame with columns: teff, logg, feh, A, lambda0, b, n,
    EW_numeric, EW_analytic, and the (wave, flux) arrays are NOT stored
    here (recomputed on demand in Part 4) to keep the table lightweight.
    """
    rows = []
    combos = [(t, g, f) for t in TEFF_VALUES for g in LOGG_VALUES for f in FEH_VALUES]
    for i, (teff, logg, feh) in enumerate(combos, 1):
        try:
            w, flux = get_model_hbeta(teff, logg, feh, R_obs=R_obs)
            mask = (w >= 4849) & (w <= 4874)  # isolate Hbeta, same window as Q5
            w_narrow, flux_narrow = w[mask], flux[mask]
            popt, perr, _ = fit_sersic(w_narrow, flux_narrow)
            A, lambda0, b, n = popt
            ew_num = equivalent_width_numeric(w_narrow, flux_narrow)
            ew_ana = equivalent_width_analytic(A, b, n)
            rows.append(dict(teff=teff, logg=logg, feh=feh,
                              A=A, lambda0=lambda0, b=b, n=n,
                              A_err=perr[0], lambda0_err=perr[1],
                              b_err=perr[2], n_err=perr[3],
                              EW_numeric=ew_num, EW_analytic=ew_ana))
            print(f"[{i}/{len(combos)}] Teff={teff} logg={logg} FeH={feh:+.1f}  "
                  f"-> A={A:.3f} b={b:.3f} n={n:.2f} EW={ew_num:.3f} A")
        except Exception as e:
            print(f"[{i}/{len(combos)}] Teff={teff} logg={logg} FeH={feh:+.1f}  FAILED: {e}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from load_observed_spectrum import load_elodie_spectrum, correct_to_rest_frame, normalize_continuum

    # --- Q5: fit the observed profile ---
    # EDIT this to your actual observed file
    filepath = "../data/observed/elodie_19970821_0032.fits"
    wave, flux, header = load_elodie_spectrum(filepath)
    V_TOTAL = -16.68 - header["BERV"]
    wave_rest = correct_to_rest_frame(wave, V_TOTAL)
    w_obs, f_obs, _ = normalize_continuum(wave_rest, flux)

    # Restrict to a window that isolates Hbeta's own wings from neighboring
    # metal lines (a single 4-parameter Sersic cannot fit the whole crowded
    # 4820-4900 A region at once -- that wider window is reserved for the
    # Part 4 chi-squared GRID comparison, which compares full profiles
    # point-by-point rather than fitting a single line shape).
    mask = (w_obs >= 4849) & (w_obs <= 4874)
    w_fit, f_fit = w_obs[mask], f_obs[mask]

    # Simple constant sigma from continuum RMS (refine in Q2c if you built
    # a wavelength-dependent sigma array there instead)
    cont_mask = ((w_obs >= 4750) & (w_obs <= 4810)) | ((w_obs >= 4920) & (w_obs <= 4950))
    sigma_val = np.std(f_obs[cont_mask])
    sigma_arr = np.full_like(f_fit, sigma_val)

    popt, perr, pcov = fit_sersic(w_fit, f_fit, sigma=sigma_arr)
    A, lambda0, b, n = popt
    print("\n--- Q5: Observed Hbeta Sersic fit ---")
    print(f"A       = {A:.4f} +/- {perr[0]:.4f}")
    print(f"lambda0 = {lambda0:.4f} +/- {perr[1]:.4f} A")
    print(f"b       = {b:.4f} +/- {perr[2]:.4f} A")
    print(f"n       = {n:.4f} +/- {perr[3]:.4f}")

    f_model = sersic(w_fit, *popt)
    chi2_red = reduced_chi_square(f_fit, f_model, sigma_arr)
    print(f"Reduced chi-squared = {chi2_red:.3f}")

    ew_num = equivalent_width_numeric(w_fit, f_fit)
    ew_ana = equivalent_width_analytic(A, b, n)
    print(f"EW (numeric)  = {ew_num:.4f} A")
    print(f"EW (analytic) = {ew_ana:.4f} A")

    # Plot: fit + residuals
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(w_fit, f_fit, "k.", ms=3, label="Observed")
    ax1.plot(w_fit, f_model, color="tab:red", lw=1.5, label="Sersic fit")
    ax1.axvline(HBETA, color="gray", linestyle="--", lw=1)
    ax1.set_ylabel("Normalized flux")
    ax1.legend()
    ax1.set_title(r"Q5: Sersic fit to observed H$\beta$ (tau Cet)")

    ax2.plot(w_fit, f_fit - f_model, "k.", ms=3)
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_xlabel("Wavelength (Angstrom)")
    ax2.set_ylabel("Residual")

    plt.tight_layout()
    plt.savefig("../figures/q5_sersic_fit_observed.png", dpi=150)
    plt.show()

    # --- Q6: fit the full model grid (requires PHOENIX grid downloaded) ---
    print("\n--- Q6: fitting full model grid (this will take a while) ---")
    grid_results = fit_full_grid()
    grid_results.to_csv("../data/sersic_grid_fits.csv", index=False)
    print(f"Saved {len(grid_results)} grid fits to ../data/sersic_grid_fits.csv")