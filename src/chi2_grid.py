"""
Part 4: Determine best-fit (Teff, log g, [Fe/H]) by minimizing chi-squared
between the observed and synthetic normalized Hbeta profiles over the
4820-4900 A window (full point-by-point profile comparison, NOT just the
Sersic-fit parameters from Part 3).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from load_observed_spectrum import load_elodie_spectrum, correct_to_rest_frame, normalize_continuum
from model_loader import get_model_hbeta

HBETA = 4861.33
FIT_LO, FIT_HI = 4820, 4900

TEFF_VALUES = [5000, 5200, 5500, 5700, 6000, 6200, 6500, 6700, 7000]
LOGG_VALUES = [3.5, 4.0, 4.5, 5.0]
FEH_VALUES = [-0.5, 0.0, 0.5]


def load_observed_profile(filepath="../data/observed/elodie_19970821_0032.fits",
                           v_r=-16.68):
    """Reproduce the Q2 pipeline: load, RV-correct, continuum-normalize."""
    wave, flux, header = load_elodie_spectrum(filepath)
    v_total = v_r - header["BERV"]
    wave_rest = correct_to_rest_frame(wave, v_total)
    w_norm, f_norm, _ = normalize_continuum(wave_rest, flux)

    mask = (w_norm >= FIT_LO) & (w_norm <= FIT_HI)
    w_fit, f_fit = w_norm[mask], f_norm[mask]

    # sigma from continuum RMS (same approach as sersic_fit.py Q5)
    cont_mask = ((w_norm >= 4750) & (w_norm <= 4810)) | ((w_norm >= 4920) & (w_norm <= 4950))
    sigma_val = np.std(f_norm[cont_mask])
    sigma = np.full_like(f_fit, sigma_val)
    return w_fit, f_fit, sigma


def chi_square_grid(w_obs, f_obs, sigma, R_obs=36800):
    """
    Compute raw chi-squared for every grid point by interpolating each
    model onto the observed wavelength grid and comparing point-by-point.
    """
    rows = []
    combos = [(t, g, f) for t in TEFF_VALUES for g in LOGG_VALUES for f in FEH_VALUES]
    for i, (teff, logg, feh) in enumerate(combos, 1):
        try:
            w_mod, f_mod = get_model_hbeta(teff, logg, feh, R_obs=R_obs,
                                            norm_region=(FIT_LO - 30, FIT_HI + 30))
            f_mod_interp = np.interp(w_obs, w_mod, f_mod)
            chi2 = np.sum(((f_obs - f_mod_interp) / sigma) ** 2)
            rows.append(dict(teff=teff, logg=logg, feh=feh, chi2=chi2))
            print(f"[{i}/{len(combos)}] Teff={teff} logg={logg} FeH={feh:+.1f}  chi2={chi2:.2f}")
        except Exception as e:
            print(f"[{i}/{len(combos)}] Teff={teff} logg={logg} FeH={feh:+.1f}  FAILED: {e}")
    return pd.DataFrame(rows)


def marginalized_profile(df, param):
    """
    Q7b: for each unique value of `param`, take the MINIMUM chi-squared
    over all other grid dimensions (profile likelihood / minimization,
    which is what the assignment's 'marginalising' means in a grid-search
    context).
    """
    return df.groupby(param)["chi2"].min().reset_index()


def parabolic_refine(df, param, chi2_min_row):
    """
    Q7c: fit a parabola to the 3-5 grid points nearest the minimum along
    one axis (holding the other two parameters at their best-fit grid
    values) and find the analytic minimum.
    """
    other_params = [p for p in ["teff", "logg", "feh"] if p != param]
    fixed = {p: chi2_min_row[p] for p in other_params}
    sub = df
    for p, v in fixed.items():
        sub = sub[np.isclose(sub[p], v)]
    sub = sub.sort_values(param)

    x = sub[param].values
    y = sub["chi2"].values
    if len(x) < 3:
        return chi2_min_row[param]  # not enough points to refine

    coeffs = np.polyfit(x, y, 2)
    if coeffs[0] <= 0:  # not a valid minimum (upward parabola required)
        return chi2_min_row[param]
    x_min = -coeffs[1] / (2 * coeffs[0])
    return x_min


def plot_delta_chi2(df, param, chi2_min, xlabel):
    prof = marginalized_profile(df, param)
    delta = prof["chi2"] - chi2_min

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(prof[param], delta, "o-", color="black")
    ax.axhline(1, color="tab:blue", linestyle="--", lw=1, label=r"$\Delta\chi^2=1$ (1$\sigma$)")
    ax.axhline(4, color="tab:red", linestyle="--", lw=1, label=r"$\Delta\chi^2=4$ (2$\sigma$)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\Delta\chi^2$")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"../figures/q7b_delta_chi2_{param}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    print("Loading observed Hbeta profile (4820-4900 A)...")
    w_obs, f_obs, sigma = load_observed_profile()

    print("\nComputing chi-squared over the full PHOENIX grid...")
    df = chi_square_grid(w_obs, f_obs, sigma)
    df.to_csv("../data/chi2_grid.csv", index=False)

    # --- Q7a: grid minimum ---
    best_row = df.loc[df["chi2"].idxmin()]
    dof = len(w_obs) - 3  # 3 free parameters (Teff, logg, FeH) at the grid level
    chi2_min = best_row["chi2"]
    chi2_red_min = chi2_min / dof
    print(f"\n--- Q7a: Grid minimum ---")
    print(f"Best-fit grid point: Teff={best_row['teff']:.0f} K, "
          f"log g={best_row['logg']:.2f}, [Fe/H]={best_row['feh']:+.1f}")
    print(f"chi2_min = {chi2_min:.2f}, reduced chi2 = {chi2_red_min:.3f} (dof={dof})")

    # Plot best-fit model over observed
    w_best, f_best = get_model_hbeta(int(best_row["teff"]), best_row["logg"], best_row["feh"],
                                       norm_region=(FIT_LO - 30, FIT_HI + 30))
    f_best_interp = np.interp(w_obs, w_best, f_best)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(w_obs, f_obs, "k.", ms=3, label="Observed")
    ax.plot(w_obs, f_best_interp, color="tab:red", lw=1.5,
            label=f"Best-fit model (Teff={best_row['teff']:.0f}, "
                  f"logg={best_row['logg']:.1f}, [Fe/H]={best_row['feh']:+.1f})")
    ax.axvline(HBETA, color="gray", linestyle="--", lw=1)
    ax.set_xlabel("Wavelength (Angstrom)")
    ax.set_ylabel("Normalized flux")
    ax.set_title(r"Q7a: Best-fit PHOENIX model vs observed H$\beta$")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../figures/q7a_best_fit_profile.png", dpi=150)
    plt.show()

    # --- Q7b: 1D delta-chi2 projections ---
    print("\n--- Q7b: 1D delta-chi2 projections ---")
    plot_delta_chi2(df, "teff", chi2_min, r"$T_{eff}$ (K)")
    plot_delta_chi2(df, "logg", chi2_min, r"$\log g$ (dex)")
    plot_delta_chi2(df, "feh", chi2_min, r"[Fe/H] (dex)")

    # --- Q7c: parabolic refinement ---
    print("\n--- Q7c: Parabolic refinement ---")
    teff_refined = parabolic_refine(df, "teff", best_row)
    logg_refined = parabolic_refine(df, "logg", best_row)
    feh_refined = parabolic_refine(df, "feh", best_row)
    print(f"Grid minimum:    Teff={best_row['teff']:.0f} K, logg={best_row['logg']:.2f}, FeH={best_row['feh']:+.2f}")
    print(f"Refined minimum: Teff={teff_refined:.1f} K, logg={logg_refined:.3f}, FeH={feh_refined:+.3f}")

    # --- Q8a: formal 1-sigma uncertainties from delta-chi2=1 ---
    print("\n--- Q8a: Formal 1-sigma uncertainties (delta-chi2=1 crossing) ---")

    def find_1sigma_bounds(df, param, chi2_min):
        prof = marginalized_profile(df, param).sort_values(param)
        delta = prof["chi2"].values - chi2_min
        x = prof[param].values
        below = x[delta <= 1]
        if len(below) == 0:
            return None, None
        return below.min(), below.max()

    for param, label in [("teff", "Teff (K)"), ("logg", "log g (dex)"), ("feh", "[Fe/H] (dex)")]:
        lo, hi = find_1sigma_bounds(df, param, chi2_min)
        best_val = best_row[param]
        print(f"{label}: {best_val} (68% CI from grid spacing: [{lo}, {hi}] "
              f"-- NOTE grid-limited resolution, refine by eye/interpolation if too coarse)")

    print("\nDone. See ../data/chi2_grid.csv for the full table used in Q7/Q8.")
