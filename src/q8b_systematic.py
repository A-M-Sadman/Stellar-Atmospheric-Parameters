"""
Q8b: Systematic uncertainty from continuum placement. Repeat the chi-squared
grid search with the continuum windows shifted +10 A and -10 A, and compare
the recovered best-fit Teff/logg/FeH to the nominal result.
"""
import sys, os
import numpy as np
sys.path.append(os.path.dirname(__file__))
from chi2_grid import (load_observed_profile, chi_square_grid, TEFF_VALUES,
                        LOGG_VALUES, FEH_VALUES, FIT_LO, FIT_HI)
from load_observed_spectrum import load_elodie_spectrum, correct_to_rest_frame, normalize_continuum


def load_observed_profile_shifted(shift, filepath="../data/observed/elodie_19970821_0032.fits", v_r=-16.68):
    wave, flux, header = load_elodie_spectrum(filepath)
    v_total = v_r - header["BERV"]
    wave_rest = correct_to_rest_frame(wave, v_total)

    cont_windows = ((4750 + shift, 4810 + shift), (4920 + shift, 4950 + shift))
    w_norm, f_norm, _ = normalize_continuum(wave_rest, flux, cont_windows=cont_windows)

    mask = (w_norm >= FIT_LO) & (w_norm <= FIT_HI)
    w_fit, f_fit = w_norm[mask], f_norm[mask]

    cont_mask = np.zeros_like(w_norm, dtype=bool)
    for lo, hi in cont_windows:
        cont_mask |= (w_norm >= lo) & (w_norm <= hi)
    sigma_val = np.std(f_norm[cont_mask])
    sigma = np.full_like(f_fit, sigma_val)
    return w_fit, f_fit, sigma


def best_fit_for_shift(shift):
    print(f"\n=== Continuum shift: {shift:+d} A ===")
    w, f, sigma = load_observed_profile_shifted(shift)
    df = chi_square_grid(w, f, sigma)
    best = df.loc[df["chi2"].idxmin()]
    print(f"Best fit: Teff={best['teff']:.0f} K, logg={best['logg']:.2f}, FeH={best['feh']:+.1f}")
    return best


if __name__ == "__main__":
    best_plus = best_fit_for_shift(+10)
    best_minus = best_fit_for_shift(-10)

    print("\n--- Q8b summary ---")
    print(f"+10 A shift: Teff={best_plus['teff']:.0f}, logg={best_plus['logg']:.2f}, FeH={best_plus['feh']:+.1f}")
    print(f"-10 A shift: Teff={best_minus['teff']:.0f}, logg={best_minus['logg']:.2f}, FeH={best_minus['feh']:+.1f}")
    print("Compare these to the nominal Q7a result to quote the systematic uncertainty.")