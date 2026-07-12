"""
Q3b: Load, trim, convolve, and normalize PHOENIX model spectra.

Reuses the same continuum-normalisation approach as the observed spectrum
(Q2b) so model and observed profiles are treated identically.
"""
import os
import numpy as np
from astropy.io import fits
from astropy.convolution import Gaussian1DKernel, convolve

import sys
sys.path.append(os.path.dirname(__file__))
from load_observed_spectrum import normalize_continuum  # reuse Q2b logic

GRID_DIR = "../data/phoenix_grid"
GRID_NAME = "PHOENIX-ACES-AGSS-COND-2011"

_WAVE_CACHE = None  # PHOENIX wavelength axis is shared across all grid files


def _feh_suffix(feh):
    """PHOENIX archive uses '-0.0' (not '+0.0') for solar metallicity filenames."""
    if feh == 0.0:
        return "-0.0"
    return f"{feh:+.1f}"


def phoenix_filename(teff, logg, feh):
    return f"lte{teff:05d}-{logg:.2f}{_feh_suffix(feh)}.{GRID_NAME}-HiRes.fits"


def load_wave():
    """Load (and cache) the shared PHOENIX wavelength axis, in Angstrom."""
    global _WAVE_CACHE
    if _WAVE_CACHE is None:
        path = os.path.join(GRID_DIR, f"WAVE_{GRID_NAME}.fits")
        _WAVE_CACHE = fits.getdata(path).astype(float)
    return _WAVE_CACHE


def load_raw_model(teff, logg, feh):
    """Load a single PHOENIX flux array (full 500-26000 A range)."""
    path = os.path.join(GRID_DIR, phoenix_filename(teff, logg, feh))
    flux = fits.getdata(path).astype(float)
    wave = load_wave()
    return wave, flux


def convolve_to_resolution(wave, flux, R_obs, lam_ref=4861.0):
    """
    Convolve a high-resolution model spectrum to the observed resolving
    power R_obs using a Gaussian kernel, following the assignment's Q3b
    hint (sigma_lambda = lambda / (2.355 * R)).
    """
    sigma_lambda = lam_ref / (2.355 * R_obs)
    pixel_scale = np.median(np.diff(wave))  # PHOENIX HiRes: ~0.01 A/pixel
    sigma_pix = sigma_lambda / pixel_scale
    kernel = Gaussian1DKernel(stddev=sigma_pix)
    return convolve(flux, kernel, boundary="extend")


def get_model_hbeta(teff, logg, feh, R_obs=36800,
                     trim_region=(4700, 5000),
                     norm_region=(4750, 4950),
                     cont_windows=((4750, 4810), (4920, 4950))):
    """
    Full Q3b pipeline for one grid point: load -> trim -> convolve ->
    continuum-normalize. Returns the normalized Hbeta region ready for
    Sersic fitting (Part 3) or chi-squared comparison (Part 4).

    Returns
    -------
    wave_norm : ndarray, wavelengths within norm_region
    flux_norm : ndarray, normalized flux within norm_region
    """
    wave, flux = load_raw_model(teff, logg, feh)

    mask = (wave >= trim_region[0]) & (wave <= trim_region[1])
    w_trim, f_trim = wave[mask], flux[mask]

    f_conv = convolve_to_resolution(w_trim, f_trim, R_obs)

    w_norm, f_norm, _ = normalize_continuum(
        w_trim, f_conv, region=norm_region, cont_windows=cont_windows
    )
    return w_norm, f_norm


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Representative model requested in Q3b: Teff=5700K (nearest valid grid
    # point to the assignment's suggested 5750K -- see Q3a note), logg=4.5, [Fe/H]=0.0
    w, f = get_model_hbeta(5700, 4.5, 0.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(w, f, color="black", lw=0.8)
    ax.axhline(1.0, color="gray", lw=0.8, linestyle=":")
    ax.axvline(4861.33, color="red", linestyle="--", lw=1, label=r"H$\beta$ rest")
    ax.set_xlabel("Wavelength (Angstrom)")
    ax.set_ylabel("Normalized flux")
    ax.set_title(r"PHOENIX model: $T_{eff}=5700$K, $\log g=4.5$, [Fe/H]=0.0 "
                 r"(convolved to $R\approx36800$)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../figures/q3b_representative_model.png", dpi=150)
    plt.show()
    