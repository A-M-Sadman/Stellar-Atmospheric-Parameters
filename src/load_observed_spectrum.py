"""
Q1: Load and plot the observed spectrum (ELODIE 'spec' format).

ELODIE 'spec' files store flux as a 1D array with a linear wavelength
solution encoded in the FITS header (CRVAL1, CDELT1, CRPIX1 or CD1_1).
"""
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

HBETA_REST = 4861.33  # Angstrom


def load_elodie_spectrum(filepath):
    """
    Load an ELODIE 'spec' format FITS file.

    Returns
    -------
    wave : ndarray, wavelength in Angstrom
    flux : ndarray, flux (arbitrary/normalized units depending on file)
    header : astropy FITS header (for SNR, resolution, etc.)
    """
    with fits.open(filepath) as hdul:
        flux = hdul[0].data.astype(float)
        header = hdul[0].header

    # Build wavelength array from the linear WCS keywords
    crval1 = header["CRVAL1"]
    cdelt1 = header.get("CDELT1", header.get("CD1_1"))
    crpix1 = header.get("CRPIX1", 1)
    n = len(flux)
    pix = np.arange(1, n + 1)
    wave = crval1 + (pix - crpix1) * cdelt1

    return wave, flux, header


C_KMS = 299792.458  # speed of light in km/s


def correct_to_rest_frame(wave, v_r_kms):
    """
    Shift an observed wavelength array to the stellar rest frame.

    Parameters
    ----------
    wave : ndarray, observed wavelength array (Angstrom)
    v_r_kms : float, star's heliocentric radial velocity (km/s),
              looked up from SIMBAD. Negative = approaching.

    Returns
    -------
    wave_rest : ndarray, wavelength array in the stellar rest frame
    """
    return wave / (1.0 + v_r_kms / C_KMS)


def normalize_continuum(wave, flux, region=(4750, 4950),
                         cont_windows=((4750, 4810), (4920, 4950)),
                         poly_order=2, sigma_clip=2.5, max_iters=5):
    """
    Continuum-normalize a spectral region around Hbeta.

    Fits a polynomial to flux inside the continuum windows, using iterative
    sigma-clipping to reject absorption-line pixels (which only pull flux
    DOWN, never up) so the fit tracks the true continuum level rather than
    being biased low by the line forest.

    Parameters
    ----------
    wave, flux : ndarray, full (rest-frame) spectrum
    region : tuple, (min, max) wavelength range to extract and return
    cont_windows : tuple of two (min, max) tuples, the line-free windows
                   used to fit the continuum
    poly_order : int, degree of the continuum polynomial (1 or 2)
    sigma_clip : float, reject points this many sigma BELOW the fit
                 (asymmetric: only low outliers are absorption lines/noise)
    max_iters : int, maximum sigma-clipping iterations

    Returns
    -------
    wave_region : ndarray, wavelengths within `region`
    flux_norm : ndarray, normalized flux (F / F_continuum) within `region`
    continuum_fit : ndarray, the fitted continuum evaluated across `region`
    """
    region_mask = (wave >= region[0]) & (wave <= region[1])
    w_region, f_region = wave[region_mask], flux[region_mask]

    cont_mask = np.zeros_like(wave, dtype=bool)
    for lo, hi in cont_windows:
        cont_mask |= (wave >= lo) & (wave <= hi)
    w_cont, f_cont = wave[cont_mask], flux[cont_mask]

    # Iterative sigma-clipping: fit, reject low outliers, refit
    keep = np.ones_like(f_cont, dtype=bool)
    for _ in range(max_iters):
        coeffs = np.polyfit(w_cont[keep], f_cont[keep], poly_order)
        fit_vals = np.polyval(coeffs, w_cont)
        resid = f_cont - fit_vals
        std = np.std(resid[keep])
        new_keep = resid > -sigma_clip * std  # only reject LOW outliers
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    continuum_fit = np.polyval(coeffs, w_region)
    flux_norm = f_region / continuum_fit
    return w_region, flux_norm, continuum_fit


def estimate_snr(wave, flux, window=(4750, 4810), poly_order=1):
    """
    Estimate SNR near Hbeta from a continuum window.

    Detrends the window with a low-order polynomial first, so the RMS
    reflects pixel-to-pixel noise rather than continuum slope or
    residual line curvature within the window.
    """
    mask = (wave >= window[0]) & (wave <= window[1])
    w, f = wave[mask], flux[mask]

    coeffs = np.polyfit(w, f, poly_order)
    trend = np.polyval(coeffs, w)
    residual = f - trend

    mean_level = np.mean(trend)
    rms = np.std(residual)
    return mean_level / rms


if __name__ == "__main__":
    filepath = "../data/observed/elodie_19970821_0032.fits"

    wave, flux, header = load_elodie_spectrum(filepath)

    print(f"Wavelength coverage: {wave.min():.1f}-{wave.max():.1f} A")
    print(f"Number of pixels: {len(wave)}")
    print(f"Resolution R (from header, if present): {header.get('CRDER1', 'not in header — see ELODIE docs, R~42000 nominal')}")
    snr = estimate_snr(wave, flux)
    print(f"Estimated SNR near Hbeta continuum: {snr:.1f}")

    # --- Q2a: radial velocity correction ---
    # Combined correction: stellar RV (SIMBAD) minus BERV (Earth's line-of-sight
    # velocity, from header). The ELODIE 'spec' wavelength axis is topocentric,
    # so Earth's motion must be removed in addition to the star's own motion.
    # Verified empirically: v_r - BERV places the Hbeta core at 4861.34 A
    # (target 4861.33 A) -- see report Q2a for the diagnostic that established this.
    V_R_TAUCET = -16.68     # km/s, stellar heliocentric RV from SIMBAD (HD 10700)
    BERV = header["BERV"]   # km/s, from FITS header
    V_TOTAL = V_R_TAUCET - BERV

    wave_rest = correct_to_rest_frame(wave, V_TOTAL)

    # Verify: parabolic fit to the flux minimum near Hbeta
    mask_hbeta = (wave_rest > 4855) & (wave_rest < 4868)
    w_h, f_h = wave_rest[mask_hbeta], flux[mask_hbeta]
    idx_min = np.argmin(f_h)
    lo, hi = max(idx_min - 5, 0), min(idx_min + 6, len(w_h))
    w_win, f_win = w_h[lo:hi], f_h[lo:hi]
    coeffs = np.polyfit(w_win, f_win, 2)
    lambda_core = -coeffs[1] / (2 * coeffs[0])
    print(f"Hbeta core (rest frame, parabolic fit) found at: {lambda_core:.2f} A  (target: {HBETA_REST} A)")

    # --- Q2b: continuum normalization ---
    w_region, flux_norm, continuum_fit = normalize_continuum(wave_rest, flux)

    # Diagnostic plot: raw flux with fitted continuum overlaid, plus the
    # continuum windows highlighted, so you can visually check the fit quality.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(w_region, flux[(wave_rest >= 4750) & (wave_rest <= 4950)],
              color="black", lw=0.6, label="Observed flux")
    ax1.plot(w_region, continuum_fit, color="tab:orange", lw=1.5, label="Fitted continuum")
    for lo, hi in [(4750, 4810), (4920, 4950)]:
        ax1.axvspan(lo, hi, color="tab:green", alpha=0.15)
    ax1.axvline(HBETA_REST, color="red", linestyle="--", lw=1)
    ax1.set_ylabel("Flux (arbitrary units)")
    ax1.legend()
    ax1.set_title(r"Continuum fit around H$\beta$ (green = fit windows)")

    ax2.plot(w_region, flux_norm, color="black", lw=0.7)
    ax2.axhline(1.0, color="gray", lw=0.8, linestyle=":")
    ax2.axvline(HBETA_REST, color="red", linestyle="--", lw=1, label=r"H$\beta$ rest")
    ax2.set_xlabel("Wavelength (Angstrom, rest frame)")
    ax2.set_ylabel("Normalized flux")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("../figures/q2b_normalized_hbeta.png", dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wave, flux, lw=0.5, color="black")
    ax.axvline(HBETA_REST, color="red", linestyle="--", label=r"H$\beta$ (4861.33 $\AA$)")
    ax.set_xlabel("Wavelength (Angstrom)")
    ax.set_ylabel("Flux (arbitrary units)")
    ax.set_title("Observed spectrum: tau Cet (HD 10700)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../figures/q1c_full_spectrum.png", dpi=150)
    plt.show()