"""
Standalone check: prints the exact sigma (per-pixel uncertainty) value used
in Q2c and reused in Q5/Q7's chi-squared calculations, so you can quote the
precise number in your report instead of an estimate.
"""
from load_observed_spectrum import load_elodie_spectrum, correct_to_rest_frame, normalize_continuum
import numpy as np

filepath = "../data/observed/elodie_19970821_0032.fits"
wave, flux, header = load_elodie_spectrum(filepath)
v_total = -16.68 - header["BERV"]
wave_rest = correct_to_rest_frame(wave, v_total)
w_norm, f_norm, _ = normalize_continuum(wave_rest, flux)

cont_mask = ((w_norm >= 4750) & (w_norm <= 4810)) | ((w_norm >= 4920) & (w_norm <= 4950))
sigma_val = np.std(f_norm[cont_mask])
print(f"sigma = {sigma_val:.4f}")