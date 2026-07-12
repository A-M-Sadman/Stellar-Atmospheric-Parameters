"""
Q9: Estimate stellar radius and luminosity from the best-fit (Teff, log g),
assuming M ~ 1 Msun, and place the star on a theoretical H-R diagram against
the given ZAMS locus.
"""
import numpy as np
import matplotlib.pyplot as plt

G = 6.674e-8       # cgs: cm^3 g^-1 s^-2
SIGMA_SB = 5.670e-5  # cgs: erg cm^-2 s^-1 K^-4
M_SUN = 1.989e33   # g
R_SUN = 6.957e10   # cm
L_SUN = 3.828e33   # erg/s
TEFF_SUN = 5778.0  # K

# EDIT these to your actual Q7a/Q7c best-fit (or refined) values
TEFF_BEST = 6500.0      # K
LOGG_BEST = 5.00        # dex
M_ASSUMED = 1.0 * M_SUN  # g, assumed for an initial estimate


def radius_from_logg(teff, logg, mass=M_ASSUMED):
    g = 10 ** logg  # cm/s^2
    R = np.sqrt(G * mass / g)
    return R


def luminosity(teff, R):
    return 4 * np.pi * R**2 * SIGMA_SB * teff**4


if __name__ == "__main__":
    R_cm = radius_from_logg(TEFF_BEST, LOGG_BEST)
    R_sun = R_cm / R_SUN
    L_erg = luminosity(TEFF_BEST, R_cm)
    L_sun = L_erg / L_SUN

    print(f"Assuming M = 1 Msun:")
    print(f"R = {R_cm:.3e} cm = {R_sun:.3f} R_sun")
    print(f"L = {L_erg:.3e} erg/s = {L_sun:.3f} L_sun")
    print(f"log(L/Lsun) = {np.log10(L_sun):.3f}")
    print(f"log(Teff) = {np.log10(TEFF_BEST):.4f}")

    # ZAMS locus given in the assignment
    zams_teff = np.array([7200, 6000, 5778, 5300, 5000])
    zams_logL = np.array([0.8, 0.2, 0.0, -0.3, -0.5])
    zams_spec = ["F2 V", "G0 V", "G2 V (Sun)", "K1 V", "K5 V"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(np.log10(zams_teff), zams_logL, "o-", color="tab:blue", label="ZAMS (given)")
    for t, l, s in zip(zams_teff, zams_logL, zams_spec):
        ax.annotate(s, (np.log10(t), l), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.plot(np.log10(TEFF_BEST), np.log10(L_sun), "*", color="tab:red", markersize=20,
            label=f"tau Cet (this work): Teff={TEFF_BEST:.0f}K")

    ax.invert_xaxis()  # H-R diagrams run hot (left) to cool (right)
    ax.set_xlabel(r"$\log T_{eff}$ (K)")
    ax.set_ylabel(r"$\log(L/L_\odot)$")
    ax.set_title("H-R Diagram: tau Cet vs ZAMS")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("../figures/q9b_hr_diagram.png", dpi=150)
    plt.show()