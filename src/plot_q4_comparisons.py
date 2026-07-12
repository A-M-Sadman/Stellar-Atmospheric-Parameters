"""
Q4: Overlay plots showing how Teff, log g, and [Fe/H] individually affect
the normalized Hbeta profile shape. Requires the PHOENIX grid to be
downloaded first (download_phoenix.py) and model_loader.py in the same
directory.
"""
import matplotlib.pyplot as plt
from model_loader import get_model_hbeta

HBETA = 4861.33


def plot_teff_comparison():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for teff, color in zip([5500, 6000, 6500], ["tab:blue", "tab:green", "tab:red"]):
        w, f = get_model_hbeta(teff, 4.5, 0.0)
        ax.plot(w, f, color=color, lw=0.9, label=f"Teff={teff} K")
    ax.axvline(HBETA, color="gray", linestyle="--", lw=1)
    ax.set_xlim(4820, 4900)
    ax.set_xlabel("Wavelength (Angstrom)")
    ax.set_ylabel("Normalized flux")
    ax.set_title(r"H$\beta$ vs $T_{eff}$ (log g=4.5, [Fe/H]=0.0)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../figures/q4a_teff_comparison.png", dpi=150)
    plt.show()


def plot_logg_comparison():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for logg, color in zip([3.5, 4.0, 4.5], ["tab:blue", "tab:green", "tab:red"]):
        w, f = get_model_hbeta(6000, logg, 0.0)
        ax.plot(w, f, color=color, lw=0.9, label=f"log g={logg}")
    ax.axvline(HBETA, color="gray", linestyle="--", lw=1)
    ax.set_xlim(4820, 4900)
    ax.set_xlabel("Wavelength (Angstrom)")
    ax.set_ylabel("Normalized flux")
    ax.set_title(r"H$\beta$ vs $\log g$ (Teff=6000K, [Fe/H]=0.0)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../figures/q4b_logg_comparison.png", dpi=150)
    plt.show()


def plot_feh_comparison():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for feh, color in zip([-0.5, 0.0, 0.5], ["tab:blue", "tab:green", "tab:red"]):
        w, f = get_model_hbeta(5700, 4.5, feh)
        ax.plot(w, f, color=color, lw=0.9, label=f"[Fe/H]={feh:+.1f}")
    ax.axvline(HBETA, color="gray", linestyle="--", lw=1)
    ax.set_xlim(4800, 4900)
    ax.set_xlabel("Wavelength (Angstrom)")
    ax.set_ylabel("Normalized flux")
    ax.set_title(r"H$\beta$ region vs [Fe/H] (Teff=5700K, log g=4.5)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../figures/q4c_feh_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    plot_teff_comparison()
    plot_logg_comparison()
    plot_feh_comparison()