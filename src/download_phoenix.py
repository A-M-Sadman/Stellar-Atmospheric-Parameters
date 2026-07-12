"""
Q3a: Download the PHOENIX-ACES HiResFITS grid.

Grid: Teff = 5000-7000 K, log g = 3.5-5.0 K (step 0.5), [Fe/H] = -0.5, 0.0, +0.5
9 x 4 x 3 = 108 files (~6 MB each, ~650 MB total) plus one shared wavelength
file (~13 MB).

IMPORTANT: the assignment specifies a 250 K Teff step (5000, 5250, 5500, ...).
However, the PHOENIX-ACES archive's native grid only has 100 K steps below
7000 K (Husser et al. 2013), so 5250/5750/6250/6750 K do not exist and any
download attempt for them 404s permanently, regardless of retries. These four
values have been substituted with the nearest available 100 K grid point
(5200/5700/6200/6700 K respectively) -- see Q3a discussion in the report.

Directory/filename convention (Husser et al. 2013, PHOENIX archive):
  https://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS/
      PHOENIX-ACES-AGSS-COND-2011/Z{FeH}/lte{Teff:05d}-{logg:.2f}{FeH:+.1f}.
      PHOENIX-ACES-AGSS-COND-2011-HiRes.fits

Note: the [Fe/H]=0.0 subdirectory is named "Z-0.0" (a quirk of the archive's
own naming, not a typo) while +0.5 and -0.5 use their natural signs. The
filename suffix for solar metallicity is ALSO "-0.0" (not "+0.0") per the
Husser et al. 2013 naming-scheme documentation.
"""
import os
import time
import requests

BASE_URL = "https://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS"
GRID_DIR = "PHOENIX-ACES-AGSS-COND-2011"
OUT_DIR = "../data/phoenix_grid"

# Substituted to the nearest valid 100K grid point where the assignment's
# 250K step lands off-grid (5250->5200, 5750->5700, 6250->6200, 6750->6700)
TEFF_VALUES = [5000, 5200, 5500, 5700, 6000, 6200, 6500, 6700, 7000]  # 9 values
LOGG_VALUES = [3.5, 4.0, 4.5, 5.0]                    # 4 values
FEH_VALUES = [-0.5, 0.0, 0.5]                         # 3 values


def feh_dirname(feh):
    """Z-0.0, Z-0.5, Z+0.5 -- note Z-0.0 (not Z+0.0) is the archive's own convention."""
    if feh == 0.0:
        return "Z-0.0"
    return f"Z{feh:+.1f}"


def feh_suffix(feh):
    """Filename suffix uses an explicit sign, e.g. -0.5, +0.5 -- EXCEPT solar
    metallicity, which the archive names '-0.0' (not '+0.0'), per the Husser
    et al. 2013 naming-scheme documentation."""
    if feh == 0.0:
        return "-0.0"
    return f"{feh:+.1f}"


def phoenix_filename(teff, logg, feh):
    return f"lte{teff:05d}-{logg:.2f}{feh_suffix(feh)}.{GRID_DIR}-HiRes.fits"


def download_file(url, dest_path, chunk_size=1 << 16, retries=5):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"  already have {os.path.basename(dest_path)}, skipping")
        return

    tmp_path = dest_path + ".part"
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, stream=True, timeout=90)
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
            os.replace(tmp_path, dest_path)  # atomic rename only on success
            return
        except requests.exceptions.RequestException as e:
            last_error = e
            if os.path.exists(tmp_path):
                os.remove(tmp_path)  # never leave a partial file lying around
            if attempt < retries:
                wait = 5 * attempt  # 5s, 10s, 15s, 20s backoff
                print(f"  attempt {attempt} failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
    raise last_error


def download_wave_file():
    os.makedirs(OUT_DIR, exist_ok=True)
    url = f"{BASE_URL}/WAVE_{GRID_DIR}.fits"
    dest = os.path.join(OUT_DIR, f"WAVE_{GRID_DIR}.fits")
    print(f"Downloading wavelength file...")
    download_file(url, dest)
    print(f"  -> {dest}")


def download_grid():
    os.makedirs(OUT_DIR, exist_ok=True)
    combos = [(t, g, f) for t in TEFF_VALUES for g in LOGG_VALUES for f in FEH_VALUES]
    print(f"Grid contains {len(combos)} files "
          f"({len(TEFF_VALUES)} Teff x {len(LOGG_VALUES)} logg x {len(FEH_VALUES)} [Fe/H])")

    failed = []
    for i, (teff, logg, feh) in enumerate(combos, 1):
        fname = phoenix_filename(teff, logg, feh)
        url = f"{BASE_URL}/{GRID_DIR}/{feh_dirname(feh)}/{fname}"
        dest = os.path.join(OUT_DIR, fname)
        print(f"[{i}/{len(combos)}] {fname}")
        try:
            download_file(url, dest)
            time.sleep(1.5)  # be gentle -- avoid tripping rate-limiting/proxy blocks
        except requests.exceptions.RequestException as e:
            print(f"  FAILED: {e}  (url: {url})")
            failed.append(fname)

    print(f"\n{len(combos) - len(failed)}/{len(combos)} files present.")
    if failed:
        print(f"{len(failed)} file(s) still missing:")
        for f in failed:
            print(f"  - {f}")
        print("Re-run this script to retry only the missing files.")


if __name__ == "__main__":
    download_wave_file()
    download_grid()
    print("Done.")