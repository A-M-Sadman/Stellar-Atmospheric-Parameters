"""
Re-fit only the grid points that failed in the first Q6 run (corrupted
downloads) and merge them into the existing sersic_grid_fits.csv, plus
re-fit everything with the loosened bounds for consistency.

Run this AFTER re-downloading the 3 corrupted files.
"""
import pandas as pd
from sersic_fit import fit_full_grid

if __name__ == "__main__":
    print("Re-fitting full grid with corrected files and loosened bounds...")
    grid_results = fit_full_grid()
    grid_results.to_csv("../data/sersic_grid_fits.csv", index=False)
    print(f"Saved {len(grid_results)} grid fits to ../data/sersic_grid_fits.csv")
    print(f"Expected 108, got {len(grid_results)}"
          + (" -- all present!" if len(grid_results) == 108 else " -- still missing some, check log above"))