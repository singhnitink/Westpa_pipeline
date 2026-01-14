import h5py
import matplotlib.pyplot as plt
import numpy as np

import sys
# Usage: python plot_pcoord_evolution.py [west.h5]

if len(sys.argv) > 1:
    WEST_H5 = sys.argv[1]
else:
    WEST_H5 = "../west.h5"

def plot_evolution():
    with h5py.File(WEST_H5, "r") as f:
        iterations = sorted([int(x.replace("iter_", "")) for x in f["iterations"].keys()])
        
        all_iters = []
        all_pcoords = []
        
        print(f"Reading data from {len(iterations)} iterations...")
        
        for n_iter in iterations:
            iter_group = f[f"iterations/iter_{n_iter:08d}"]
            # pcoord is usually shape (n_walkers, n_timepoints, n_dim)
            # We take the LAST timepoint of each walker as its final position
            pcoords = iter_group["pcoord"][:, -1, 0] 
            
            all_iters.extend([n_iter] * len(pcoords))
            all_pcoords.extend(pcoords)
            
    plt.figure(figsize=(10, 6))
    plt.scatter(all_iters, all_pcoords, s=5, alpha=0.5, c='blue')
    plt.xlabel("Iteration")
    plt.ylabel("RMSD (Angstrom)")
    plt.title("Progress Coordinate Evolution")
    plt.grid(True)
    
    output_file = "pcoord_evolution.png"
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    try:
        plot_evolution()
    except Exception as e:
        print(f"Error: {e}")
        print("Note: If the simulation is running, west.h5 might be locked. Try copying it to a temporary file first.")
