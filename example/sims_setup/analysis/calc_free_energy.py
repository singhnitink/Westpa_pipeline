#!/usr/bin/env python3
"""Calculate and plot free energy profile."""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

WEST_H5 = sys.argv[1] if len(sys.argv) > 1 else "../west.h5"
BIN_WIDTH = 0.1

def calculate_fes():
    pcoords = []
    weights = []
    
    print("Reading simulation data...")
    with h5py.File(WEST_H5, "r") as f:
        iterations = sorted([int(x.replace("iter_", "")) for x in f["iterations"].keys()])
        
        for n_iter in iterations:
            iter_group = f[f"iterations/iter_{n_iter:08d}"]
            pc = iter_group["pcoord"][:, :, 0].flatten()
            w = iter_group["seg_index"]["weight"]
            n_points = iter_group["pcoord"].shape[1]
            w_repeated = np.repeat(w, n_points)
            
            pcoords.append(pc)
            weights.append(w_repeated)
            
    pcoords = np.concatenate(pcoords)
    weights = np.concatenate(weights)
    
    print(f"Total data points: {len(pcoords)}")
    
    # Histogramming
    bins = np.arange(np.min(pcoords), np.max(pcoords) + BIN_WIDTH, BIN_WIDTH)
    hist, edges = np.histogram(pcoords, bins=bins, weights=weights, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    
    # Calculate Free Energy
    kT = 0.593  # kcal/mol at 298K
    valid = hist > 0
    G = -kT * np.log(hist[valid])
    G = G - np.min(G)
    
    plt.figure(figsize=(8, 5))
    plt.plot(centers[valid], G, linewidth=2, color='red')
    plt.xlabel("Progress Coordinate")
    plt.ylabel("Free Energy (kcal/mol)")
    plt.title("Free Energy Profile")
    plt.grid(True)
    
    plt.savefig("free_energy_profile.png", dpi=300)
    print("Free Energy plot saved to free_energy_profile.png")

if __name__ == "__main__":
    calculate_fes()
