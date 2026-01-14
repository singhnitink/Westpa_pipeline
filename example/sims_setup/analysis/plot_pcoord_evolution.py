#!/usr/bin/env python3
"""Plot progress coordinate evolution over iterations."""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys

WEST_H5 = sys.argv[1] if len(sys.argv) > 1 else "../west.h5"

def plot_evolution():
    with h5py.File(WEST_H5, "r") as f:
        iterations = sorted([int(x.replace("iter_", "")) for x in f["iterations"].keys()])
        
        all_iters = []
        all_pcoords = []
        
        print(f"Reading data from {len(iterations)} iterations...")
        
        for n_iter in iterations:
            iter_group = f[f"iterations/iter_{n_iter:08d}"]
            pcoords = iter_group["pcoord"][:, -1, 0]
            all_iters.extend([n_iter] * len(pcoords))
            all_pcoords.extend(pcoords)
            
    plt.figure(figsize=(10, 6))
    plt.scatter(all_iters, all_pcoords, s=5, alpha=0.5, c='blue')
    plt.xlabel("Iteration")
    plt.ylabel(f"Progress Coordinate")
    plt.title("Progress Coordinate Evolution")
    plt.grid(True)
    
    plt.savefig("pcoord_evolution.png", dpi=300)
    print("Plot saved to pcoord_evolution.png")

if __name__ == "__main__":
    plot_evolution()
