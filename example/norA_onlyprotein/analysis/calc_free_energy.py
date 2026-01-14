import h5py
import numpy as np
import matplotlib.pyplot as plt

import sys

# Usage: python calc_free_energy.py [west.h5]

if len(sys.argv) > 1:
    WEST_H5 = sys.argv[1]
else:
    WEST_H5 = "../west.h5"
BIN_WIDTH = 0.1  # Angstroms

def calculate_fes():
    pcoords = []
    weights = []
    
    print("Reading simulation data...")
    with h5py.File(WEST_H5, "r") as f:
        iterations = sorted([int(x.replace("iter_", "")) for x in f["iterations"].keys()])
        # Skip the first few iterations as equilibration if desired (e.g., last 50%)
        # Here we use all data for visualization
        
        for n_iter in iterations:
            iter_group = f[f"iterations/iter_{n_iter:08d}"]
            
            # shape: (n_walkers, n_timepoints, n_dim)
            pc = iter_group["pcoord"][:, :, 0].flatten()
            
            # shape: (n_walkers,) -> need to repeat for timepoints
            w = iter_group["seg_index"]["weight"]
            # Repeat weight for each timepoint
            n_points = iter_group["pcoord"].shape[1]
            w_repeated = np.repeat(w, n_points)
            
            pcoords.append(pc)
            weights.append(w_repeated)
            
    pcoords = np.concatenate(pcoords)
    weights = np.concatenate(weights)
    
    print(f"Total data points: {len(pcoords)}")
    
    # Histogramming
    min_val = np.min(pcoords)
    max_val = np.max(pcoords)
    bins = np.arange(min_val, max_val + BIN_WIDTH, BIN_WIDTH)
    
    hist, edges = np.histogram(pcoords, bins=bins, weights=weights, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    
    # Calculate Free Energy: G = -kT * ln(P)
    # k_B = 0.001987 kcal/mol/K, T = 300 (assume)
    kT = 0.593  # kcal/mol at 298K
    
    # Avoid log(0)
    valid_indices = hist > 0
    G = -kT * np.log(hist[valid_indices])
    centers = centers[valid_indices]
    
    # Shift so min is 0
    G = G - np.min(G)
    
    plt.figure(figsize=(8, 5))
    plt.plot(centers, G, linewidth=2, color='red')
    plt.xlabel("RMSD (Angstrom)")
    plt.ylabel("Free Energy (kcal/mol)")
    plt.title("Spatially Averaged Free Energy Profile")
    plt.grid(True)
    
    output_file = "free_energy_profile.png"
    plt.savefig(output_file, dpi=300)
    print(f"Free Energy plot saved to {output_file}")

if __name__ == "__main__":
    try:
        calculate_fes()
    except Exception as e:
        print(f"Error: {e}")
