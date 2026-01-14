import h5py
import numpy as np

with h5py.File('west_copy.h5', 'r') as f:
    # Get the last iteration
    n_iter = f.attrs['west_current_iteration'] - 1
    print(f"Checking iteration {n_iter}")
    
    # Get pcoords for this iteration
    # Shape is (n_segments, n_timepoints, n_dim)
    pcoords = f[f'iterations/iter_{n_iter:08d}/pcoord'][:]
    
    # We care about the final point of each segment usually, or the initial
    final_pcoords = pcoords[:, -1, 0]
    
    print(f"Number of segments: {len(final_pcoords)}")
    print(f"Min pcoord: {np.min(final_pcoords)}")
    print(f"Max pcoord: {np.max(final_pcoords)}")
    print(f"Mean pcoord: {np.mean(final_pcoords)}")
    print(f"Standard Dev: {np.std(final_pcoords)}")
    
    # Histogram
    hist, edges = np.histogram(final_pcoords, bins=10)
    print("\nHistogram of pcoords:")
    for i in range(len(hist)):
        print(f"{edges[i]:.2f} - {edges[i+1]:.2f}: {hist[i]} walkers")

    # Check weights to see if probability is concentrated
    seg_index = f[f'iterations/iter_{n_iter:08d}/seg_index']
    weights = seg_index['weight']
    print(f"\nTotal probability weight: {np.sum(weights)}")
    print(f"Max weight segment: {np.max(weights)}")
    print(f"Min weight segment: {np.min(weights)}")
