import h5py
import numpy as np

try:
    with h5py.File("west_copy.h5", "r") as f:
        iters = sorted([int(x.replace("iter_", "")) for x in f["iterations"].keys()])
        last_iter = iters[-1]
        
        # Iter 1 stats
        pcoord_1 = f["iterations/iter_00000001"]["pcoord"][:, -1, 0]
        min_1 = np.min(pcoord_1)
        max_1 = np.max(pcoord_1)
        
        # Last Iter stats
        pcoord_last = f[f"iterations/iter_{last_iter:08d}"]["pcoord"][:, -1, 0]
        min_last = np.min(pcoord_last)
        max_last = np.max(pcoord_last)
        
        print(f"Latest Iteration: {last_iter}")
        print(f"Iter 1 Min RMSD: {min_1:.4f}")
        print(f"Latest Min RMSD: {min_last:.4f}")
        
except Exception as e:
    print(f"Error: {e}")
