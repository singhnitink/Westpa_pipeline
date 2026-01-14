import h5py
import numpy as np
import sys

# Usage: python trace_walker.py [west.h5]

if len(sys.argv) > 1:
    WEST_H5 = sys.argv[1]
else:
    WEST_H5 = "../west.h5"

def trace_best_walker():
    with h5py.File(WEST_H5, "r") as f:
        # Get last iteration
        iters = sorted([int(x.replace("iter_", "")) for x in f["iterations"].keys()])
        last_iter = iters[-1]
        
        print(f"Last iteration: {last_iter}")
        
        iter_group = f[f"iterations/iter_{last_iter:08d}"]
        
        # Find walker with minimum RMSD in the last iteration
        # pcoord shape: (n_walkers, n_timepoints, n_dim)
        final_pcoords = iter_group["pcoord"][:, -1, 0]
        best_walker_idx = np.argmin(final_pcoords)
        best_rmsd = final_pcoords[best_walker_idx]
        
        print(f"Best walker in Iter {last_iter}: SegID {best_walker_idx} with RMSD {best_rmsd:.4f}")
        
        # Trace back
        trace = []
        current_iter = last_iter
        current_seg_id = best_walker_idx
        
        while current_iter >= 1:
            trace.append((current_iter, current_seg_id))
            
            # Get parent info
            # seg_index column 1 is 'parent_id'
            curr_group = f[f"iterations/iter_{current_iter:08d}"]
            parent_id = curr_group["seg_index"][current_seg_id][1]
            
            current_iter -= 1
            current_seg_id = parent_id
            
    print(f"\nTrace (Iter, SegID) for the best walker:")
    print("-----------------------------------------")
    # Reverse to show start -> end
    for it, seg in reversed(trace):
        print(f"{it:06d}  {seg:06d}")
        
    print("\nTo reconstruct trajectory, you can concatenate these DCD files:")
    print(f"(Files are in ../traj_segs/ITER/SEGID/seg.dcd)")

if __name__ == "__main__":
    try:
        trace_best_walker()
    except Exception as e:
        print(f"Error: {e}")
