#!/usr/bin/env python3
"""Trace the best walker back to its origin."""

import h5py
import numpy as np
import sys

WEST_H5 = sys.argv[1] if len(sys.argv) > 1 else "../west.h5"

def trace_best_walker():
    with h5py.File(WEST_H5, "r") as f:
        iters = sorted([int(x.replace("iter_", "")) for x in f["iterations"].keys()])
        last_iter = iters[-1]
        
        print(f"Last iteration: {last_iter}")
        
        iter_group = f[f"iterations/iter_{last_iter:08d}"]
        final_pcoords = iter_group["pcoord"][:, -1, 0]
        best_idx = np.argmin(final_pcoords)
        best_rmsd = final_pcoords[best_idx]
        
        print(f"Best walker: SegID {best_idx} with RMSD {best_rmsd:.4f}")
        
        # Trace back
        trace = []
        current_iter = last_iter
        current_seg_id = best_idx
        
        while current_iter >= 1:
            trace.append((current_iter, current_seg_id))
            curr_group = f[f"iterations/iter_{current_iter:08d}"]
            parent_id = curr_group["seg_index"][current_seg_id][1]
            current_iter -= 1
            current_seg_id = parent_id
            
    print(f"\nTrace (Iter, SegID):")
    print("-" * 20)
    for it, seg in reversed(trace):
        print(f"{it:06d}  {seg:06d}")
        
    print(f"\nTrajectory files are in: ../traj_segs/ITER/SEGID/seg.dcd")

if __name__ == "__main__":
    trace_best_walker()
