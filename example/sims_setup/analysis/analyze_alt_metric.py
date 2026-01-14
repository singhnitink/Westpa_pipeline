#!/usr/bin/env python3
"""
Analyze Alternate Metrics from Existing WESTPA Simulation.
This script calculates a new metric (e.g., RoG, SASA) from existing trajectories
and plots the free energy profile.
"""

import h5py
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze alternate metrics from WESTPA simulation")
    parser.add_argument('--west-h5', default="../west.h5", help='Path to west.h5 file')
    parser.add_argument('--traj-dir', default="../traj_segs", help='Path to trajectory segments')
    parser.add_argument('--top', required=True, help='Topology file (PDB/PSF)')
    parser.add_argument('--ref', default=None, help='Reference PDB (if needed for metric)')
    parser.add_argument('--type', required=True, choices=['rmsd', 'distance', 'rog', 'sasa', 'dihedral'], 
                        help='Metric type')
    parser.add_argument('--sel', required=True, help='Selection string 1')
    parser.add_argument('--sel2', default=None, help='Selection string 2 (for distance)')
    parser.add_argument('--first-iter', type=int, default=1, help='First iteration to analyze')
    parser.add_argument('--last-iter', type=int, default=None, help='Last iteration to analyze')
    parser.add_argument('--bin-width', type=float, default=0.1, help='Bin width for FES')
    parser.add_argument('--temp', type=float, default=298.0, help='Temperature in K')
    return parser.parse_args()

def calculate_metric(traj, args, ref=None):
    """Calculate the requested metric for a trajectory chunk."""
    try:
        val = None
        if args.type == 'rmsd':
            if ref is None: raise ValueError("RMSD requires --ref")
            
            # Use tripe quotes for safety
            sel_str = """name CA""" # Ignored here, using args.sel from runtime
            sel_runtime = args.sel # User input from this script execution
            
            # Logic similar to setup_westpa.py but using runtime args
            traj_idx = traj.topology.select(sel_runtime)
            ref_idx = ref.topology.select(sel_runtime) # Assuming simple matching for analysis
            
            if len(traj_idx) == 0: raise ValueError(f"No atoms for selection: {sel_runtime}")
            
            traj.superpose(ref, frame=0, atom_indices=traj_idx, ref_atom_indices=ref_idx)
            val = md.rmsd(traj, ref, frame=0, atom_indices=traj_idx, ref_atom_indices=ref_idx) * 10.0
            
        elif args.type == 'distance':
            if args.sel2 is None: raise ValueError("Distance requires --sel2")
            
            sel1 = args.sel
            sel2 = args.sel2
            
            # COM distance logic
            com1 = md.compute_center_of_mass(traj, select=sel1)
            com2 = md.compute_center_of_mass(traj, select=sel2)
            val = np.linalg.norm(com1 - com2, axis=1) * 10.0
            
        elif args.type == 'rog':
            idx = traj.topology.select(args.sel)
            val = md.compute_rg(traj, atom_indices=idx) * 10.0
            
        elif args.type == 'sasa':
            idx = traj.topology.select(args.sel)
            sasa = md.shrake_rupley(traj, atom_indices=idx, mode='residue')
            val = np.sum(sasa, axis=1) * 100.0
            
        elif args.type == 'dihedral':
            idx = traj.topology.select(args.sel)
            if len(idx) != 4: raise ValueError("Dihedral requires exactly 4 atoms")
            idx = idx.reshape(1, 4)
            val = np.degrees(md.compute_dihedrals(traj, indices=idx).flatten())
            
        return val
        
    except Exception as e:
        print(f"Error calculating metric: {e}")
        return None

def main():
    args = parse_args()
    
    print(f"Analyzing {args.type} for iterations {args.first_iter} to {args.last_iter if args.last_iter else 'END'}")
    
    # Load Reference if needed
    ref = None
    if args.ref:
        print(f"Loading reference: {args.ref}")
        ref = md.load(args.ref)
    
    # Open WESTPA H5
    with h5py.File(args.west_h5, 'r') as f:
        # Get iterations
        all_iters = sorted([int(x.replace("iter_", "")) for x in f["iterations"].keys()])
        if args.last_iter:
            all_iters = [i for i in all_iters if args.first_iter <= i <= args.last_iter]
        else:
            all_iters = [i for i in all_iters if i >= args.first_iter]
            
        print(f"Processing {len(all_iters)} iterations...")
        
        all_vals = []
        all_weights = []
        
        for n_iter in all_iters:
            print(f"  Iteration {n_iter}...", end='\r')
            iter_group = f[f"iterations/iter_{n_iter:08d}"]
            weights = iter_group['seg_index']['weight']
            
            # Filter active walkers
            active_mask = weights > 0
            active_indices = np.where(active_mask)[0]
            
            for seg_id in active_indices:
                weight = weights[seg_id]
                
                # Construct path to DCD
                dcd_path = f"{args.traj_dir}/{n_iter:06d}/{seg_id:06d}/seg.dcd"
                
                if not os.path.exists(dcd_path):
                    continue
                    
                try:
                    traj = md.load(dcd_path, top=args.top)
                    vals = calculate_metric(traj, args, ref)
                    
                    if vals is not None:
                        all_vals.extend(vals)
                        # Replicate weight for each frame
                        all_weights.extend([weight] * len(vals))
                        
                except Exception as e:
                    # Silent fail for corrupt DCDs
                    continue
                    
        print("\nDone processing trajectories.")
        
        all_vals = np.array(all_vals)
        all_weights = np.array(all_weights)
        
        # Calculate FES
        print("Calculating Free Energy Surface...")
        kT = 0.001987 * args.temp
        
        hist, edges = np.histogram(all_vals, bins=np.arange(all_vals.min(), all_vals.max() + args.bin_width, args.bin_width), weights=all_weights, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        
        valid = hist > 0
        G = -kT * np.log(hist[valid])
        G = G - np.min(G)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(centers[valid], G, linewidth=2)
        plt.xlabel(f"{args.type} (Angstrom/Deg)")
        plt.ylabel("Free Energy (kcal/mol)")
        plt.title(f"FES along {args.type}")
        plt.grid(True)
        plt.savefig(f"fes_{args.type}.png", dpi=300)
        print(f"Plot saved to fes_{args.type}.png")

if __name__ == "__main__":
    main()
