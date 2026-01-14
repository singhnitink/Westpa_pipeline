#!/usr/bin/env python3
"""Calculate progress coordinate for WESTPA."""

import mdtraj as md
import sys
import numpy as np

# Usage: python calc_pcoord.py <topology_pdb> <reference_pdb> <trajectory_file> <output_file>

if len(sys.argv) < 5:
    print("Usage: python calc_pcoord.py <topology_pdb> <reference_pdb> <trajectory_file> <output_file>")
    sys.exit(1)

top_file = sys.argv[1]
ref_file = sys.argv[2]  # May act as 2nd structure or just reference
traj_file = sys.argv[3]
out_file = sys.argv[4]

try:
    # Load reference/topology
    ref = md.load(ref_file)
    
    # Load trajectory
    traj = md.load(traj_file, top=top_file)
except Exception as e:
    print(f"Error loading files: {e}")
    sys.exit(1)

pcoord_values = []

# RMSD Calculation
# Selection: name CA
# Reference Selection: name CA

try:
    pcoord_sel = """name CA"""
    ref_sel = """name CA"""
    
    # Select atoms in Reference
    ref_indices = ref.topology.select(ref_sel)
    if len(ref_indices) == 0:
        raise ValueError(f"No atoms found for reference selection: {ref_sel}")

    # Select atoms in Trajectory
    traj_indices = traj.topology.select(pcoord_sel)
    if len(traj_indices) == 0:
        raise ValueError(f"No atoms found for trajectory selection: {pcoord_sel}")
        
    # Standard RMSD usually requires matching atom counts and order
    # We'll use common residue mapping for robustness if selections differ slightly, 
    # but strictly user should provide matching selections.
    
    # Map atoms based on Residue Sequence Number for robust comparison
    # (Assuming we want CA-to-CA or similar backbone alignment)
    is_backbone = "name CA" in pcoord_sel or "name C" in pcoord_sel or "name N" in pcoord_sel
    
    if is_backbone:
        # Robust mapping for proteins - map residue sequence number to atom index
        # Only include atoms that are in our selection
        ref_indices_set = set(ref_indices)
        traj_indices_set = set(traj_indices)
        
        ref_res_map = {r.resSeq: a.index for r in ref.topology.residues for a in r.atoms if a.index in ref_indices_set}
        traj_res_map = {r.resSeq: a.index for r in traj.topology.residues for a in r.atoms if a.index in traj_indices_set}
        
        common_res = sorted(set(ref_res_map.keys()) & set(traj_res_map.keys()))
        if not common_res:
            raise ValueError("No common residues found for RMSD.")
            
        ref_atom_indices = [ref_res_map[r] for r in common_res]
        traj_atom_indices = [traj_res_map[r] for r in common_res]
    else:
        # Direct index mapping (trusting the user's selection matches counts)
        if len(ref_indices) != len(traj_indices):
             raise ValueError(f"Atom count mismatch: Ref {len(ref_indices)} vs Traj {len(traj_indices)}")
        ref_atom_indices = ref_indices
        traj_atom_indices = traj_indices

    # Superpose and calculate RMSD
    traj.superpose(ref, frame=0, atom_indices=traj_atom_indices, ref_atom_indices=ref_atom_indices)
    rmsd_nm = md.rmsd(traj, ref, frame=0, atom_indices=traj_atom_indices, ref_atom_indices=ref_atom_indices)
    
    # Convert to Angstroms
    pcoord_values = rmsd_nm * 10.0

except Exception as e:
    print(f"Error in RMSD calculation: {e}")
    sys.exit(1)

print(f"Calculated pcoord for {len(pcoord_values)} frames.")
print(f"Range: {pcoord_values.min():.2f} - {pcoord_values.max():.2f}")

# Save to file
np.savetxt(out_file, pcoord_values)
