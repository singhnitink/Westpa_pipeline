import mdtraj as md
import sys
import numpy as np

# Usage: python calc_pcoord.py <topology_pdb> <reference_pdb> <trajectory_file> <output_file> [--last-only]
# --last-only: Output only the last frame's RMSD (for get_pcoord.sh / basis state init)
# Without flag: Output pcoord_len values for segment propagation

PCOORD_LEN = 100  # Must match west.cfg pcoord_len

if len(sys.argv) < 5:
    print("Usage: python calc_pcoord.py <topology_pdb> <reference_pdb> <trajectory_file> <output_file> [--last-only]")
    sys.exit(1)

top_file = sys.argv[1]
ref_file = sys.argv[2]
traj_file = sys.argv[3]
out_file = sys.argv[4]
last_only = len(sys.argv) > 5 and sys.argv[5] == "--last-only"

print(f"Loading topology: {top_file}")
print(f"Loading reference: {ref_file}")
print(f"Loading trajectory: {traj_file}")
print(f"Mode: {'last-only' if last_only else 'full (pcoord_len=' + str(PCOORD_LEN) + ')'}")

# Load reference (Protein only target)
ref = md.load(ref_file)

# Load trajectory - if last_only, load only the last frame for efficiency
try:
    if last_only:
        # Get number of frames first, then load only the last one
        n_frames = len(md.open(traj_file))
        traj = md.load(traj_file, top=top_file, frame=n_frames-1)
        print(f"Loaded only frame {n_frames} (last frame)")
    else:
        traj = md.load(traj_file, top=top_file)
except IOError as e:
    print(f"Error loading trajectory: {e}")
    sys.exit(1)

# Select CA atoms in Reference
ref_ca_indices = ref.topology.select("name CA")
if len(ref_ca_indices) == 0:
    print("Error: No CA atoms found in reference structure.")
    sys.exit(1)

# Select CA atoms in Trajectory
traj_ca_indices = traj.topology.select("name CA")
if len(traj_ca_indices) == 0:
    print("Error: No CA atoms found in trajectory structure.")
    sys.exit(1)

# Map atoms based on Residue Sequence Number
ref_res_map = {r.resSeq: a.index for r in ref.topology.residues for a in r.atoms if a.name == 'CA'}
traj_res_map = {r.resSeq: a.index for r in traj.topology.residues for a in r.atoms if a.name == 'CA'}

# Find common residues
common_res_seqs = sorted(list(set(ref_res_map.keys()) & set(traj_res_map.keys())))

if len(common_res_seqs) == 0:
    print("Error: No common residues found between reference and trajectory based on resSeq.")
    sys.exit(1)

print(f"Found {len(common_res_seqs)} common residues for RMSD calculation.")

# Create the index arrays for the common atoms
ref_atom_indices = [ref_res_map[r] for r in common_res_seqs]
traj_atom_indices = [traj_res_map[r] for r in common_res_seqs]

# Superpose Trajectory onto Reference
traj.superpose(ref, frame=0, atom_indices=traj_atom_indices, ref_atom_indices=ref_atom_indices)

# Calculate RMSD (returns nm, convert to Angstrom)
rmsd_nm = md.rmsd(traj, ref, frame=0, atom_indices=traj_atom_indices, ref_atom_indices=ref_atom_indices)
rmsd_angstrom = rmsd_nm * 10.0

print(f"Calculated RMSD (Angstrom) for {len(rmsd_angstrom)} frames.")
print(f"First: {rmsd_angstrom[0]:.4f}, Last: {rmsd_angstrom[-1]:.4f}")

# Output based on mode
if last_only:
    # For basis state initialization - output only last frame
    output_values = [rmsd_angstrom[-1]]
else:
    # For segment propagation - output exactly pcoord_len values
    if len(rmsd_angstrom) >= PCOORD_LEN:
        output_values = rmsd_angstrom[-PCOORD_LEN:]
    else:
        # Pad with the last value
        padding_needed = PCOORD_LEN - len(rmsd_angstrom)
        padding = np.full(padding_needed, rmsd_angstrom[-1])
        output_values = np.concatenate([rmsd_angstrom, padding])

print(f"Writing {len(output_values)} values to output file.")
np.savetxt(out_file, output_values)

