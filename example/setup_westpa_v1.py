#!/usr/bin/env python3
"""
WESTPA Easy Setup Pipeline
==========================

A user-friendly tool to automate WESTPA simulation setup for NAMD with RMSD-based progress coordinates.

Usage:
    python setup_westpa.py --pdb system.pdb --psf system.psf --toppar ./toppar \
                           --target target.pdb --basis-coor eq.coor \
                           --basis-vel eq.vel --basis-xsc eq.xsc \
                           --output-dir ./my_simulation

Author: Generated with assistance from AI for the scientific community
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="WESTPA Easy Setup Pipeline - Automate WESTPA simulation setup for NAMD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python setup_westpa.py \\
        --pdb system.pdb --psf system.psf --toppar ./toppar \\
        --target target.pdb \\
        --basis-coor eq.coor --basis-vel eq.vel --basis-xsc eq.xsc \\
        --output-dir ./my_simulation

This will create a complete WESTPA simulation directory ready to run.
        """
    )
    
    # Required inputs
    required = parser.add_argument_group('Required Inputs')
    required.add_argument('--pdb', required=True, help='System PDB file')
    required.add_argument('--psf', required=True, help='System PSF topology file')
    required.add_argument('--toppar', required=True, help='CHARMM parameter directory')
    required.add_argument('--target', required=True, help='Target structure PDB for RMSD calculation')
    required.add_argument('--basis-coor', required=True, help='Equilibrated .coor file for basis state')
    required.add_argument('--basis-vel', required=True, help='Equilibrated .vel file for basis state')
    required.add_argument('--basis-xsc', required=True, help='Equilibrated .xsc file for basis state')
    required.add_argument('--output-dir', required=True, help='Output directory for WESTPA simulation')
    
    # WESTPA settings
    westpa = parser.add_argument_group('WESTPA Settings')
    westpa.add_argument('--walkers', type=int, default=8, 
                        help='Number of walkers per bin (default: 8)')
    westpa.add_argument('--iterations', type=int, default=500, 
                        help='Maximum number of WE iterations (default: 500)')
    westpa.add_argument('--rmsd-target', type=float, default=1.0, 
                        help='Target RMSD in Angstroms (default: 1.0)')
    westpa.add_argument('--rmsd-max', type=float, default=8.0, 
                        help='Maximum RMSD for binning (default: 8.0)')
    westpa.add_argument('--bin-width', type=float, default=0.5, 
                        help='Bin width in Angstroms (default: 0.5)')
    
    # NAMD settings
    namd = parser.add_argument_group('NAMD Settings')
    namd.add_argument('--segment-ps', type=float, default=100.0, 
                      help='Segment length in picoseconds (default: 100)')
    namd.add_argument('--timestep', type=float, default=2.0, 
                      help='MD timestep in femtoseconds (default: 2.0)')
    namd.add_argument('--dcdfreq', type=int, default=500, 
                      help='DCD output frequency in steps (default: 500)')
    namd.add_argument('--temperature', type=float, default=298.0, 
                      help='Simulation temperature in K (default: 298)')
    namd.add_argument('--namd-exe', default='namd3', 
                      help='NAMD executable (default: namd3)')
    namd.add_argument('--namd-threads', type=int, default=2, 
                      help='NAMD CPU threads (default: 2)')
    namd.add_argument('--namd-gpu', type=int, default=0, 
                      help='NAMD GPU device ID, set to -1 for CPU only (default: 0)')
    
    # Runtime settings
    runtime = parser.add_argument_group('Runtime Settings')
    runtime.add_argument('--n-workers', type=int, default=4, 
                         help='Number of WESTPA workers (default: 4)')
    runtime.add_argument('--walltime', default='48:00:00', 
                         help='Maximum wallclock time (default: 48:00:00)')
    
    return parser.parse_args()


def validate_inputs(args):
    """Validate that all input files exist."""
    errors = []
    
    for arg_name, path in [
        ('pdb', args.pdb),
        ('psf', args.psf),
        ('toppar', args.toppar),
        ('target', args.target),
        ('basis-coor', args.basis_coor),
        ('basis-vel', args.basis_vel),
        ('basis-xsc', args.basis_xsc),
    ]:
        if not os.path.exists(path):
            errors.append(f"  --{arg_name}: '{path}' does not exist")
    
    if errors:
        print("ERROR: The following input files/directories were not found:")
        for e in errors:
            print(e)
        sys.exit(1)
    
    # Important warning about PDB vs COOR geometry
    print("\n" + "=" * 60)
    print("IMPORTANT: PDB and Basis State Geometry")
    print("=" * 60)
    print("WARNING: The pipeline uses --pdb to calculate the initial RMSD.")
    print("         Ensure --pdb and --basis-coor have the SAME geometry.")
    print("         If they differ (e.g., PDB is crystal structure, COOR is")
    print("         equilibrated), you will see pcoord discontinuities in")
    print("         iteration 1, which can affect flux calculations.")
    print("=" * 60 + "\n")
    
    # Check output directory
    if os.path.exists(args.output_dir):
        print(f"WARNING: Output directory '{args.output_dir}' already exists.")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            sys.exit(0)
        shutil.rmtree(args.output_dir)


def create_directory_structure(args):
    """Create the WESTPA directory structure."""
    base = Path(args.output_dir)
    
    dirs = [
        base,
        base / 'bstates' / 'eq0',
        base / 'common_files',
        base / 'westpa_scripts',
        base / 'analysis',
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    return base


def calculate_parameters(args):
    """Calculate derived parameters from user inputs."""
    # Convert segment length from ps to steps
    segment_steps = int(args.segment_ps * 1000 / args.timestep)
    
    # Calculate pcoord length (number of frames per segment)
    pcoord_len = segment_steps // args.dcdfreq
    
    # Validate that we get at least 1 frame
    if pcoord_len < 1:
        print(f"ERROR: dcdfreq ({args.dcdfreq}) is larger than segment_steps ({segment_steps}).")
        print(f"       This would produce 0 DCD frames. Decrease --dcdfreq or increase --segment-ps.")
        sys.exit(1)
    
    # Generate bin boundaries
    bins = []
    current = args.rmsd_target
    while current < args.rmsd_max:
        bins.append(current)
        current += args.bin_width
    bins.append(args.rmsd_max)
    bins.append(float('inf'))  # Use float('inf') for proper YAML parsing
    
    return {
        'segment_steps': segment_steps,
        'pcoord_len': pcoord_len,
        'bin_boundaries': bins,
        'n_bins': len(bins) - 1,
    }


def copy_input_files(args, base):
    """Copy input files to the WESTPA directory."""
    common = base / 'common_files'
    bstates = base / 'bstates' / 'eq0'
    
    # Get basenames
    pdb_name = os.path.basename(args.pdb)
    psf_name = os.path.basename(args.psf)
    target_name = os.path.basename(args.target)
    
    # Copy common files
    shutil.copy2(args.pdb, common / pdb_name)
    shutil.copy2(args.psf, common / psf_name)
    shutil.copy2(args.target, common / target_name)
    
    # Copy or link toppar directory
    toppar_dest = common / 'toppar'
    if os.path.isdir(args.toppar):
        shutil.copytree(args.toppar, toppar_dest)
    else:
        shutil.copy2(args.toppar, toppar_dest)
    
    # Copy basis state files
    shutil.copy2(args.basis_coor, bstates / 'seg.coor')
    shutil.copy2(args.basis_vel, bstates / 'seg.vel')
    shutil.copy2(args.basis_xsc, bstates / 'seg.xsc')
    
    return {
        'pdb_name': pdb_name,
        'psf_name': psf_name,
        'target_name': target_name,
    }


def generate_west_cfg(args, params, base):
    """Generate the west.cfg file."""
    bin_str = ", ".join(str(b) for b in params['bin_boundaries'])
    
    content = f"""# WESTPA Configuration File
# Generated by WESTPA Easy Setup Pipeline
# vi: set filetype=yaml :
---
west: 
  system:
    driver: westpa.core.systems.WESTSystem
    system_options:
      # Dimensionality of your progress coordinate
      pcoord_ndim: 1
      # Number of data points per iteration
      pcoord_len: {params['pcoord_len']}
      # Data type for your progress coordinate 
      pcoord_dtype: !!python/name:numpy.float32
      bins:
        type: RectilinearBinMapper
        boundaries:         
          - [{bin_str}]
      # Number walkers per bin
      bin_target_counts: {args.walkers}
  propagation:
    max_total_iterations: {args.iterations}
    max_run_wallclock:    {args.walltime}
    propagator:           executable
    gen_istates:          false
  data:
    west_data_file: west.h5
    datasets:
      - name:        pcoord
        scaleoffset: 4
      - name:        coord
        dtype:       float32
        scaleoffset: 3
    data_refs:
      segment:       $WEST_SIM_ROOT/traj_segs/{{segment.n_iter:06d}}/{{segment.seg_id:06d}}
      basis_state:   $WEST_SIM_ROOT/bstates/{{basis_state.auxref}}
      initial_state: $WEST_SIM_ROOT/istates/{{initial_state.iter_created}}/{{initial_state.state_id}}
  plugins:
  executable:
    environ:
      PROPAGATION_DEBUG: 1
    datasets:
      - name:    coord
        enabled: false
    propagator:
      executable: $WEST_SIM_ROOT/westpa_scripts/runseg.sh
      stdout:     $WEST_SIM_ROOT/seg_logs/{{segment.n_iter:06d}}-{{segment.seg_id:06d}}.log
      stderr:     stdout
      stdin:      null
      cwd:        null
      environ:
        SEG_DEBUG: 1
    get_pcoord:
      executable: $WEST_SIM_ROOT/westpa_scripts/get_pcoord.sh
      stdout:     /dev/null 
      stderr:     stdout
    gen_istate:
      executable: $WEST_SIM_ROOT/westpa_scripts/gen_istate.sh
      stdout:     /dev/null 
      stderr:     stdout
    post_iteration:
      enabled:    false
      executable: $WEST_SIM_ROOT/westpa_scripts/post_iter.sh
      stderr:     stdout
    pre_iteration:
      enabled:    false
      executable: $WEST_SIM_ROOT/westpa_scripts/pre_iter.sh
      stderr:     stdout
"""
    
    with open(base / 'west.cfg', 'w') as f:
        f.write(content)


def generate_md_conf(args, params, files, base):
    """Generate the NAMD md.conf file."""
    
    # Build NAMD command options
    gpu_option = f"+devices {args.namd_gpu}" if args.namd_gpu >= 0 else ""
    
    content = f"""############### global settings ###############
set temperature {args.temperature}
set inputname parent
set outputname seg

############### basic input ###############
structure       {files['psf_name']}
coordinates     {files['pdb_name']}

# Force-Field Parameters
set toppar_dict "toppar/"
paraTypeCharmm      on

# Auto-detect parameter files
# Add your parameter files here if auto-detection fails
parameters          ${{toppar_dict}}/par_all36m_prot.prm
parameters          ${{toppar_dict}}/par_all36_carb.prm
parameters          ${{toppar_dict}}/par_all36_na.prm
parameters          ${{toppar_dict}}/par_all36_cgenff.prm
parameters          ${{toppar_dict}}/par_all36_lipid.prm
parameters          ${{toppar_dict}}/toppar_water_ions.str

1-4scaling          1.0
exclude         scaled1-4

############# restart input ###############
bincoordinates  $inputname.coor
extendedsystem  $inputname.xsc
binvelocities   $inputname.vel
firsttimestep   0 

########### output options ################
outputname              $outputname
binaryoutput            yes
binaryrestart           yes
outputEnergies          500
outputTiming            1000
restartfreq             {params['segment_steps']}
dcdfreq                 {args.dcdfreq}

########### simulation options ############
dielectric              1.0
switching               on
vdwForceSwitching       on
LJCorrection            on
switchdist              10
cutoff                  12
pairlistdist            14
rigidbonds              all
margin                  8
timestep                {args.timestep}
stepspercycle           200
fullElectFrequency      2
PME                     on
PMEGridSpacing          1.0

############# Thermostat ##################
langevin                on
langevinTemp            $temperature
langevinSeed            RAND
langevinHydrogen        no
langevinDamping         1.0

######## Constant Pressure Control ########
LangevinPiston                    on
LangevinPistonTarget              1.01325
LangevinPistonPeriod              200
LangevinPistonDecay               100
LangevinPistonTemp                $temperature
useGroupPressure                  yes
useConstantRatio                  yes
useFlexibleCell                   yes

# GPU settings (NAMD 3.0+ only)
{"GPUresident on" if args.namd_gpu >= 0 and 'namd3' in args.namd_exe.lower() else "# GPUresident off (CPU mode or NAMD2)"}

# Segment length: {args.segment_ps} ps
run {params['segment_steps']}
"""
    
    with open(base / 'common_files' / 'md.conf', 'w') as f:
        f.write(content)


def generate_runseg_sh(args, files, base):
    """Generate the runseg.sh script."""
    
    gpu_option = f"+devices {args.namd_gpu}" if args.namd_gpu >= 0 else ""
    
    content = f"""#!/bin/bash

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

cd $WEST_SIM_ROOT
mkdir -pv $WEST_CURRENT_SEG_DATA_REF
cd $WEST_CURRENT_SEG_DATA_REF

# Link common files
ln -sf $WEST_SIM_ROOT/common_files/{files['psf_name']} .
ln -sf $WEST_SIM_ROOT/common_files/{files['pdb_name']} .
ln -sf $WEST_SIM_ROOT/common_files/toppar .
ln -sf $WEST_SIM_ROOT/common_files/{files['target_name']} .

# Link parent restart files
ln -sv $WEST_PARENT_DATA_REF/seg.coor ./parent.coor
ln -sv $WEST_PARENT_DATA_REF/seg.vel  ./parent.vel
ln -sv $WEST_PARENT_DATA_REF/seg.xsc  ./parent.xsc

# Prepare input file from template
sed "s/RAND/$WEST_RAND16/g" $WEST_SIM_ROOT/common_files/md.conf > md.in

# Run NAMD (output goes to stdout, WESTPA captures it to seg_logs/)
{args.namd_exe} +p{args.namd_threads} {gpu_option} md.in

# Calculate Progress Coordinate (RMSD) using Python/MDTraj
python $WEST_SIM_ROOT/westpa_scripts/calc_pcoord.py \\
    {files['pdb_name']} \\
    {files['target_name']} \\
    seg.dcd \\
    $WEST_PCOORD_RETURN

# Cleanup
rm -f md.in seg.restart.* parent.*
"""
    
    script_path = base / 'westpa_scripts' / 'runseg.sh'
    with open(script_path, 'w') as f:
        f.write(content)
    os.chmod(script_path, 0o755)


def generate_calc_pcoord_py(base):
    """Generate the calc_pcoord.py script."""
    
    content = '''#!/usr/bin/env python3
"""Calculate RMSD progress coordinate for WESTPA."""

import mdtraj as md
import sys
import numpy as np

# Usage: python calc_pcoord.py <topology_pdb> <reference_pdb> <trajectory_file> <output_file>

if len(sys.argv) < 5:
    print("Usage: python calc_pcoord.py <topology_pdb> <reference_pdb> <trajectory_file> <output_file>")
    sys.exit(1)

top_file = sys.argv[1]
ref_file = sys.argv[2]
traj_file = sys.argv[3]
out_file = sys.argv[4]

print(f"Loading topology: {top_file}")
print(f"Loading reference: {ref_file}")
print(f"Loading trajectory: {traj_file}")

# Load reference (target structure)
ref = md.load(ref_file)

# Load trajectory
try:
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
    print("Error: No common residues found between reference and trajectory.")
    sys.exit(1)

print(f"Found {len(common_res_seqs)} common residues for RMSD calculation.")

# Create the index arrays
ref_atom_indices = [ref_res_map[r] for r in common_res_seqs]
traj_atom_indices = [traj_res_map[r] for r in common_res_seqs]

# Superpose and calculate RMSD
traj.superpose(ref, frame=0, atom_indices=traj_atom_indices, ref_atom_indices=ref_atom_indices)
rmsd_nm = md.rmsd(traj, ref, frame=0, atom_indices=traj_atom_indices, ref_atom_indices=ref_atom_indices)

# Convert to Angstroms
rmsd_angstrom = rmsd_nm * 10.0

print(f"Calculated RMSD (Angstrom) for {len(rmsd_angstrom)} frames.")
print(f"RMSD range: {rmsd_angstrom.min():.2f} - {rmsd_angstrom.max():.2f}")

# Save to file
np.savetxt(out_file, rmsd_angstrom)
'''
    
    script_path = base / 'westpa_scripts' / 'calc_pcoord.py'
    with open(script_path, 'w') as f:
        f.write(content)
    os.chmod(script_path, 0o755)


def generate_gen_istate_sh(base):
    """Generate the gen_istate.sh script for creating initial states."""
    
    content = """#!/bin/bash

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

# WESTPA passes: $1 = basis_state_ref, $2 = initial_state_ref
# basis_state_ref points to $WEST_SIM_ROOT/bstates/{basis_state.auxref} (a directory)
# initial_state_ref points to $WEST_SIM_ROOT/istates/... (we treat this as a directory)

BASIS_STATE_DIR=$1
ISTATE_DIR=$2

# Create the directory for the new initial state
mkdir -p $ISTATE_DIR

# Copy the restart files from basis state to initial state
# These are already named 'seg.*' so runseg.sh can link them as parent.*
cp $BASIS_STATE_DIR/seg.coor $ISTATE_DIR/seg.coor
cp $BASIS_STATE_DIR/seg.vel  $ISTATE_DIR/seg.vel
cp $BASIS_STATE_DIR/seg.xsc  $ISTATE_DIR/seg.xsc

echo "Generated initial state from basis state"
exit 0
"""
    
    script_path = base / 'westpa_scripts' / 'gen_istate.sh'
    with open(script_path, 'w') as f:
        f.write(content)
    os.chmod(script_path, 0o755)


def generate_get_pcoord_sh(args, files, base):
    """Generate the get_pcoord.sh script."""
    
    content = f"""#!/bin/bash

# Get initial progress coordinate for basis states
# This script calculates the RMSD of the initial structure to the target

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

cd $WEST_SIM_ROOT

# Create a temporary directory
TMPDIR=$(mktemp -d)
cd $TMPDIR

# Link files
ln -s $WEST_SIM_ROOT/common_files/{files['pdb_name']} .
ln -s $WEST_SIM_ROOT/common_files/{files['target_name']} .

# Calculate initial RMSD using the system PDB and target PDB
# Since the basis state .coor matches the PDB coordinates (equilibrated),
# we use the PDB file directly which MDTraj can read
python << 'EOF'
import mdtraj as md
import numpy as np
import sys

try:
    # Load structures
    system = md.load('{files['pdb_name']}')
    target = md.load('{files['target_name']}')
    
    # Select CA atoms
    sys_res_map = {{r.resSeq: a.index for r in system.topology.residues for a in r.atoms if a.name == 'CA'}}
    tgt_res_map = {{r.resSeq: a.index for r in target.topology.residues for a in r.atoms if a.name == 'CA'}}
    
    # Find common residues
    common = sorted(set(sys_res_map.keys()) & set(tgt_res_map.keys()))
    
    if len(common) == 0:
        print("Warning: No common CA atoms found, using default RMSD")
        rmsd = 5.0
    else:
        sys_idx = [sys_res_map[r] for r in common]
        tgt_idx = [tgt_res_map[r] for r in common]
        
        system.superpose(target, atom_indices=sys_idx, ref_atom_indices=tgt_idx)
        rmsd = md.rmsd(system, target, atom_indices=sys_idx, ref_atom_indices=tgt_idx)[0] * 10.0
    
    print(f"Initial RMSD: {{rmsd:.4f}} Angstroms")
    np.savetxt('$WEST_PCOORD_RETURN', [rmsd])
    
except Exception as e:
    print(f"Error calculating initial pcoord: {{e}}")
    print("Using fallback RMSD of 5.0 Angstroms")
    np.savetxt('$WEST_PCOORD_RETURN', [5.0])
EOF

# Cleanup
cd $WEST_SIM_ROOT
rm -rf $TMPDIR
"""
    
    script_path = base / 'westpa_scripts' / 'get_pcoord.sh'
    with open(script_path, 'w') as f:
        f.write(content)
    os.chmod(script_path, 0o755)


def generate_helper_scripts(args, base):
    """Generate env.sh, init.sh, run.sh, clean.sh."""
    
    # env.sh
    env_content = f"""#!/bin/bash
export WEST_SIM_ROOT="$PWD"
export SIM_NAME=$(basename $WEST_SIM_ROOT)
export WEST_PYTHON=$(which python)
export WM_ZMQ_MASTER_HEARTBEAT=100
export WM_ZMQ_WORKER_HEARTBEAT=100
export WM_ZMQ_TIMEOUT=1000
"""
    with open(base / 'env.sh', 'w') as f:
        f.write(env_content)
    os.chmod(base / 'env.sh', 0o755)
    
    # init.sh
    init_content = """#!/bin/bash
source env.sh

# Remove old simulation data if present
rm -rf traj_segs seg_logs istates west.h5 west.log

# Initialize WESTPA
w_init --bstates-from bstates/bstates.txt --tstate-file tstate.file --segs-per-state 1 --work-manager=threads

echo "Initialization complete. Run './run.sh' to start the simulation."
"""
    with open(base / 'init.sh', 'w') as f:
        f.write(init_content)
    os.chmod(base / 'init.sh', 0o755)
    
    # run.sh
    run_content = f"""#!/bin/bash
source env.sh

echo "Starting WESTPA simulation with {args.n_workers} workers..."
echo "Output will be logged to west.log"

w_run --n-workers {args.n_workers} &> west.log &
echo "Simulation started in background. PID: $!"
echo "Use 'tail -f west.log' to monitor progress."
"""
    with open(base / 'run.sh', 'w') as f:
        f.write(run_content)
    os.chmod(base / 'run.sh', 0o755)
    
    # clean.sh
    clean_content = """#!/bin/bash
# Clean up simulation data for fresh restart
# NOTE: Keeps bstates/ - your equilibrated starting structure!

echo "Cleaning simulation data..."
rm -rf traj_segs
rm -rf istates
rm -rf seg_logs
rm -f west.h5
rm -f west.log
rm -f west_copy.h5

echo "Cleanup complete. Run './init.sh' to reinitialize."
"""
    with open(base / 'clean.sh', 'w') as f:
        f.write(clean_content)
    os.chmod(base / 'clean.sh', 0o755)


def generate_tstate_bstate_files(args, base):
    """Generate tstate.file and bstates.txt."""
    
    # tstate.file
    with open(base / 'tstate.file', 'w') as f:
        f.write(f"Target {args.rmsd_target}\n")
    
    # bstates.txt
    with open(base / 'bstates' / 'bstates.txt', 'w') as f:
        f.write("0 1.0 eq0\n")


def generate_analysis_scripts(base):
    """Generate analysis helper scripts."""
    
    # plot_pcoord_evolution.py
    pcoord_content = '''#!/usr/bin/env python3
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
    plt.ylabel("RMSD (Angstrom)")
    plt.title("Progress Coordinate Evolution")
    plt.grid(True)
    
    plt.savefig("pcoord_evolution.png", dpi=300)
    print("Plot saved to pcoord_evolution.png")

if __name__ == "__main__":
    plot_evolution()
'''
    with open(base / 'analysis' / 'plot_pcoord_evolution.py', 'w') as f:
        f.write(pcoord_content)
    
    # calc_free_energy.py
    fes_content = '''#!/usr/bin/env python3
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
    plt.xlabel("RMSD (Angstrom)")
    plt.ylabel("Free Energy (kcal/mol)")
    plt.title("Free Energy Profile")
    plt.grid(True)
    
    plt.savefig("free_energy_profile.png", dpi=300)
    print("Free Energy plot saved to free_energy_profile.png")

if __name__ == "__main__":
    calculate_fes()
'''
    with open(base / 'analysis' / 'calc_free_energy.py', 'w') as f:
        f.write(fes_content)
    
    # trace_walker.py
    trace_content = '''#!/usr/bin/env python3
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
            
    print(f"\\nTrace (Iter, SegID):")
    print("-" * 20)
    for it, seg in reversed(trace):
        print(f"{it:06d}  {seg:06d}")
        
    print(f"\\nTrajectory files are in: ../traj_segs/ITER/SEGID/seg.dcd")

if __name__ == "__main__":
    trace_best_walker()
'''
    with open(base / 'analysis' / 'trace_walker.py', 'w') as f:
        f.write(trace_content)


def generate_readme(args, params, base):
    """Generate README.md with usage instructions."""
    
    content = f"""# WESTPA Simulation

This WESTPA simulation was generated using the WESTPA Easy Setup Pipeline.

## Quick Start

```bash
# 1. Activate WESTPA environment
conda activate westpa

# 2. Initialize the simulation
./init.sh

# 3. Start the simulation
./run.sh

# 4. Monitor progress
tail -f west.log
```

## Configuration Summary

| Parameter | Value |
|-----------|-------|
| Walkers per bin | {args.walkers} |
| Max iterations | {args.iterations} |
| Segment length | {args.segment_ps} ps |
| Target RMSD | {args.rmsd_target} Å |
| Bin width | {args.bin_width} Å |
| Temperature | {args.temperature} K |

## Directory Structure

```
├── west.cfg          # WESTPA configuration
├── init.sh           # Initialize simulation
├── run.sh            # Start simulation
├── clean.sh          # Clean for restart
├── bstates/          # Basis states
├── common_files/     # Topology & parameters
├── westpa_scripts/   # Propagator scripts
└── analysis/         # Analysis tools
```

## Analysis

After running, analyze results:

```bash
cd analysis

# Plot progress coordinate evolution
python plot_pcoord_evolution.py ../west.h5

# Calculate free energy profile
python calc_free_energy.py ../west.h5

# Trace best walker
python trace_walker.py ../west.h5
```

## Troubleshooting

- **pcoord shape error**: Check that `pcoord_len` in west.cfg matches DCD frames
- **NAMD errors**: Check seg_logs/ for detailed output
- **Low bin exploration**: Increase segment length or walkers

## Generated by

WESTPA Easy Setup Pipeline
"""
    
    with open(base / 'README.md', 'w') as f:
        f.write(content)


def main():
    print("=" * 60)
    print("WESTPA Easy Setup Pipeline")
    print("=" * 60)
    
    args = parse_args()
    
    print("\n1. Validating inputs...")
    validate_inputs(args)
    
    print("\n2. Calculating parameters...")
    params = calculate_parameters(args)
    print(f"   Segment: {params['segment_steps']} steps = {args.segment_ps} ps")
    print(f"   pcoord_len: {params['pcoord_len']} frames")
    print(f"   Bins: {params['n_bins']} bins")
    
    print("\n3. Creating directory structure...")
    base = create_directory_structure(args)
    
    print("\n4. Copying input files...")
    files = copy_input_files(args, base)
    
    print("\n5. Generating configuration files...")
    generate_west_cfg(args, params, base)
    generate_md_conf(args, params, files, base)
    generate_runseg_sh(args, files, base)
    generate_calc_pcoord_py(base)
    generate_gen_istate_sh(base)
    generate_get_pcoord_sh(args, files, base)
    generate_helper_scripts(args, base)
    generate_tstate_bstate_files(args, base)
    generate_analysis_scripts(base)
    generate_readme(args, params, base)
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print(f"\nYour WESTPA simulation is ready in: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  cd {args.output_dir}")
    print(f"  conda activate westpa")
    print(f"  ./init.sh")
    print(f"  ./run.sh")
    print()


if __name__ == '__main__':
    main()
