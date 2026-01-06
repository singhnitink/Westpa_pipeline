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
    
    # Progress Coordinate settings
    pcoord = parser.add_argument_group('Progress Coordinate Settings')
    pcoord.add_argument('--bin-width', type=float, default=0.5, 
                        help='Bin width (default: 0.5) - Ignored if --adaptive-bins is used')
    pcoord.add_argument('--adaptive-bins', action='store_true',
                        help='Use Minimal Adaptive Binning (MAB) instead of fixed bins')
    pcoord.add_argument('--mab-bins', type=int, default=15,
                        help='Number of bins for MAB (default: 15)')
    pcoord.add_argument('--pcoord-type', choices=['rmsd', 'distance', 'rog', 'sasa', 'dihedral'], default='rmsd',
                        help='Type of progress coordinate (default: rmsd)')
    pcoord.add_argument('--pcoord-sel', default='name CA',
                        help='Atom selection string (MDTraj syntax). For distance/dihedral, this is the 1st group.')
    pcoord.add_argument('--pcoord-ref', default=None,
                        help='Reference atom selection string. For distance, this is the 2nd group. For RMSD, defaults to pcoord-sel.')
    pcoord.add_argument('--pcoord-val-min', type=float, default=None,
                        help='Minimum progress coordinate value for binning (replaces --rmsd-target)')
    pcoord.add_argument('--pcoord-val-max', type=float, default=None,
                        help='Maximum progress coordinate value for binning (replaces --rmsd-max)')
    pcoord.add_argument('--pcoord-target', type=float, default=None,
                        help='Target progress coordinate value')
    
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
    namd.add_argument('--max-run-wallclock', default='48:00:00',
                      help='Maximum wallclock time (default: 48:00:00)')
    
    # Runtime settings
    runtime = parser.add_argument_group('Runtime Settings')
    runtime.add_argument('--n-workers', type=int, default=4, 
                         help='Number of WESTPA workers (default: 4)')
    
    # HPC Submission settings
    hpc = parser.add_argument_group('HPC Submission Settings')
    hpc.add_argument('--gen-submit', choices=['slurm', 'pbs', 'none'], default='none',
                     help='Generate submission script for scheduler (default: none)')
    hpc.add_argument('--job-name', default='westpa_run',
                     help='Job name for scheduler (default: westpa_run)')

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

    # Validate pcoord requirements
    if args.pcoord_type == 'distance' and not args.pcoord_ref:
        print("ERROR: --pcoord-type 'distance' requires --pcoord-ref (the second group of atoms).")
        sys.exit(1)
        
    # Force explicit bin boundaries for non-RMSD metrics
    # RMSD has defaults (1.0-8.0), but others have vastly different ranges
    # EXCEPTION: If using Adaptive Binning (MAB), boundaries are not strictly required for bins,
    # but might be needed for target state or visualization. We'll relax the strict check.
    if args.pcoord_type != 'rmsd' and not args.adaptive_bins:
        if args.pcoord_val_min is None or args.pcoord_val_max is None:
            print(f"ERROR: For pcoord type '{args.pcoord_type}', you MUST specify --pcoord-val-min and --pcoord-val-max.")
            print("       (Alternatively, use --adaptive-bins to automate binning)")
            sys.exit(1)


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
    
    # Map legacy arguments to new generic ones if not provided
    val_min = args.pcoord_val_min if args.pcoord_val_min is not None else args.rmsd_target
    val_max = args.pcoord_val_max if args.pcoord_val_max is not None else args.rmsd_max
    
    # Adjust for target direction:
    # If using RMSD, we usually go from High -> Low (Target 0).
    # But WESTPA bins are typically left-to-right.
    # We will assume a simple rectilinear grid from val_min to val_max
    
    if val_min > val_max:
        val_min, val_max = val_max, val_min
    
    # Generate bin boundaries (Only if NOT using MAB)
    bins = []
    if not args.adaptive_bins:
        current = val_min
        while current < val_max:
            bins.append(round(current, 4))
            current += args.bin_width
        bins.append(val_max)
        bins.append(float('inf'))  # Use float('inf') for proper YAML parsing
    
    return {
        'segment_steps': segment_steps,
        'pcoord_len': pcoord_len,
        'bin_boundaries': bins,
        'n_bins': len(bins) - 1 if bins else args.mab_bins,
        'val_min': val_min,
        'val_max': val_max,
        'use_mab': args.adaptive_bins,
        'mab_bins': args.mab_bins
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
    
    # Construct bin mapper section based on MAB setting
    if params.get('use_mab', False):
        mab_bins = params.get('mab_bins', 15)
        # MABBinMapper syntax
        bin_mapper_section = f"""      bin_mapper:
        type: MABBinMapper
        nbins: [{mab_bins}]
        direction: [0]
        skip_lagged: true"""
    else:
        # RectilinearBinMapper syntax
        bin_str = ", ".join(str(b) for b in params['bin_boundaries'])
        bin_mapper_section = f"""      bin_mapper:
        type: RectilinearBinMapper
        boundaries:         
          - [{bin_str}]"""

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
{bin_mapper_section}
      # Number of walkers per bin
      bin_target_counts: {args.walkers}
  propagation:
    max_total_iterations: {args.iterations}
    max_run_wallclock:    {args.max_run_wallclock}
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


def generate_calc_pcoord_py(base, args):
    """Generate the calc_pcoord.py script dynamically based on metric."""
    
    header = '''#!/usr/bin/env python3
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
'''

    # Logic blocks for different metrics
    if args.pcoord_type == 'rmsd':
        ref_sel = args.pcoord_ref if args.pcoord_ref else args.pcoord_sel
        logic = f'''
# RMSD Calculation
# Selection: {args.pcoord_sel}
# Reference Selection: {ref_sel}

try:
    pcoord_sel = """{args.pcoord_sel}"""
    ref_sel = """{ref_sel}"""
    
    # Select atoms in Reference
    ref_indices = ref.topology.select(ref_sel)
    if len(ref_indices) == 0:
        raise ValueError(f"No atoms found for reference selection: {{ref_sel}}")

    # Select atoms in Trajectory
    traj_indices = traj.topology.select(pcoord_sel)
    if len(traj_indices) == 0:
        raise ValueError(f"No atoms found for trajectory selection: {{pcoord_sel}}")
        
    # Standard RMSD usually requires matching atom counts and order
    # We'll use common residue mapping for robustness if selections differ slightly, 
    # but strictly user should provide matching selections.
    
    # Map atoms based on Residue Sequence Number for robust comparison
    # (Assuming we want CA-to-CA or similar backbone alignment)
    is_backbone = "name CA" in pcoord_sel or "name C" in pcoord_sel or "name N" in pcoord_sel
    
    if is_backbone:
        # Robust mapping for proteins
        ref_res_map = {{r.resSeq: a.index for r in ref.topology.residues for a in r.atoms if a.isInSelection(ref_indices)}}
        traj_res_map = {{r.resSeq: a.index for r in traj.topology.residues for a in r.atoms if a.isInSelection(traj_indices)}}
        
        common_res = sorted(set(ref_res_map.keys()) & set(traj_res_map.keys()))
        if not common_res:
            raise ValueError("No common residues found for RMSD.")
            
        ref_atom_indices = [ref_res_map[r] for r in common_res]
        traj_atom_indices = [traj_res_map[r] for r in common_res]
    else:
        # Direct index mapping (trusting the user's selection matches counts)
        if len(ref_indices) != len(traj_indices):
             raise ValueError(f"Atom count mismatch: Ref {{len(ref_indices)}} vs Traj {{len(traj_indices)}}")
        ref_atom_indices = ref_indices
        traj_atom_indices = traj_indices

    # Superpose and calculate RMSD
    traj.superpose(ref, frame=0, atom_indices=traj_atom_indices, ref_atom_indices=ref_atom_indices)
    rmsd_nm = md.rmsd(traj, ref, frame=0, atom_indices=traj_atom_indices, ref_atom_indices=ref_atom_indices)
    
    # Convert to Angstroms
    pcoord_values = rmsd_nm * 10.0

except Exception as e:
    print(f"Error in RMSD calculation: {{e}}")
    sys.exit(1)
'''

    elif args.pcoord_type == 'distance':
        ref_sel_arg = args.pcoord_ref if args.pcoord_ref else "none"
        logic = f'''
# Distance Calculation
# Group 1: {args.pcoord_sel}
# Group 2: {ref_sel_arg}

try:
    sel1 = """{args.pcoord_sel}"""
    sel2 = """{ref_sel_arg}"""
    
    if sel2 == "none" or not sel2:
        raise ValueError("Distance metric requires --pcoord-ref to specify the second group.")
        
    indices1 = traj.topology.select(sel1)
    indices2 = traj.topology.select(sel2)
    
    if len(indices1) == 0 or len(indices2) == 0:
        raise ValueError("One of the selections returned 0 atoms.")
        
    # Compute center of mass for each group
    # mdtraj.compute_distances expects pairs of atom indices.
    # To do COM distance, we calculate COMs manually.
    
    com1 = md.compute_center_of_mass(traj, select=sel1) # shape (n_frames, 3)
    com2 = md.compute_center_of_mass(traj, select=sel2) # shape (n_frames, 3)
    
    # Euclidean distance
    dists_nm = np.linalg.norm(com1 - com2, axis=1)
    
    # Convert to Angstroms
    pcoord_values = dists_nm * 10.0

except Exception as e:
    print(f"Error in Distance calculation: {{e}}")
    sys.exit(1)
'''

    elif args.pcoord_type == 'rog':
        logic = f'''
# Radius of Gyration Calculation
# Selection: {args.pcoord_sel}

try:
    sel = """{args.pcoord_sel}"""
    indices = traj.topology.select(sel)
    
    if len(indices) == 0:
        raise ValueError(f"No atoms found for selection: {{sel}}")
        
    rg_nm = md.compute_rg(traj, atom_indices=indices)
    
    # Convert to Angstroms
    pcoord_values = rg_nm * 10.0

except Exception as e:
    print(f"Error in RoG calculation: {{e}}")
    sys.exit(1)
'''

    elif args.pcoord_type == 'sasa':
        logic = f'''
# SASA Calculation
# Selection: {args.pcoord_sel}

try:
    sel = """{args.pcoord_sel}"""
    indices = traj.topology.select(sel)
    
    if len(indices) == 0:
        raise ValueError(f"No atoms found for selection: {{sel}}")
        
    # shrake_rupley returns (n_frames, n_atoms)
    # We sum over the selected atoms to get total SASA
    sasa_nm2 = md.shrake_rupley(traj, atom_indices=indices, mode='residue')
    total_sasa_nm2 = np.sum(sasa_nm2, axis=1)
    
    # Convert to Angstroms^2 (1 nm^2 = 100 A^2)
    pcoord_values = total_sasa_nm2 * 100.0

except Exception as e:
    print(f"Error in SASA calculation: {{e}}")
    sys.exit(1)
'''
    elif args.pcoord_type == 'dihedral':
        logic = f'''
# Dihedral Calculation
# Selection: {args.pcoord_sel} (Should be 4 atoms)

try:
    sel = """{args.pcoord_sel}"""
    indices = traj.topology.select(sel)
    
    if len(indices) != 4:
        raise ValueError(f"Dihedral selection must return exactly 4 atoms. Found {{len(indices)}}.")
        
    # compute_dihedrals expects list of [a1, a2, a3, a4]
    indices = indices.reshape(1, 4)
    
    rads = md.compute_dihedrals(traj, indices).flatten()
    
    # Convert to Degrees
    pcoord_values = np.degrees(rads)

except Exception as e:
    print(f"Error in Dihedral calculation: {{e}}")
    sys.exit(1)
'''
    else:
        logic = "print('Unknown pcoord type')\nsys.exit(1)"

    footer = '''
print(f"Calculated pcoord for {len(pcoord_values)} frames.")
print(f"Range: {pcoord_values.min():.2f} - {pcoord_values.max():.2f}")

# Save to file
np.savetxt(out_file, pcoord_values)
'''
    content = header + logic + footer
    
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
# We use the generated calc_pcoord.py to ensure consistency with the chosen metric.
# The basis state structure (PDB) serves as the "trajectory" (1 frame).

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

# Run calc_pcoord.py
# Input 1: Topology (System PDB)
# Input 2: Reference (Target PDB) - used for RMSD, ignored for others
# Input 3: Trajectory (System PDB) - treating the start structure as a 1-frame traj
# Input 4: Output File

python $WEST_SIM_ROOT/westpa_scripts/calc_pcoord.py \\
    {files['pdb_name']} \\
    {files['target_name']} \\
    {files['pdb_name']} \\
    $WEST_PCOORD_RETURN

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
    # Use pcoord-target if provided, else rmsd-target
    target_val = args.pcoord_target if args.pcoord_target is not None else args.rmsd_target
    
    with open(base / 'tstate.file', 'w') as f:
        f.write(f"Target {target_val}\n")
    
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
    plt.ylabel(f"Progress Coordinate")
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
    plt.xlabel("Progress Coordinate")
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


def generate_slurm_script(args, base):
    """Generate Slurm submission script."""
    content = f"""#!/bin/bash
#SBATCH --job-name={args.job_name}
#SBATCH --time={args.max_run_wallclock}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=8G
#SBATCH --output=slurm-%j.log

echo "Job started on $(date)"
echo "Node: $(hostname)"

# Assume user has configured environment
# source ~/.bashrc
# conda activate westpa

echo "Starting WESTPA..."
./run.sh

echo "Job finished on $(date)"
"""
    with open(base / 'submit.sh', 'w') as f:
        f.write(content)
    os.chmod(base / 'submit.sh', 0o755)
    print("Generated submit.sh (Slurm)")


def generate_pbs_script(args, base):
    """Generate PBS submission script."""
    content = f"""#!/bin/bash
#PBS -N {args.job_name}
#PBS -l walltime={args.max_run_wallclock}
#PBS -l nodes=1:ppn=4
#PBS -l mem=8gb
#PBS -j oe

echo "Job started on $(date)"
echo "Node: $(hostname)"

cd $PBS_O_WORKDIR

# Assume user has configured environment
# source ~/.bashrc
# conda activate westpa

echo "Starting WESTPA..."
./run.sh

echo "Job finished on $(date)"
"""
    with open(base / 'submit.sh', 'w') as f:
        f.write(content)
    os.chmod(base / 'submit.sh', 0o755)
    print("Generated submit.sh (PBS)")


def generate_alt_analysis_script(base, args):
    """Generate analyze_alt_metric.py for post-simulation analysis."""
    
    content = '''#!/usr/bin/env python3
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
            sel_str = """''' + args.pcoord_sel + '''""" # Ignored here, using args.sel from runtime
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
            print(f"  Iteration {n_iter}...", end='\\r')
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
                    
        print("\\nDone processing trajectories.")
        
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
'''
    
    script_path = base / 'analysis' / 'analyze_alt_metric.py'
    with open(script_path, 'w') as f:
        f.write(content)
    os.chmod(script_path, 0o755)


def generate_readme(args, params, base):
    """Generate README.md with usage instructions."""
    
    mab_info = "Enabled (MAB)" if params.get('use_mab') else f"{params['val_min']} to {params['val_max']} ({params['n_bins']} bins)"
    
    content = f"""# WESTPA Simulation

Generated by WESTPA Easy Setup Pipeline

## Overview
- **Progress Coordinate**: {args.pcoord_type.upper()}
- **Selection**: {args.pcoord_sel}
- **Reference**: {args.pcoord_ref if args.pcoord_ref else "N/A"}
- **Binning**: {mab_info}
- **Walkers per bin**: {args.walkers}
- **Max Iterations**: {args.iterations}

## Directory Structure
- `west.cfg`: Main configurations
- `init.sh`: Initialize simulation
- `run.sh`: Start simulation
- `submit.sh`: Job submission script ({args.gen_submit})
- `clean.sh`: Clean for restart
- `bstates/`: Basis states
- `common_files/`: Topology & parameters
- `westpa_scripts/`: Propagator scripts
- `analysis/`: Analysis tools

## Configuration Summary
| Parameter | Value |
|-----------|-------|
| Job Name | {args.job_name} |
| Walltime | {args.max_run_wallclock} |
| Walkers/bin | {args.walkers} |
| Max iterations | {args.iterations} |
| Target RMSD | {args.rmsd_target} Å |
| Bin width | {args.bin_width} Å |
| Temperature | {args.temperature} K |


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
Scripts in `analysis/` folder:
- `plot_pcoord_evolution.py`: Plot pcoord vs iteration
- `calc_free_energy.py`: Calculate Free Energy Surface from simulation pcoord
- `trace_walker.py`: Trace path of the best walker
- `analyze_alt_metric.py`: recalculate alternative metrics (RoG, SASA, etc.) from trajectories

## Troubleshooting

- **pcoord shape error**: Check `west.cfg` pcoord_len matches derived data.
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
    generate_calc_pcoord_py(base, args)
    generate_gen_istate_sh(base)
    generate_get_pcoord_sh(args, files, base)
    generate_helper_scripts(args, base)
    generate_tstate_bstate_files(args, base)
    generate_analysis_scripts(base)
    generate_alt_analysis_script(base, args)
    
    if args.gen_submit == 'slurm':
        generate_slurm_script(args, base)
    elif args.gen_submit == 'pbs':
        generate_pbs_script(args, base)
        
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
