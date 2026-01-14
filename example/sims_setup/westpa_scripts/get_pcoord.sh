#!/bin/bash

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
ln -s $WEST_SIM_ROOT/common_files/norA_inopen.pdb .
ln -s $WEST_SIM_ROOT/common_files/norA_outopen.pdb .

# Run calc_pcoord.py
# Input 1: Topology (System PDB)
# Input 2: Reference (Target PDB) - used for RMSD, ignored for others
# Input 3: Trajectory (System PDB) - treating the start structure as a 1-frame traj
# Input 4: Output File

python $WEST_SIM_ROOT/westpa_scripts/calc_pcoord.py \
    norA_inopen.pdb \
    norA_outopen.pdb \
    norA_inopen.pdb \
    $WEST_PCOORD_RETURN

# Cleanup
cd $WEST_SIM_ROOT
rm -rf $TMPDIR
