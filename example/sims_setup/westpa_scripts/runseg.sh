#!/bin/bash

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

cd $WEST_SIM_ROOT
mkdir -pv $WEST_CURRENT_SEG_DATA_REF
cd $WEST_CURRENT_SEG_DATA_REF

# Link common files
ln -sf $WEST_SIM_ROOT/common_files/norA_inopen.psf .
ln -sf $WEST_SIM_ROOT/common_files/norA_inopen.pdb .
ln -sf $WEST_SIM_ROOT/common_files/toppar .
ln -sf $WEST_SIM_ROOT/common_files/norA_outopen.pdb .

# Link parent restart files
ln -sv $WEST_PARENT_DATA_REF/seg.coor ./parent.coor
ln -sv $WEST_PARENT_DATA_REF/seg.vel  ./parent.vel
ln -sv $WEST_PARENT_DATA_REF/seg.xsc  ./parent.xsc

# Prepare input file from template
sed "s/RAND/$WEST_RAND16/g" $WEST_SIM_ROOT/common_files/md.conf > md.in

# Run NAMD (output goes to stdout, WESTPA captures it to seg_logs/)
namd3 +p2 +devices 0 md.in

# Calculate Progress Coordinate (RMSD) using Python/MDTraj
python $WEST_SIM_ROOT/westpa_scripts/calc_pcoord.py \
    norA_inopen.pdb \
    norA_outopen.pdb \
    seg.dcd \
    $WEST_PCOORD_RETURN

# Cleanup
rm -f md.in seg.restart.* parent.*
