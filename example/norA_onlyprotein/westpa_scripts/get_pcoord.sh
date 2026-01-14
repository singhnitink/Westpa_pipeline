#!/bin/bash

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

cd $WEST_SIM_ROOT

# Determine input file
if [ -d "$WEST_STRUCT_DATA_REF" ]; then
    # Prefer DCD if available for MDTraj, else seg.coor (if supported or if we convert)
    # seg.coor is NAMD binary restart, MDTraj might not support.
    # We added seg.dcd to bstates/eq0.
    if [ -f "$WEST_STRUCT_DATA_REF/seg.dcd" ]; then
        INPUT="$WEST_STRUCT_DATA_REF/seg.dcd"
    else
        # Fallback to coor? or pdb if valid. 
        INPUT="$WEST_STRUCT_DATA_REF/seg.coor" 
    fi
else
    INPUT="$WEST_STRUCT_DATA_REF"
fi

python $WEST_SIM_ROOT/westpa_scripts/calc_pcoord.py \
    $WEST_SIM_ROOT/common_files/norA_inopen.pdb \
    $WEST_SIM_ROOT/common_files/norA_outopen.pdb \
    $INPUT \
    $WEST_PCOORD_RETURN \
    --last-only

