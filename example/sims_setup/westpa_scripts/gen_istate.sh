#!/bin/bash

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
