#!/bin/bash
source env.sh
rm -rf traj_segs seg_logs istates west.h5 
mkdir   seg_logs traj_segs istates
BSTATE_ARGS="--bstate-file $WEST_SIM_ROOT/bstates/bstates.txt"
TSTATE_ARGS="--tstate-file $WEST_SIM_ROOT/tstate.file"
w_init \
  $BSTATE_ARGS \
  $TSTATE_ARGS \
  --segs-per-state 4 \
  --work-manager=threads "$@"
