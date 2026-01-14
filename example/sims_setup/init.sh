#!/bin/bash
source env.sh

# Remove old simulation data if present
rm -rf traj_segs seg_logs istates west.h5 west.log

# Initialize WESTPA
w_init --bstates-from bstates/bstates.txt --tstate-file tstate.file --segs-per-state 1 --work-manager=threads

echo "Initialization complete. Run './run.sh' to start the simulation."
