#!/bin/bash
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
