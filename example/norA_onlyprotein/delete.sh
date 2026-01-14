#!/bin/bash
# Clean up simulation data for fresh restart
# NOTE: Keep bstates/ - it has your equilibrated starting structure!

rm -rf traj_segs
rm -rf istates
rm -rf seg_logs
rm -f west.h5
rm -f west.log
rm -f west_copy.h5
rm -f analysis/fluxanl.h5