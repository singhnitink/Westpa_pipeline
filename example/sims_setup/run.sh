#!/bin/bash
source env.sh

echo "Starting WESTPA simulation with 4 workers..."
echo "Output will be logged to west.log"

w_run --n-workers 4 &> west.log &
echo "Simulation started in background. PID: $!"
echo "Use 'tail -f west.log' to monitor progress."
