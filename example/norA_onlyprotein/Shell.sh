#!/bin/bash
#conda activate westpa
./init.sh                           # Creates new west.h5
w_run --n-workers 4 &> west.log &   # Start simulation