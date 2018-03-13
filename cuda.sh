#!/bin/bash
#$ -N REDUCE
#$ -q gpu
#$ -l gpu=1
#$ -pe gpu-node-cores 6

module load  cuda/5.0
module load  gcc/4.4.3

# Runs a bunch of standard command-line
# utilities, just as an example:

echo "Script began:" `date`
echo "Node:" `hostname`
echo "Current directory: ${PWD}"

echo ""
echo "=== Running 5 trials of naive ... ==="
  ./s1

echo ""
echo "=== Done! ==="

# eof
