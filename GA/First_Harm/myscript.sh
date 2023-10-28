#!/bin/sh
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
time /apps/bin/python3 ~/models/XSENSE2_sin/XSENSE.py
