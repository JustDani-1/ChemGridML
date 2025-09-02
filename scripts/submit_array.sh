#!/bin/bash -l

# Batch script to run a serial array job under SGE.

# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=12:00:0

# Request 1 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=8G

# Set up the job array. Upper bound specifies the number of jobs
#$ -t 1-144

# Set the name of the job.
#$ -N AllArray

# Set the working directory to somewhere in your scratch space. 
#$ -wd /home/ucnwdma/Scratch/UCL_internship

conda activate internship

python ./code/main.py $JOB_ID $SGE_TASK_ID

mkdir -p ./output/$JOB_ID
