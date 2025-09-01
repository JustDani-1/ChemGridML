#!/bin/bash -l

# Batch script to run a serial array job under SGE.

# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=0:2:0

# Request 1 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=1G

# Set up the job array.  In this instance we have requested 10000 tasks
# numbered 1 to 100.
#$ -t 1-20

# Set the name of the job.
#$ -N MyArrayJob

# Set the working directory to somewhere in your scratch space. 
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/ucnwdma/Scratch/UCL_internship/output

# Run the application.

echo "$JOB_NAME $SGE_TASK_ID"