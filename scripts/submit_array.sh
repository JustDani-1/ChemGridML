#!/bin/bash -l

# Batch script to run a serial array job under SGE.

# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=0:20:0

# Request 1 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=2G

# Set up the job array.  In this instance we have requested 10000 tasks
# numbered 1 to 10000.
#$ -t 1-144

# Set the name of the job.
#$ -N FirstArrayJob

# Set the working directory to somewhere in your scratch space. 
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/ucnwdma/Scratch/UCL_internship

# Notify phone
curl -d "Myriad: ntfy_job running" ntfy.sh/myriad_job_running

conda activate internship

python ./code/main.py $SGE_TASK_ID