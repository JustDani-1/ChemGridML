#!/bin/bash -l

# Batch script to run a serial job under SGE.

# Request a number of GPU cards, in this case 2 (the maximum)
#$ -l gpu=1

# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=0:20:0

# Request 1 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=8G

# Request 15 gigabyte of TMPDIR space (default is 10 GB - remove if cluster is diskless)
#$ -l tmpfs=10G

# Set the name of the job.
#$ -N first_test

# Set the working directory to somewhere in your scratch space.  
# This is a necessary step as compute nodes cannot write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID.
#$ -wd /home/ucnwdma/Scratch/UCL_internship/output

cp -r /home/ucnwdma/Scratch/UCL_internship/code/ $TMPDIR/
cp -r /home/ucnwdma/Scratch/UCL_internship/models/ $TMPDIR/

cd $TMPDIR
mkdir output
cd ./code

conda activate internship

python ./main.py

# Preferably, tar-up (archive) all output files onto the shared scratch area
tar -zcvf $HOME/Scratch/UCL_internship/output/files_from_job_$JOB_ID.tar.gz $TMPDIR/output/

# Make sure you have given enough time for the copy to complete!
