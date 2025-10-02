#!/bin/bash -l

#$ -l h_rt=0:10:0
#$ -l mem=1G
#$ -pe smp 1

#$ -N MASTER

#$ -j y
#$ -o $HOME/Scratch/ChemGridML/_master.log
#$ -wd $HOME/Scratch/ChemGridML

# Create output directory
mkdir -p ./output/$JOB_ID

conda activate ChemGridML

# Run master script with job ID and experiment names
python ./code/master.py CLUSTER $JOB_ID "$@"

# Move log to output directory
mv ./_master.log ./output/$JOB_ID