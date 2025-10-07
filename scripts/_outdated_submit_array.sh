#!/bin/bash -l

#$ -l h_rt=10:00:0
#$ -l mem=8G
#$ -t 1-12
#$ -pe smp 5

#$ -N Graph

#$ -j y
#$ -wd $HOME/Scratch/UCL_internship

conda activate internship

python ./code/main.py $JOB_ID $SGE_TASK_ID

mkdir -p ./output/$JOB_ID
