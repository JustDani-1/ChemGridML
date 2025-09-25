# ChemGridML: A small framework for high-throughput molecular ML experiments on SGE clusters

This project originated from my internship "Leveraging computational chemistry to optimise Machine Learning Models in Drug Discovery" at the UCL School of Pharmacy.

Its purpose is to provide an easy-to-use interface for molecular machine learning, supporting both local execution and SGE-based HPC systems (developed using UCL's Myriad cluster). A collection of molecular featurization methods and ML models are included, with code designed to be easily extensible. The cluster acceleration enables systematic exploration of many feature-model combinations to improve performance for a given dataset.

## Setup

### Environment

There are two possbile ways to created the conda environment:

#### Option 1: Create environment from file

```console
conda env create -f environment.yml
```

#### Option 2: Using setup script

```console
# Make script executable
chmod +x environment.sh

# Run setup script
./environment.sh
```

### HPC clusters

In order to use a cluster, log onto it and execute:

```console
# Go into home directory
cd ~
# Create Scratch directory if not exists
mkdir -p Scratch
cd Scratch
# Clone this repository
git clone https://github.com/JustDani-1/ChemGridML.git
cd ChemGridML
```

Afterwards, set up your conda environment as described above.

## Getting Started

ChemGridML uses the following abstraction for executing ML experiments:

![Experiment Composition](assets/strategy.png)

### Local

### HPC clusters

The syntax for running experiments is:

```console
qsub ./scripts/submit_master.sh <experiment1> [experiment2] ...
```

For example:

```console
qsub ./scripts/submit_master.sh FINGERPRINT
```