# ChemGridML: A small framework for high-throughput molecular ML experiments on SGE clusters

This project originated from my internship "Leveraging computational chemistry to optimise Machine Learning Models in Drug Discovery" at the UCL School of Pharmacy.

Its purpose is to provide an easy-to-use interface for molecular machine learning, supporting both local execution and SGE-based HPC systems (developed using UCL's Myriad cluster). A collection of molecular featurization methods and ML models are included, with code designed to be easily extensible. The cluster acceleration enables systematic exploration of many feature-model combinations to improve performance for a given dataset.

## Setup

### Environment

There are two possbile ways to created the conda environment needed:

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

### Prerequisites

Anaconda or Miniconda installed
CUDA-compatible GPU (recommended)

## Installation

Option 1: Using environment.yml (Recommended)
bash# Clone the repository
git clone 
cd ChemGridML

### Create environment from file

conda env create -f environment.yml

### Activate environment

conda activate ChemGridML
Option 2: Using setup script

```console
# Make script executable
chmod +x environment.sh

### Run setup script
./environment.sh
```
