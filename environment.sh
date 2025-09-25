#!/bin/bash

# ChemGridML Environment Setup Script

echo "Setting up ChemGridML conda environment..."

# Deactivate any active environment
conda deactivate

# Remove existing environment if it exists
echo "Removing existing ChemGridML environment (if it exists)..."
conda remove --name ChemGridML --all -y

# Create new environment
echo "Creating new ChemGridML environment with Python 3.9..."
conda create -n ChemGridML python=3.9 -y

# Activate environment
echo "Activating ChemGridML environment..."
conda activate ChemGridML

# Install main packages
echo "Installing main packages..."
conda install -c conda-forge numpy pandas pytorch transformers torchdata torch-geometric tqdm matplotlib optuna xgboost lightning jax dm-haiku plotly -y

# Install chemistry-specific packages
echo "Installing chemistry-specific packages..."
conda install -c conda-forge dgl dgllife rdkit pytdc deepchem gensim joblib scikit-learn -y

# Install CUDA toolkit
echo "Installing CUDA toolkit..."
conda install cudatoolkit cuda-version=11 -y

# Install mol2vec via pip
echo "Installing mol2vec..."
pip install git+https://github.com/samoturk/mol2vec

# Install specific torchdata version
echo "Installing specific torchdata version..."
conda install torchdata=0.7.1 -c conda-forge -y

echo "Environment setup complete!"
echo "To activate the environment, run: conda activate ChemGridML"