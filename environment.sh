#!/bin/bash
# Conda environment setup for molecular property prediction
# Handles both CUDA and CPU-only systems

# Initialize conda for this shell session
eval "$(conda shell.bash hook)"

# Safely deactivate if in an environment
if [[ $CONDA_DEFAULT_ENV == "internship" ]]; then
    echo "Deactivating current internship environment..."
    conda deactivate
fi

# Remove existing environment if it exists
conda env remove --name internship --yes 2>/dev/null || true
conda create -n internship python=3.9
conda activate internship

# Core packages (CPU/GPU agnostic)
conda install -c conda-forge numpy transformers pandas tqdm gensim joblib scikit-learn matplotlib optuna xgboost lightning plotly

# Chemistry and molecular modeling packages
conda install -c conda-forge rdkit pytdc deepchem

# Install PyTorch with CUDA support (fallback to CPU if no CUDA)
# This will automatically detect and install appropriate version
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install TensorFlow (GPU-enabled by default, falls back to CPU)
conda install tensorflow -c conda-forge

# Graph learning packages
conda install -c conda-forge torch-geometric dgl dgllife

# JAX installation with CUDA 11 support
# Note: Will fallback gracefully to CPU if CUDA not available
pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install dm-haiku -c conda-forge

# Fixed version for compatibility
conda install torchdata=0.7.1 -c conda-forge

# Install mol2vec from git
pip install git+https://github.com/samoturk/mol2vec

# Optional: Test GPU availability
echo "Testing GPU availability..."
python -c "
import tensorflow as tf
import torch
import jax

print('=== GPU Availability Check ===')
print(f'TensorFlow GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
print(f'JAX devices: {jax.devices()}')

if len(tf.config.list_physical_devices('GPU')) > 0 or torch.cuda.is_available() or 'gpu' in str(jax.devices()).lower():
    print('GPU acceleration available!')
else:
    print('Running on CPU (GPU not detected)')
"