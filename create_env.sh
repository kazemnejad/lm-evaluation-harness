#!/bin/bash

cd $SLURM_TMPDIR
module load miniconda
conda create -y -p venv python=3.9
conda activate ./venv
conda install -y pytorch torchvision torchaudio -c pytorch
ln -snf ~/codes/lm-evaluation-harness/ code
cd code
pip install -e .
pip install psutil pytest
export TRANSFORMERS_CACHE=~/scratch/experiments/hf_cache_dir
export HF_DATASETS_CACHE=~/scratch/experiments/hf_ds_cache
export HUGGINGFACE_HUB_CACHE=~/scratch/experiments/hf_hub_cache
cp run_all_phase_shift_experiments.py run_all_phase_shift_experiments2.py
nohup python run_all_phase_shift_experiments2.py >logs/125m.log 2>&1 &