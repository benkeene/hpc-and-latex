#!/bin/bash
module load anaconda3
conda init bash
source ~/.bashrc
conda create --prefix /ocean/projects/cis240124p/$(whoami)/envs/hpclatex python=3.12 -y
conda activate /ocean/projects/cis240124p/$(whoami)/envs/hpclatex
pip install -r requirements.txt