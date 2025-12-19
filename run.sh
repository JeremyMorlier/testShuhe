#!/bin/bash
#SBATCH --job-name=CA # nom du job
#SBATCH --output=log/CA/%j/logs.out # fichier de sortie (%j = job ID)
#SBATCH --error=log/CA/%j/errors.err # fichier d’erreur (%j = job ID)
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 #reserver 10 taches (ou processus)
#SBATCH --time=168:00:00 # temps d'allocation
#SBATCH --cpus-per-gpu=48
#SBATCH --gres=gpu:a100:1
#SBATCH -p BrainA100

source .venv/bin/activate
srun python3 few_shot_resnet12_vits.py