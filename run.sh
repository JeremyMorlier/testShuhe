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

source .venvbatch/bin/activate
srun python3 few_shot_resnet12_vits.py --num-epochs 100 --val_crop_size 84 --train_crop_size 84 --val_resize_size 84 --student_input_size 84 --batch_size 376 