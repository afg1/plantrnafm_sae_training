#!/bin/bash

#Submit this script with: sbatch thefilename
#For more details about each parameter, please check SLURM sbatch documentation https://slurm.schedmd.com/sbatch.html

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=1   # number of tasks
#SBATCH --cpus-per-task=8   # number of CPUs Per Task i.e if your code is multi-threaded
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --mem=64G   # memory per node
#SBATCH -J "SAE_training"   # job name
#SBATCH -o "out_sae_train"   # job output file
#SBATCH -e "err_sae_train"   # job error file
#SBATCH --mail-user=agreen@ebi.ac.uk   # email address
#SBATCH --mail-type=END


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE


source .venv/bin/activate

export TRACKIO_DIR=/hps/nobackup/agb/rnacentral/agreen/ARISE/mech_interp/plantrnafm_sae_training/trackio


expansions=(16 32 64)
layers=(0 1 2 3 4 5 6 7 8 9 10 11 12)
sparsities=(32 64 128 256 512)

for expansion in "${expansions[@]}"; do
    for layer in "${layers[@]}"; do
        for sparsity in "${sparsities[@]}"; do
            echo "Starting training for layer $layer with sparsity $sparsity and expansion $expansion"
            python main.py --layer "$layer" --sparsity "$sparsity" --expansion_factor "$expansion" --device cuda --training_steps 10000 --base_save_dir trained_saes
            echo "Finished training for layer $layer with sparsity $sparsity and expansion $expansion"
        done
    done
done
for layer in "${layers[@]}"; do
    for sparsity in "${sparsities[@]}"; do
        echo "Starting training for layer $layer with sparsity $sparsity"
        python main.py --layer "$layer" --sparsity "$sparsity" --device cuda --training_steps 10000 --base_save_dir trained_saes
        echo "Finished training for layer $layer with sparsity $sparsity"
    done
done
