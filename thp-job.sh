#!/bin/tcsh
#SBATCH --job-name=thp
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --gpus=1

# load and activate conda
module load anaconda3/2023.09
conda activate torch-env

# ensure we're in the correct directory
cd ~/classes/genai/project/Transformer-Hawkes-Process

# run the script in this directory and save outputs to file
./run.sh

# print something to shell as confirmation
echo "Complete"