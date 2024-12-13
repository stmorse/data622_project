#!/bin/tcsh
#SBATCH --job-name=thp
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

# load and activate conda
module load anaconda3/2023.09
conda activate torch-tik-env

# ensure we're in the correct directory
cd ~/classes/genai/project/MHP

# run the script in this directory and save outputs to file
python -u mhp_run.py > log.txt

# print something to shell as confirmation
echo "Complete"