#!/bin/bash
# Job Parameters
#SBATCH --job-name=config
#SBATCH --array=0-21
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=21
#SBATCH --time=2-00:00:0
#SBATCH --mem=20G
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=rancze@crick.ac.uk
#SBATCH --output=reports/%A_%a_%u_report.out
#SBATCH --error=errors/%A_%a_%u_log.err

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_ARRAY_TASK_ID"=$SLURM_ARRAY_TASK_ID

ml purge
ml Anaconda3
#source activate /camp/home/tranvaa/.conda/envs/nxenv

python grid_config_model.py $SLURM_ARRAY_TASK_ID