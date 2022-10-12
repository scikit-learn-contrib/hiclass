#!/bin/bash
#SBATCH --job-name=snakemake                       # Job name
#SBATCH -A renard                                  # Account
#SBATCH --partition cauldron                       # Partition
#SBATCH --mail-type=END,FAIL                       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=fabio.malchermiranda@hpi.de    # Where to send mail
#SBATCH --nodes=1                                  # Run all processes on a single node
#SBATCH --ntasks=144                               # Number of processes
#SBATCH --mem=60gb                                 # Total memory limit
#SBATCH --output=snakemake_%j.log # Standard output and error log
date && hostname && pwd

conda activate snakemake

snakemake --keep-going --printshellcmds --reason --use-conda --cores 144 --restart-times 4 --resources mem_gb=4800 --conda-frontend mamba

date