#!/bin/bash
#SBATCH --job-name=snakemake                       # Job name
#SBATCH -A renard                                  # Account
#SBATCH --partition cauldron                       # Partition
#SBATCH --mail-type=END,FAIL                       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=fabio.malchermiranda@hpi.de    # Where to send mail
#SBATCH --nodes=1                                  # Run all processes on a single node
#SBATCH --ntasks=72                                # Number of processes
#SBATCH --mem=30gb                                 # Total memory limit
#SBATCH --time=336:00:00                           # Time limit hrs:min:sec
#SBATCH --output=snakemake_%j.log # Standard output and error log
date && hostname && pwd

conda activate snakemake

snakemake --keep-going --printshellcmds --reason --use-conda --cores 72 --resources mem_gb=2700 --conda-frontend mamba

date