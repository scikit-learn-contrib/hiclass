.PHONY: help
help: ## display this help
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: mamba
mamba:  ## install mamba
	conda install -n base -c conda-forge mamba

.PHONY: snakemake
snakemake:  ## install snakemake
	-mamba remove -n snakemake --all -y
	-mamba create -c conda-forge -c bioconda -n snakemake snakemake -y

.PHONY: local
local:  ## Run pipeline without slurm
	-sed -i 's/^#//' configs/submitit.yaml
	-sed -i '/^  \- override hydra\/launcher: submitit\_slurm/ s/./#&/' configs/submitit.yaml
	-sed -i '/^    account: renard/ s/./#&/' configs/submitit.yaml
	-sed -i '/^    partition: magic/ s/./#&/' configs/submitit.yaml
	-sed -i "s?workdir.*?workdir: `pwd`?" config.yml
	-sed -i "s?threads.*?threads: 12?" config.yml
#	-sed -i "s?#nrows.*?nrows: 1000?" config.yml
	-snakemake --keep-going --printshellcmds --reason --use-conda --cores 12 --conda-frontend mamba

.PHONY: delab
delab:  ## Run pipeline on delab with slurm
	-sed -i 's/^#//' configs/submitit.yaml
	-sed -i '/^  \- override hydra\/launcher: submitit\_local/ s/./#&/' configs/submitit.yaml
	-sed -i "s?workdir.*?workdir: `pwd`?" config.yml
	-sed -i "s?threads.*?threads: 12?" config.yml
	-sed -i "s?nrows.*?#nrows: 1000?" config.yml
	- srun -A renard --mem=30G --ntasks=72 --time=30-00:00:00 -p cauldron snakemake --keep-going --printshellcmds --reason --use-conda --cores 72 --resources mem_gb=2700 --conda-frontend mamba
#	-sbatch submit.slurm

.PHONY: clean
clean:  ## delete temporary files from Snakemake
	-rm -rf .snakemake

.PHONY: delete-results
delete-results:  ## delete results from pipeline
	-rm -rf results

.PHONY: git
git:  ## Update local repository
	-git reset --hard
	-git fetch
	-git pull
