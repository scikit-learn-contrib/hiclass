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
	-sed -i "s?workdir.*?workdir: `pwd`?" configs/snakemake.yml
	-sed -i "s?threads.*?threads: 12?" configs/snakemake.yml
	-sed -i "s?#nrows.*?nrows: 2000?" configs/snakemake.yml
	-snakemake --printshellcmds --reason --use-conda --cores 12 --conda-frontend mamba
	-sed -i "s?nrows.*?#nrows: 2000?" configs/snakemake.yml

.PHONY: delab
delab:  ## Run pipeline on delab with slurm
	-sed -i "s?workdir.*?workdir: `pwd`?" configs/snakemake.yml
	-sed -i "s?threads.*?threads: 12?" configs/snakemake.yml
	-sed -i "s?nrows.*?#nrows: 2000?" configs/snakemake.yml
	- srun -A renard --mem=30G --cpus-per-task=12 --time=30-00:00:00 -p cauldron snakemake --keep-going --printshellcmds --reason --use-conda --cores 1 --resources mem_gb=2700 --restart-times 5 --jobs 6 --cluster "sbatch -A renard -p magic --mem=450G --cpus-per-task=12 --time=5-00:00:00" --dry-run

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
