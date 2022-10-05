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

.PHONY: laptop
laptop:  ## Run methods pipeline on laptop
	-sed -i "s?workdir.*?workdir: `pwd`?" config.yml
	-snakemake --keep-going --printshellcmds --reason --use-conda --cores 12 --conda-frontend mamba

.PHONY: delab
delab:  ## Run methods pipeline on delab
	-sed -i "s?workdir.*?workdir: `pwd`?" config.yml
	-srun -A renard --mem=450G --cpus-per-task=256 --time=120:0:0 -p magic snakemake --keep-going --printshellcmds --reason --use-conda --cores 256 --restart-times 10 --resources mem_gb=450 --conda-frontend mamba

.PHONY: clean
clean:  ## delete temporary files from pipeline
	-rm -rf .snakemake

.PHONY: delete-results
delete-results:  ## delete results from pipeline
	-rm -rf results

.PHONY: git
git:  ## Update local repository
	-git reset --hard
	-git fetch
	-git pull
