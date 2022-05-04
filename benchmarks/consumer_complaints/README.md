# Consumer Complaints Benchmark

This is a reproducible Snakemake pipeline to benchmark HiClass on the consumer complaints dataset released by the [Consumer Financial Protection Bureau](https://www.consumerfinance.gov/data-research/consumer-complaints/).

## Installation

The main requirement to run this pipeline is Anaconda 3. Please, install the latest version of [Anaconda 3](https://www.anaconda.com/products/distribution) on your machine beforehand.

With Anaconda installed, we can create a new environment with snakemake by running the following commands:

```
git clone https://github.com/mirand863/hiclass.git
cd hiclass/benchmarks/consumer_complaints
conda env create --name snakemake --file envs/snakemake.yml
```

The file `config.yml` holds configuration information to run the pipeline, e.g., working directory, number of threads to run tasks in parallel, random seed to enable reproducibility, number of rows to run the pipeline on a subset of the data, the base classifiers from scikit-learn and the hierarchical and flat models. For the purpose of this tutorial, we will keep most parameters intact and modify only the working directory. In order to do that, run the command `pwd` and update the `workdir` parameter with the output from `pwd`.

## Running

After a successful installation, we can activate the newly created environment and run the pipeline (please don't forget to modify the config file with your working directory as described in the last section).

```
conda activate snakemake
snakemake --keep-going --printshellcmds --reason --use-conda --cores 48
```

The parameter --keep-going forces Snakemake to keep executing independent tasks if an unrelated one fails, while the parameter --printshellcmds enables printing the commands that will be executed, the parameter --reason makes Snakemake print the reason for each executed rule, the parameter --use-conda is necessary to indicate that conda will be used to manage the software dependencies of the pipeline, and the parameter --cores tells Snakemake how many cpus can be used overall (the more cpus you can spare, the faster the pipeline will be completed).

The trained models, predictions and benchmarks for each model are saved in the results folder.
