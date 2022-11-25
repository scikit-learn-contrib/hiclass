# Consumer Complaints Benchmark

This is a reproducible Snakemake pipeline to benchmark HiClass on the consumer complaints dataset released by the [Consumer Financial Protection Bureau](https://www.consumerfinance.gov/data-research/consumer-complaints/).

## Installation

The main requirement to run this pipeline is Anaconda 3. Please, install the latest version of [Anaconda 3](https://www.anaconda.com/products/distribution) on your machine beforehand.

With Anaconda installed, you can create a new environment with snakemake by running the following commands:

```shell
git clone https://github.com/mirand863/hiclass.git
cd hiclass/benchmarks/consumer_complaints
conda env create --name snakemake --file envs/snakemake.yml
```

The file `configs/snakemake.yml` holds configuration information to run the pipeline, e.g.:
- working directory;
- number of threads to run tasks in parallel;
- random seed to enable reproducibility when splitting the dataset;
- number of splits for k-fold cross validation;
- allocated memory in GB for tuning and training the models;
- number of rows to run the pipeline on a subset of the data;
- the base classifiers from scikit-learn;
- the hierarchical and flat models.

For the purpose of this tutorial, we will keep most parameters intact and modify only the working directory. In order to do that, run the command `pwd` and update the `workdir` parameter with the output from `pwd`. Alternatively, execute the following command to update the working directory automatically:

```shell
sed -i "s?workdir.*?workdir: `pwd`?" configs/snakemake.yml
```

## Hyperparameter tuning

The hyperparameters tested during the tuning stage are defined in the files [lightgbm.yaml](configs/lightgbm.yaml), [logistic_regression.yaml](configs/logistic_regression.yaml) and [random_forest.yaml](configs/random_forest.yaml). For example, for random forest we used the following `yaml` file:

```yml
defaults:
  - _self_
  - optuna

hydra:
  sweeper:
    params:
      n_estimators: choice(100, 200)
      criterion: choice("gini", "entropy", "log_loss")

n_estimators: 1
criterion: 1
```

The intervals for testing can be defined with the functions `range` or `choice`, as described on [Hydra's documentation](https://hydra.cc/docs/plugins/optuna_sweeper/). If you wish to add more parameters for testing, you can simply add the parameter name inside the `params` field and at the end of the file set it to 1 in order to enable its usage in Hydra.

## Running locally

After a successful installation, you can activate the newly created environment and run the pipeline locally (please don't forget to modify the config file with your working directory as described in the last section).

```shell
conda activate snakemake
snakemake --keep-going --printshellcmds --reason --use-conda --cores 48
```

The parameter --keep-going forces Snakemake to keep executing independent tasks if an unrelated one fails, while the parameter --printshellcmds enables printing the commands that will be executed, the parameter --reason makes Snakemake print the reason for each executed rule, the parameter --use-conda is necessary to indicate that conda will be used to manage the software dependencies of the pipeline, and the parameter --cores tells Snakemake how many cpus can be used overall (the more cpus you can spare, the faster the pipeline will finish).

The trained models, predictions and benchmarks for each model are saved in the results folder.

## Running on clusters

Running the pipeline on a Slurm cluster requires using more parameters in order to inform Snakemake the rules for submitting jobs:

```shell
srun \
--account <name> \
--mem=<memory-snakemake>G \
--cpus-per-task=<cpus> \
--time=<days>-<hours>:<minutes>:<seconds> \
--partition <partition> \
snakemake \
--keep-going \
--printshellcmds \
--reason \
--use-conda \
--cores <cores> \
--resources mem_gb=<total-memory-for-all-jobs> \
--restart-times <restart> \
--jobs <jobs> \
--cluster \
"sbatch \
--account <account> \
--partition <partition> \
--mem=<memory-per-job>G \
--cpus-per-task=<cpus-per-job> \
--time=<days>-<hours>:<minutes>:<seconds>"
```

`srun` is the command used to submit Snakemake to Slurm. The parameters have the following meanings:

- `--account` specifies which account to use when submitting jobs to Slurm;
- `--mem` specifies how much memory should be used by Snakemake (not necessary to be a large value since individual jobs will have more memory allocated in another parameter that will be described later);
- `--cpus-per-task` is the number of cores used by Snakemake;
- `--time` states how long the Snakemake job should run for;
- `--partition` designates the partition to run the job.

Regarding the parameters for Snakemake, they have the following meanings:
- `--keep-going` forces Snakemake to keep executing independent tasks if an unrelated one fails;
- `--printshellcmds` enables printing the commands that will be executed;
- `--reason` makes Snakemake print the reason for each executed rule;
- `--use-conda` is necessary to indicate that conda will be used to manage the software dependencies of the pipeline;
- `--cores` tells Snakemake how many cpus can be used overall (the more cpus you can spare, the faster the pipeline will finish). For a cluster execution, 12 cores is more than enough since individual jobs will have more CPUs allocated later;
- `--resources mem_gb` specifies the total ammount of RAM that should be allocated for all jobs (only used during tuning and training);
- `--restart-times` defines how many times the pipeline should restart a job if it fails. This could be useful if the reason for failing is out of memory, since each retry will allocate more memory for the failed job;
- `--jobs` the number of jobs that can be submitted simultaneously;
- `--cluster` the parameters for individual jobs can be set inside the quotation marks.

The parameters inside quotation marks describe the rules for each individual job submitted by Snakemake, i.e.:
- `--account` the account to submit jobs;
- `--partition` the partition to run jobs;
- `--mem` memory allocated for each job. This parameter is combined with the number of retries as `retry * mem`. For example, if we initially allocate 500GB for each job, then at the third retry it will allocate `3 * 500GB = 1.5TB`. Snakemake is careful not to exceed the maximum memory usage specified in the parameter `--resources mem_gb`;
- `--cpus-per-task` the number of CPUs allocated for each job submitted by Snakemake;
- `--time` how long each job can run for;

Here is more information on how to run Snakemake using other [cluster engines](https://snakemake.readthedocs.io/en/stable/executing/cluster.html) and the [cloud](https://snakemake.readthedocs.io/en/stable/executing/cloud.html). In the [Makefile](Makefile) you can check the parameters we used to submit jobs to our cluster.

## Results

The results generated by the pipeline can be found in the [results](results/) folder. For example, for the flat model with LightGBM as the base classifier we have the following files:
- [metrics.csv](results/flat/lightgbm/metrics.csv): the hierarchical F-score when using 70% of the data for training and 30% for testing;
- [optimization_results.md](results/flat/lightgbm/optimization_results.md): all the results from hyperparameter tuning, i.e., parameters tested, their respective F-scores, the average F-score and the standard deviation;
- [optimization_results.yaml](results/flat/lightgbm/optimization_results.yaml): the best parameters found by hyperparameter tuning;
- [training_benchmark.txt](results/flat/lightgbm/training_benchmark.txt): the time and memory consumption for training the best model on 70% of the data;
- [prediction_benchmark.txt](results/flat/lightgbm/prediction_benchmark.txt): the time and memory consumption for predicting with the best model on 30% of the data.
