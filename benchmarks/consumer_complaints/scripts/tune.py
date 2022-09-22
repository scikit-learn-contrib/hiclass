import submitit
from time import sleep

def add(a, b):
    sleep(5)
    return a + b

# executor is the submission interface (logs are dumped in the folder)
executor = submitit.AutoExecutor(folder="log_test")
# set timeout in min, and partition for running the job
executor.update_parameters(timeout_min=1, slurm_partition="magic", slurm_array_parallelism=2)
a = [1, 2, 3, 4]
b = [10, 20, 30, 40]
jobs = executor.map_array(add, a, b)  # just a list of jobs
print([job.result() for job in jobs])