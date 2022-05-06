rule split_data:
    input:
        "data/complaints.csv.zip"
    output:
        x_train = "results/split_data/x_train.csv.zip",
        x_test = "results/split_data/x_test.csv.zip",
        y_train = "results/split_data/y_train.csv.zip",
        y_test = "results/split_data/y_test.csv.zip",
        benchmark = "results/split_data/benchmark.txt"
    params:
        nrows = config["nrows"] if "nrows" in config else None,
        random_state = config["random_state"]
    conda:
        "../envs/hiclass.yml"
    shell:
        """
        /usr/bin/time -v \
        -o {output.benchmark} \
        python scripts/split_data.py \
        --data {input} \
        --x-train {output.x_train} \
        --x-test {output.x_test} \
        --y-train {output.y_train} \
        --y-test {output.y_test} \
        --random-state {params.random_state} \
        --nrows {params.nrows}
        """
