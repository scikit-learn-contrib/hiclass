rule statistics:
    input:
        data = "data/complaints.csv.zip",
        x_train = "results/split_data/x_train.csv.zip",
        y_train = "results/split_data/y_train.csv.zip",
        x_test = "results/split_data/x_test.csv.zip",
        y_test ="results/split_data/y_test.csv.zip",
    output:
        statistics = "results/statistics/statistics.csv",
    conda:
        "../envs/hiclass.yml"
    shell:
        """
        python scripts/statistics.py \
        --data {input.data} \
        --x-train {input.x_train} \
        --y-train {input.y_train} \
        --x-test {input.x_test} \
        --y-test {input.y_test} \
        --statistics {output.statistics}
        """
