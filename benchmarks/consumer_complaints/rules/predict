rule predict:
    input:
        trained_model = "results/{model}/{classifier}/trained_model.sav",
        x_test = "results/split_data/x_test.csv.zip"
    output:
        predictions = "results/{model}/{classifier}/predictions.csv.zip",
        benchmark = "results/{model}/{classifier}/prediction_benchmark.txt"
    params:
        model = "{model}"
    conda:
        "../envs/hiclass.yml"
    threads:
        config["threads"]
    shell:
        """
        /usr/bin/time -v \
        -o {output.benchmark} \
        python scripts/predict.py \
        --trained-model {input.trained_model} \
        --x-test {input.x_test} \
        --predictions {output.predictions} \
        --classifier {params.model}
        """
