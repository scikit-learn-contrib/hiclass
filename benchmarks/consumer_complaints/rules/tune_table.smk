rule tune_table:
    input:
        best_parameters = "results/{model}/{classifier}/optimization_results.yaml",
    output:
        table = "results/{model}/{classifier}/optimization_results.md",
    params:
        model = "{model}",
        classifier = "{classifier}",
        folder = "results/{model}/{classifier}",
    conda:
        "../envs/hiclass.yml"
    threads:
        config["threads"]
    resources:
        mem_gb = config["mem_gb"]
    shell:
        """
        python scripts/tune_table.py \
        --folder {params.folder} \
        --model {params.model} \
        --classifier {params.classifier} \
        --output {output.table}
        """
