from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider

HTTP = HTTPRemoteProvider()

rule download:
    input:
        ancient(HTTP.remote(config["data"], keep_local=True))
    output:
        "data/complaints.csv.zip"
    shell:
        """
        mv {input} {output}
        """
