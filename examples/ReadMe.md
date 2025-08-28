# Examples
This folder contains a small number of examples to help you get started. The examples are config files that overwrite default settings.
 You can find out more in the [documentation](https://pik-piam.github.io/PyPSA-China-PIK/) and by going through the default configuration files (`configs/default_config.yaml`, (`configs/technology_config.yaml`).

## runing the examples
After installing the workflow, run the examples with the following command (e.g. for myopic)
`snakemake -configfile=examples/myopic.yaml --cores 1`