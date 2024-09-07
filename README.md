# D4L-Hackaton

## Setup
We recommend creating ```venv``` or ```conda``` environment with ```python==3.12```.

### Conda and requirements.txt

```bash
conda create -n multimodal_conditions python=3.12
source activate multimodal_conditions
```

And then:
```bash
pip3 install -r requirements.txt
```

### Managing config files
There are three types of config files separated into `config/data`, `config/models` and `config/trainer`. To get to know the required structure of each file search for `_config_structure: ConfigStructure = {...}` variable is appropriate modules.

## Overview

### Prior Conditioned Model - MNIST
To run an experiment run the following line in terminal from project directory
```bash
python3 src/run_pcm_mnist_experiment.py [ARGUMENTS]
```

with arguments:
  * ```-h, --help```: show this help message and exit.
  * ```--project_name``` (default='multimodal/vaes'): Neptune project name.
  * ```--experiment_name```: Neptune experiment name.
  * ```--batch_size``` (default=128): Batch size.
  * ```--subsample_size``` (default=None): Size of the subsample.
  * ```--max_epochs``` (default=None): Maximum number of traininig epochs.
  * ```--check_val_every_n_epoch```: Frequency of validation.
  * ```--model_cfg```: Model config file filename.
  * ```--data_train```: Train dataset config file filename.
  * ```--data_val``` : Validation dataset config file filename.
  * ```--trainer_cfg```: Trainer config file filename.

## Dataset


### Metrics


### Viewing logs
