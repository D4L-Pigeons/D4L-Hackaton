# D4L-Hackaton

## Setup
We recommend creating ```venv``` or ```conda``` environment with ```python==3.12```.

### Conda and requirements.txt

```bash
conda create -n multimodal_hackaton python=3.12
source activate multimodal_hackaton
```

And then:
```bash
pip3 install -r requirements.txt
```

## Overview

### Prior Conditioned Model - MNIST
To run an experiment the following line in terminal from project directory
```bash
python3 src/run_pcm_mnist_experiment.py [ARGUMENTS]
```

with arguments:
  * ```-h, --help```: show this help message and exit.
  * ```--data_train```: Train dataset config file filename.
  * ```--data_val``` : Validation dataset config file filename.
  * ```--batch_size``` (default=128): Batch size.
  * ```--subsample_size``` (default=None): Size of the subsample.
  * ```--project_name``` (default='multimodal/vaes'): Neptune project name.
  * ```--experiment_name```: Neptune experiment name.
  * ```--model_cfg```: Model config file filename.
  * ```--max_epochs``` (default=None): Maximum number of traininig epochs.
  * ```--check_val_every_n_epoch```: Frequency of validation.
  * ```--trainer_cfg```: Trainer config file filename.

## Dataset


### Metrics


### Viewing logs
