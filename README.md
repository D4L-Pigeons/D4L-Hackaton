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

### Config
Configuration is handled with use of `yaml` files.

#### Managing config files
There are three types of config files separated into `config/data`, `config/models` and `config/trainer`. To get to know the required structure of each file search for `_config_structure: ConfigStructure = {...}` variable is appropriate modules.
Since all config files are logged the name of the config does not have to point to specific experiment and its conent may be flexibly changed.

### Config Structure Type
Objects typed as `ConfigStructure` as used in par with `utils.config.validate_config_strructure` to validate `cfg` objects.

`ConfigStructure` objects may be coposed of object of type `dict`, `[ConfigStructure]`, `(ConfigStructure, ..., ConfigStructure)`, `type`, `None` or `Namespace`.
- `dict` allows to nest the `ConfigStructure` objects
- `[ConfigStructure]` is used when a `cfg` is a list of other `cfg` of expected `ConfigStructure` 
- `(ConfigStructure, ..., ConfigStructure)` validates the `cfg` as long as it matches any of the `ConfigStructure` objects specified in the tuple
- `type` specifies a primitive type like `int` or `str` of which `cfg` needs to be an instance
- 'None' means that the 'cfg' needs to be 'None' (used with `(ConfigStructure, ..., ConfigStructure)` to make a `cfg` object optional)
- `Namespace` means stop of validation. `cfg` is validated by some other module init or it is not a `cfg` object but e.g. a dict parsed with `vars()`


#### Chain Model
`Chain` has the following `_config_structure`:
```python
    _config_structure: ConfigStructure = {
        "chain": [{"chain_link_spec_path": str, "name": str, "cfg": Namespace}],
        "optimizers": [
            {
                "links": [str],
                "optimizer_name": str,
                "kwargs": Namespace,
                "lr_scheduler": (
                    None,
                    {
                        "schedulers": [{"name": str, "kwargs": Namespace}],
                        "out_cfg_kwargs": Namespace,
                    },
                ),
            }
        ],
        "training": {
            "batch_size": int,
            "max_epochs": int,
            "check_val_every_n_epoch": (None, int),
        },
        "loss_manager": Namespace,
        "commands": [
            {
                "command_name": str,
                "processing_steps": [{"link": str, "method": str, "kwargs": Namespace}],
            }
        ],
    }
``` 
When using Chain as a model base keep in mind that consistent naming convention should be established between different configurations to allow easy comparison of logs e.g. decoder part may be cosistently called `decoder`. Debugging requires setting breakpoints in the `run_command` and other called module of interest.

## Experiments

### Prior Conditioned Model - MNIST
To run an experiment run the following line in terminal from project directory
```bash
python3 src/run_pcm_mnist_experiment.py [ARGUMENTS]
```

With the following options:
```
usage: run_pcm_mnist_experiment.py [-h] --data_train DATA_TRAIN --data_val DATA_VAL [--subsample_size SUBSAMPLE_SIZE] [--project_name PROJECT_NAME] [--tags [TAGS ...]] [--experiment_name EXPERIMENT_NAME] --model_cfg MODEL_CFG [--checkpoint_experiment_id CHECKPOINT_EXPERIMENT_ID] [--checkpoint_name CHECKPOINT_NAME]
                                   [--trainer_cfg TRAINER_CFG] [--verbose]

Run PCM MNIST Experiment

options:
  -h, --help            show this help message and exit
  --verbose             Weather to present the status of script run.

Data configuration:
  --data_train DATA_TRAIN
                        Filename of the training data configuration file
  --data_val DATA_VAL   Filename of the validation data configuration file.
  --subsample_size SUBSAMPLE_SIZE
                        Subsample size for the dataset

Experiment configuration:
  --project_name PROJECT_NAME
                        Neptune project name
  --tags [TAGS ...]     List of tags for the experiment
  --experiment_name EXPERIMENT_NAME
                        Name of the experiment

Model configuration:
  --model_cfg MODEL_CFG
                        Filename of the model config file
  --checkpoint_experiment_id CHECKPOINT_EXPERIMENT_ID
                        Id of the experiment required for loading from checkpoint.
  --checkpoint_name CHECKPOINT_NAME
                        Model checkpoint name e.g. 'epoch=1-step-1876.ckpt' required for loading from checkpoint.

Trainer configuration:
  --trainer_cfg TRAINER_CFG
                        Filename of the trainer configuration file.
```

## Dataset


### Metrics


### Logs
Open neptune app or `.neptune` local folder.