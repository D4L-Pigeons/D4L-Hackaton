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
When you want to run any experiment, run:
```bash
cd src
```
and then
```bash
python3 train_and_validate.py [ARGUMENTS]
```

with possible options:
  * ```-h, --help```: show this help message and exit
  * ```--method``` {"omivae", "babel", "advae"} 
  * ```--config``` (default="standard"): Name of a configuration in src/config/{method} directory.
  * ```--cv-seed``` (default=42): Seed used to make k folds for cross validation.
  * ```--n-folds``` (default=5): Number of folds in cross validation.
  * ```--subsample-frac``` (default=None): Fraction of the data to use for cross validation.
  * ```--retrain``` (default=True): Retrain a model using the whole dataset.

## Dataset
We use 

### Viewing logs
To see the logs from different runs from the `src` directory run

```bash
tensorboard --logdir=logs
```
