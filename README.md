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
Papers: Multimodal single cell data integration challenge: Results and lessons learned https://proceedings.mlr.press/v176/lance22a/lance22a.pdf
Documentation: https://openproblems.bio/events/2021-09_neurips/documentation/data/about_multimodal
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

### 

## Dataset
We use dataset from NeurIPS competition: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122 named GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz.
We assume data is in ``` path_to_repo\repo_name\data```.

### Metrics
**Predicting one modality from another**: As metrics, we consider root mean squared error (RMSE) and Pearson correlation on
log-scaled counts, as well as Spearman correlation.

**Matching cells between modalities**: As metrics, we consider area under the precision recall curve (AUPR) and the average
probability assigned to the correct matching. The latter is a relative measure per dataset that accounts
for non-identifiability among cells with the same identity.

### Viewing logs
To see the logs from different runs from the `src` directory run

```bash
tensorboard --logdir=logs
```
