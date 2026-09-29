# TabPC: A Sobering Look at Tabular Data Generation via Probabilistic Circuits

This is the repository for the article ["A Sobering Look at Tabular Data Generation via Probabilistic Circuits"](https://arxiv.org/abs/2603.23016), where we revisit advances in deep generative models for tabular data generation through the lens of hierarchical mixture models; in particular, probabilistic circuits.

## Setup

### Installation

We suggest creating a Python virtual environment with Python 3.10. After activating the environment, run:
```bash
make install
```
to install all the requirements.

### Initialising the `cirkit` submodule

To use the intended version of `cirkit`, the submodule in the repo must first be initialised and brought up to date. This can be done by running the following commands:
```bash
git submodule init
```
This will initialise your local configuration file. Follow this by running:
```bash
git submodule update
```
This fetches the data from the `cirkit` project and checks out the correct commit for the repo.

### Downloading datasets

All datasets are publicly available.

To download the datasets used in our article, run the following command:
```bash
python datasets_scripts/tabdiff_data_download.py
```

Followed by
```bash
python datasets_scripts/tabdiff_process_dataset.py
```
for processing the datasets.

These scripts are taken from the [TabDiff repository](https://github.com/MinkaiXu/TabDiff).

### W&B placeholder

Due to some legacy code, you may also need to create a placeholder WANDB file under `src/private_constants.py`. If you are not interested in W&B, you can use the following template:
```
ENTITY="placeholder"
PROJECT="TabPC"
```

## Replicating article experiments

### Main experiment

To replicate the main experiment (i.e. training our PCs with tuned hyperparameters on each of the datasets and computing the desired metrics on the generated data), run:
```bash
python scripts/experiment_main.py --train-models all --sample --evaluate --update-tables
```
This script has multiple command-line arguments to enable or disable parts of the experiment pipeline. These are:
- `--train-models` or `-t`, followed by a selection of the models to train. Valid choices are:
    - "all", "tab_pc", "shallow_mixture", "fully_factorized", "none"
- `--sample` or `-s`. This runs sampling for each of the trained models in the `ARTICLE_MODELS_FOLDER` directory.
- `--evaluate` or `-e`. This runs metrics computation for the generated data of each of our model types (TabPC, ShallowMixture, FullyFactorizedPreprocessed, FullyFactorizedRandom). Generated data should be stored in `artifacts/generated_data/{method}`.
- `--evaluate-baselines` or `-b`. This runs metrics computation for the generated data of each baseline model. Generated data should be stored in `artifacts/generated_data/{method}`. This data must be generated externally from this repository, e.g. by using [the TabSyn repo](https://github.com/amazon-science/tabsyn/). For full comparability, one must ensure that they have generated 1 dataset from 5 different trained models for each of the baseline (which unfortunately that repository cannot do out of the box).
- `--update-tables`. This updates the LaTeX tables stored in `article_material/article_tables`. It also computes the ranked versions of the result CSV files, stored by default in `artifacts/ranked_results`.

Results for this experiment are stored by default in `article_material/our_results` (as CSVs) and `article_material/article_tables` (as the corresponding LaTeX tables, formatted as in the paper).

#### Generating data with TabPFN

We include a script for generating data with TabPFN, `scripts/experiment_tabpfn.py`, as this is not present in the TabSyn repository unlike the other baselines. Data generated in this way should be placed in `artifacts/generated_data/TabPFN` for integration with the article pipeline.

A new environment must also be created to be able to run TabPFN. To do this, create a new environment with Python 3.11.

Then, install the version of `tabpfn-extensions` corresponding to what we use in the paper via
```
pip install "tabpfn-extensions[all] @ git+https://github.com/PriorLabs/tabpfn-extensions.git@7f84343b59ffb03fdecd4fb6a8576d0eea09db57"
```
Please also ensure that you install version 8.0.8 of the `tabpfn` package to ensure reproducibility, and that the version for the TabPFN checkpoints is TabPFN-3. Also ensure that the datasets are downloaded and processed [as above](#downloading-datasets).

The first time you run the script, it will likely fail due to requiring the TabPFN model weights to be downloaded. You should also be asked to generate an API key to use TabPFN; follow the instructions from the pop-up to do this and enter it there. Note: this requires your compute source to have internet access. After doing this, kill the script and rerun to ensure that the correct number of datasets are generated.

When running the script, specify your chosen dataset by `--dataset` and desired location for the generated data by `--output-data-dir` (recall you should set `artifacts/generated_data/TabPFN` if you want integration with the pipeline).

- Other CLI options also available:
	- `--num-repeats` controls how many datasets are generated. Default: 5.
	- `--save-raw` saves raw generated data before inverting encoding (only recommended if you encounter problems with the inverting the encoder and want to store some intermediate result).
	- `--overwrite` enables the overwriting of existing generated data.
	- `--n-permutations` changes the number of feature permutations over which TabPFN averages to generate its data. We use their default value of 3. Increasing this improves sample robustness at the cost of a linear increase in sampling time.
	- `--use-only-ordinal` enables the use of just the ordinal encoder from sklearn. By default we use (the required parts of) the TabPC preprocessing pipeline for comparability with other methods.
	- `--max-failures` controls how many times generation is attempted before skipping to the next repeat. Default: 3.

### BPD vs C2ST experiment

To replicate the BPD vs C2ST experiment discussed in [Section 4.2](https://arxiv.org/pdf/2603.23016#subsection.4.2), run:
```bash
python scripts/experiment_likelihoods.py --train --compute-likelihoods --sample --evaluate --scrape
```
This script similarly has multiple command-line arguments to enable or disable parts of the experiment pipeline. These are:
- `--train` or `-t`. This enables the training of PCs on the specified hyperparameter grid (see below for more details about this grid and how to change it).
- `--compute-likelihoods` or `-l`. This enables the computation of dataset split likelihoods under the trained PCs in the folder containing the models. These likelihood values are stored in JSON files under the model directory.
- `--sample` or `-s`. This enables the sampling from the trained PCs in the folder containing the models. These samples are stored under the model directory.
- `--evaluate` or `-e`. This enables the evaluation of generated samples. By default, only C2ST (XGB) is evaluated. Further metrics can be computed by uncommenting lines in the `METRICS` dictionary.
- `--scrape` or `-r`. This enables the scraping of results into consolidated CSV files, one for the C2ST and one for the likelihood values.
- `--models-folder`. This allows for the specification of a custom directory in which to store models. Default: `artifacts/ll_models`.
- `--metrics-folder`. This allows for the specification of a custom directory in which to store computed metrics. Default: `artifacts/ll_metrics_results`.
- `--results-folder`. This allows for the specification of a custom directory in which to store consolidated CSV result files. Default: `article_material/ll_results`.
- `--plots-folder`. This allows for the specification of a custom directory in which to store optionally generated plots. Default: `article_material/ll_results/plots`.
- `--seed`. This allows for the specification of a custom seed used for the training of each of the PCs. Default: `0` (so the training should be deterministic; for random training, set this to `None`).
- `--dataset`. This allows for the specification of which dataset to run the PC training on. Default: "all" (Other options are the named datasets used in the paper, e.g. "adult", "beijing", ...)
- `--do-plots`. This enables the generation of plots similar to those found in the paper. Requires scraped CSVs stored in `--results-folder`.

#### Customising this experiment

In this file, there is also a grid of specified hyperparameters to train models for. To modify this grid, change the entries in the lists `LEARNING_RATES`, `BATCH_SIZES`, and `NUM_UNITS`. 

By default, these are set to:
```python
LEARNING_RATES = [0.1, 0.25, 0.5]
BATCH_SIZES = [64, 256, 512]
NUM_UNITS = [128, 512, 2048]
```

For the News dataset, there is a manual check to decrease the number of units from `2048` to `1024`. This is due to memory constraints on our used GPUs.

### Conditional sampling experiment

To replicate the conditional sampling experiment also discussed in [Section 4.2](https://arxiv.org/pdf/2603.23016#subsection.4.2), run:
```bash
python scripts/experiment_conditional.py --train --sample --evaluate --impute --scrape --overwrite
```
Again, this script has several command-line arguments for customising the experiment pipeline.
- `--train` or `-t`. This enables the training of PCs.
- `--sample` or `-s`. This enables the conditional sampling for each trained model at each of the specified conditioning percentages (see below for more details). Additionally, this 
- `--evaluate` or `-e`. This enables the evaluation of generated samples using the specified metrics of Shape (+Trend), wNMIS, and C2ST (XGB) (see below for details on how to modify this).
- `--impute` or `-i`. This enables the {mean, mode} imputation of samples as a simple baseline. This requires the script to be / have been run with `--sample` as it requires the masks saved at this step.
- `--scrape` or `-r`. This enables the scraping of metrics results into consolidated CSV files. These are saved to `--results-folder`.
- `--overwrite` or `-o`. This enables the overwriting of existing samples (otherwise generation is skipped if a sample is already stored).
- `--do-plots`. This enables the plotting of metric results in a format similar to that in the paper.
- `--uncond-batch-size`. This allows for specification of a custom batch size for unconditional sampling (i.e. with 0% conditioning). Default: "None", which enables automatic selection based on GPU memory. Also accepts integer batch sizes.
- `--cond-batch-size`. This allows for specification of a custom batch size for conditional sampling (i.e. with non-zero conditioning percentage). Default: 10 (arbitrary choice found to work on our GPU).
- `--use-train-set-for-conditioning`. This enables the use of the training set as the conditioning set (which is much larger than the test set but less principled, since the model could just memorise this). If not set, the test set is used for conditioning.
- `--models-folder`. This allows for the specification of a custom directory in which to store models. Default: `artifacts/cond_sampling_models`.
- `--metrics-folder`. This allows for the specification of a custom directory in which to store computed metrics. Default: `artifacts/cond_sampling_metrics_results`.
- `--results-folder`. This allows for the specification of a custom directory in which to store consolidated CSV result files. Default: `article_material/cond_sampling_results`.
- `--plots-folder`. This allows for the specification of a custom directory in which to store optionally generated plots. Default: `article_material/cond_sampling_results/plots`.
- `--seed`. This allows for the specification of a custom seed used for the training of each of the PCs. Default: `0` (so the training should be deterministic; for random training, set this to `None`).
- `--dataset`. This allows for the specification of which dataset to run the PC training on. Default: "all" (Other options are the named datasets used in the paper, e.g. "adult", "beijing", ...)

#### Customising this experiment

The metrics computed for the datasets are stored in the `METRICS` dictionary in the script. Additional metrics can be added to this dictionary if desired; see `scripts/experiment_main.py` for examples.

The conditioning percentages are set in the list `COND_PERCENTAGES`. This runs from 0% to 100% in increments of 10%. Modify if other conditioning amounts are desired.

## Training specific models
To run a specific experiment (i.e. combination of model and dataset), run one of the Python files containing experiment configurations in the `train_scripts` folder. Typically, each script has a `--path` option allowing for the specification of a given dataset by its path. One can also set the folder in which to store the trained model via the `--target-folder` option (each file otherwise has its own default path).

### Basic options
- To train an FF model, use `train_scripts/fully_factorized.py`.
- To train an SM model, use `train_scripts/shallow_mixture.py`.
- To train a PC with our tuned hyperparameters, use `train_scripts/pc_sota.py`.

### Customisable options
- To train a PC with desired hyperparameters, use `train_scripts/pc_lls.py` and the command-line arguments `--num-units`, `--batch-size`, `--lr`.
- To train a PC with desired hyperparameters and pre-processing components (e.g. as we do in [Section D.2](https://arxiv.org/pdf/2603.23016#subsection.D.2)), use `train_scripts/pc_ablation.py` and the command-line arguments `--num-units`, `--batch-size`, `--lr` for the hyperparameters, and `--dequantize-all-floats`, `--handle-inflated-values`, `--quantile-normalizer` for the pre-processing.

### Other files

As a simple baseline, we also include `train_scripts/copy_data.py` which simply copies data (as a sanity check to be able to achieve 'perfect' performance). There is also a test script `train_scripts/test.py`.

## Using your own dataset

We also add functionality for using your own datasets. This uses a preprocessing script adapted from TabDiff.

To do this, place your data in a chosen directory, in its own named directory, e.g. `CUSTOM_DATA/custom_dataset/custom_dataset.csv`. Then you can run
```bash
python dataset_scripts/process_dataset.py --dataname custom_dataset --data-root CUSTOM_DATA
```

**Important considerations**:
- The processing script requires the CSV format.
- Values within a row should be comma-separated.
- The first row should contain the header names for each column.

Additionally, you must create a json file `custom_dataset.json` containing the metadata for your dataset:
```json
{
    "name": "<custom_dataset>",
    "task_type": "[NAME_OF_TASK]", # binclass or regression
    "header": "infer",
    "column_names": null,
    "num_col_idx": [LIST],  # list of indices of numerical columns
    "cat_col_idx": [LIST],  # list of indices of categorical columns
    "target_col_idx": [LIST], # list of indices of the target columns (for MLE)
    "file_type": "csv",
    "data_path": "<CUSTOM_DATA>/<custom_dataset>/<custom_dataset>.csv",
    "test_path": null,
    "val_path": null
}
```

To train a model on a processed dataset, you can use `train_scripts/pc_lls.py` by specifying `--path` to the processed training set, alongside chosen hyperparameters [as described above](#customisable-options). Once you have generated data from your model, to compute metrics you can use `scripts/array_evaluate.py`, e.g. by:
```bash
python scripts/array_evaluate.py --target-folder [DESIRED-FOLDER] --original-data-path <CUSTOM_DATA>/<custom_dataset>/<custom_dataset>.csv --generated-data-folder [DESIRED-FOLDER]
```
Certain metrics may not work, such as MLE. This can be patched in the future if it is useful; please open an issue if so. This functionality has not been rigorously tested, so please also open an issue if you run into problems.

## Features to be added

We have several additional features which require some cleaning and reorganisation before they are added to this repository. These include:
- A script for running and collecting the results of the ablation experiment, as we do in [Section D.2](https://arxiv.org/pdf/2603.23016#subsection.D.2).
- A script for replicating the FF (trained) vs FF (random) experiment, as we discuss in [Section 2.2](https://arxiv.org/pdf/2603.23016#subsection.2.2).
- A script for generating critical difference diagrams (CDDs), as we report in [Section E.2](https://arxiv.org/pdf/2603.23016#subsection.E.2).

Finally, we also plan to upload our final trained model checkpoints and all generated data. This repo will be updated with the link once this is complete.

## Reference

If you use this repository, please don't forget to cite the corresponding paper:
```bibtex
@inproceedings{
    scassola2026sobering,
    title={A Sobering Look at Tabular Data Generation via Probabilistic Circuits},
    author={Davide Scassola and Dylan Ponsford and Adri{\'a}n Javaloy and Sebastiano Saccani and Luca Bortolussi and Henry Gouk and Antonio Vergari},
    booktitle={Forty-Second Annual Conference on Uncertainty in Artificial Intelligence},
    year={2026},
    url={https://openreview.net/forum?id=QndFLaQBSj}
}
```
