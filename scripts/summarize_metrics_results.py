import os
import sys
import tqdm
import numpy as np
import pandas as pd

sys.path.append('.')
from src.util import find_files, load_json


TARGET_FOLDER = 'artifacts/metrics_results_summary'
DATASET_NAMES = [
    'adult',
    'beijing',
    'default',
    'diabetes',
    'magic',
    'news',
    'shoppers',    
]
SORTED_METHODS = [
    'CTGAN',
    'TVAE',
    'GOGGLE',
    'GReaT',
    'STaSy',
    'CoDi',
    'TabDDPM',
    'TabSyn',
    'TabDiff',
    'SimpleFactors',
    'FullyFactorizedPreprocessed',
    'FullyFactorizedRandom',
    'ShallowMixture',
    'TabPC',
]
DIGITS = 4


def sort_rows(df: pd.DataFrame, col: str, names: list) -> pd.DataFrame:
    """
    Sort the rows based on the values of a given column.
    If the value do not appear in the list of given names, the values are put at the end.
    The rest of the rows are sorted according to the given list of names.
    """
    # Create a categorical type with the specified order
    cat_type = pd.CategoricalDtype(categories=names, ordered=True)
    df[col] = df[col].astype(cat_type)
    return df.sort_values([col, 'dataset'])


def remove_lists_from_dict(d: dict) -> dict:
    return {k: v for k, v in d.items() if not isinstance(v, list)}


def get_mean_std_of_submetrics(folder: str) -> dict:
    json_files = find_files(starting_folder=folder, pattern="*.json")
    rows = []
    for json_file in json_files:
        rows.append(remove_lists_from_dict(load_json(json_file)))
    df = pd.DataFrame(rows)
    mean = df.mean().to_dict()
    std = df.std().to_dict()
    submetrics = df.columns.tolist()
    return {submetric: {'mean': mean[submetric], 'std': std[submetric]} for submetric in submetrics}


def get_mean_std_of_submetrics_new_protocol(base_dir: str, metric: str, method: str, dataset: str) -> dict:
    json_files = find_files(starting_folder=base_dir, pattern=f"*{method}/{dataset}/*/{metric}/*.json")
    rows = []
    for json_file in json_files:
        rows.append(remove_lists_from_dict(load_json(json_file)))
    df = pd.DataFrame(rows)
    mean = df.mean().round(DIGITS).to_dict()
    std = df.std().round(DIGITS).to_dict()
    submetrics = df.columns.tolist()
    return {submetric: {'mean': mean[submetric], 'std': std[submetric]} for submetric in submetrics}


def is_integer(s: pd.Series) -> bool:
    return bool((s.astype(int) == s).all())


def edit_csv(*, file: str, d: dict) -> None:
    """
    Edit or add a row in a CSV file.
    Identity columns are any columns except 'mean' and 'std'.
    
    Args:
        file: Path to CSV file
        d: Dictionary with column names as keys and values to update/add
    """
    df = pd.read_csv(file)
    
    # ID columns are everything except 'mean' and 'std'
    id_cols = [col for col in d.keys() if col not in ['mean', 'std']]

    # If the CSV already contains duplicate rows for the identity columns,
    # remove them now (keep the last occurrence) to avoid accumulating
    # duplicate entries. Only consider identity columns that actually exist
    # in the dataframe.
    existing_id_cols = [c for c in id_cols if c in df.columns]
    if existing_id_cols:
        before = len(df)
        df = df.drop_duplicates(subset=existing_id_cols, keep='last').reset_index(drop=True)
        after = len(df)
        if before != after:
            print(f"Removed {before-after} duplicate rows in {file} based on {existing_id_cols}")

    # Create mask to check if row exists based on id_cols.
    # Build the mask in an index-aligned way and compare as strings (strip
    # whitespace) to be robust to dtype/whitespace differences.
    if id_cols:
        mask = pd.Series(True, index=df.index)
        for col in id_cols:
            # If the column is not present in the CSV, it can't match any row
            if col not in df.columns:
                mask &= False
                break
            mask &= df[col].astype(str).str.strip() == str(d[col]).strip()
    else:
        # No identity columns provided -> nothing to match
        mask = pd.Series([False] * len(df), index=df.index)
    
    if mask.any():
        # Update existing entry
        for key, value in d.items():
            df.loc[mask, key] = value
    else:
        # Add new entry
        df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)

    if is_integer(df['mean']):
        df['mean'] = df['mean'].astype(int)
        
    df.to_csv(file, index=False, float_format=f'%.{DIGITS}f')


def get_method_name(experiment_folder: str) -> str:
    results = load_json(f"{experiment_folder}/results.json")
    method_name = results['config']['model']['type']
    if method_name == 'ProbabilisticCircuit':
        return 'TabPC'
    if method_name == 'OnlyMarginals':
        q_norm = "QuantileNormalizer" in results['config']['model']['preprocessors']
        nan_handling = results['config']['model']['preprocessors']['NanHandler']['handle_inflated_values']
        if "random_fit" in experiment_folder:
            return "FullyFactorizedRandom"
        elif q_norm and nan_handling:
            return "FullyFactorizedPreprocessed"
        elif q_norm:
            return "FullyFactorizedQuantileNorm"
        elif nan_handling:
            return "FullyFactorizedNanHandling"
        else:
            return "SimpleFactors"
    return method_name


def get_dataset_name(experiment_folder: str) -> str:
    results = load_json(f"{experiment_folder}/results.json")
    dataset_path = results['config']['dataset_path']
    dataset_name = dataset_path.split('/')[-1].split('.')[0]
    return dataset_name


def update_model_statistics(models_folder: str, target_folder: str, model_name: str, pattern: str = "*/results.json", ignore_pattern: str = None):
    print(f"Updating training time, sampling time, and number of parameters for {models_folder}...")
    
    # Set default entry in these dicts to an empty list
    training_times = {dataset: [] for dataset in DATASET_NAMES}
    num_parameters = {dataset: [] for dataset in DATASET_NAMES}
    sampling_times = {dataset: [] for dataset in DATASET_NAMES}

    for results_file in find_files(starting_folder=models_folder, pattern=pattern):
        if ignore_pattern and ignore_pattern in results_file:
            continue
        experiment_folder = os.path.dirname(results_file)
        results = load_json(results_file)
        training_time = results['train_report']['training_time']
        num_params_model = results['train_report']['num_parameters']
        sampling_time = results['train_report']['generation_time']
        # method = get_method_name(experiment_folder)
        dataset = get_dataset_name(experiment_folder)
        training_times[dataset].append(training_time)
        num_parameters[dataset].append(num_params_model)
        sampling_times[dataset].append(sampling_time)
    
    for dataset in DATASET_NAMES:
        # Compute mean and stddev of training time and number of parameters for this dataset
        training_time = np.array(training_times[dataset])
        num_params = np.array(num_parameters[dataset])
        sampling_time = np.array(sampling_times[dataset])
        if len(training_time) > 0:
            training_time_mean = np.mean(training_time)
            training_time_std = np.std(training_time)
        else:
            training_time_mean = 0.0
            training_time_std = 0.0
        if len(num_params) > 0:
            num_params_mean = np.mean(num_params)
            num_params_std = np.std(num_params)
        else:
            num_params_mean = 0.0
            num_params_std = 0.0
        if len(sampling_time) > 0:
            sampling_time_mean = np.mean(sampling_time)
            sampling_time_std = np.std(sampling_time)
        else:
            sampling_time_mean = 0.0
            sampling_time_std = 0.0

        # Update the corresponding CSV files in the target folder (create them if they don't exist)

        # Training time CSV
        if not os.path.exists(f"{target_folder}/training_time/training_time.csv"):
            os.makedirs(f"{target_folder}/training_time", exist_ok=True)
            pd.DataFrame(columns=['method', 'dataset', 'mean', 'std']).to_csv(f"{target_folder}/training_time/training_time.csv", index=False)
        edit_csv(
            file=f"{target_folder}/training_time/training_time.csv",
            d={'method': model_name, 'dataset': dataset, 'mean': training_time_mean, 'std': training_time_std})

        # Number of parameters CSV
        if not os.path.exists(f"{target_folder}/num_parameters/num_parameters.csv"):
            os.makedirs(f"{target_folder}/num_parameters", exist_ok=True)
            pd.DataFrame(columns=['method', 'dataset', 'mean', 'std']).to_csv(f"{target_folder}/num_parameters/num_parameters.csv", index=False)
        edit_csv(
            file=f"{target_folder}/num_parameters/num_parameters.csv",
            d={'method': model_name, 'dataset': dataset, 'mean': num_params_mean, 'std': num_params_std})

        # Sampling time CSV
        if not os.path.exists(f"{target_folder}/sampling_time/sampling_time.csv"):
            os.makedirs(f"{target_folder}/sampling_time", exist_ok=True)
            pd.DataFrame(columns=['method', 'dataset', 'mean', 'std']).to_csv(f"{target_folder}/sampling_time/sampling_time.csv", index=False)
        edit_csv(
            file=f"{target_folder}/sampling_time/sampling_time.csv",
            d={'method': model_name, 'dataset': dataset, 'mean': sampling_time_mean, 'std': sampling_time_std})


def summarize_metrics(*, results_folder, target_folder, models_folder, new_protocol=False, update_model_stats=True):
    """
    Summarize metric results across 5 runs by computing the mean and stddev of each submetric,
    and save the results in a CSV file for each metric and submetric.
    The CSV files have columns: method, dataset, mean, std.
    The method column is sorted according to the SORTED_METHODS list.
    The dataset column is sorted alphabetically.
    The results_folder is expected to have the following structure:
        results_folder/method/dataset/run_i/metric/method_dataset_run_i.json
    where method is the name of the method, dataset is the name of the dataset, i is the run index,
    and metric is the name of the metric.
    The target_folder will have the following structure:
        target_folder/metric/submetric.csv
    where metric is the name of the metric, and submetric is the name of the submetric.

    Parameters:
        results_folder: the folder where the metric results are stored in JSON files
        target_folder: the folder where the summarized CSV files will be saved
        models_folder: the folder where the trained model results are stored,
            used to extract training time, sampling time, and number of parameters for each method/dataset
        new_protocol: whether to use the new protocol for summarizing metrics, where we have multiple
            JSON files for the same method/dataset/metric, and we compute the mean and stddev across these
            files. If False, we assume there is only one run and 20 results for that run.
    """
    metrics_files = find_files(starting_folder=results_folder, pattern="*.json")
    metrics_folders = list(set([os.path.dirname(f) for f in metrics_files]))
    
    # exclude 'old' folders
    metrics_folders = [f for f in metrics_folders if 'old' not in f]
    
    os.makedirs(target_folder, exist_ok=True)
    
    rows = []

    stored_tuples = set() # stored seen tuples to skip duplicates in the new protocol, where we have multiple json files for the same method/dataset/metric)
    # Specifically, we have five repetitions per method, and we list the tuple (method, dataset, metric) only once
    # For this tuple, when it is seen for the first time, we get the mean and stddev across the five repetitions and store it
    # When the same tuple is seen again, we then skip it
    
    k = len(results_folder.split('/'))
    for metric_folder in tqdm.tqdm(metrics_folders):
        subfolders = metric_folder.split('/')[k:]
        method = subfolders[0]
        metric_name = subfolders[-1]
        dataset = subfolders[1] if len(subfolders) == 3 else "_".join(subfolders[1:-1]).replace('_samples','')

        if new_protocol:
            if "run" in dataset:
                dataset = dataset.split("_")[0]
            if (method, dataset, metric_name) in stored_tuples:
                continue # Skip if we've already seen this tuple of method/dataset/metric
            stored_tuples.add((method, dataset, metric_name))
            submetrics = get_mean_std_of_submetrics_new_protocol(results_folder, metric_name, method, dataset)
        else:
            submetrics = get_mean_std_of_submetrics(metric_folder)

        rows.append({
            'dataset': dataset,
            'method': method,
            'metric': metric_name,
            'submetrics': submetrics
        })

    df = pd.DataFrame(rows)
        
    # group by metric, and the save by splitting each submetric
    for metric_name, group in df.groupby('metric'):
        summary_folder = os.path.join(target_folder, metric_name)
        os.makedirs(summary_folder, exist_ok=True)
        subm_df = pd.DataFrame(group['submetrics'].to_list())
        for submetric_name in subm_df.keys():
            df_mean_std = pd.DataFrame(subm_df[submetric_name].to_list())
            df_out = pd.concat([group[['method','dataset']].reset_index(drop=True), df_mean_std.reset_index(drop=True)], axis=1)
            df_out.sort_values(by=['method', 'dataset'], inplace=True)
            df_out = sort_rows(df_out, 'method', SORTED_METHODS)

            df_out.to_csv(
                os.path.join(summary_folder, f"{submetric_name.replace('/','-').replace(' ','')}.csv"),
                index=False,
                float_format=f'%.{DIGITS}f'
            )

    if update_model_stats:
        # Note: the following assumes all trained models are stored in the same folder (models_folder)
        update_model_statistics(models_folder=models_folder, target_folder=target_folder, model_name='TabPC', pattern="*pc*/results.json")
        update_model_statistics(models_folder=models_folder, target_folder=target_folder, model_name='ShallowMixture', pattern="*shallow_mixture*/results.json")
        update_model_statistics(models_folder=models_folder, target_folder=target_folder, model_name='FullyFactorizedPreprocessed', pattern="*FullyFactorized_with_preprocessing*/results.json", ignore_pattern="*FullyFactorized_with_preprocessing_random*")
        update_model_statistics(models_folder=models_folder, target_folder=target_folder, model_name='FullyFactorizedRandom', pattern="*FullyFactorized_with_preprocessing_random*/results.json")
        # update_model_statistics(models_folder=models_folder, target_folder=target_folder, model_name='SimpleFactors', pattern="*FullyFactorized_no_preprocessing*/results.json")

        # Also sort the training, sampling, and num_parameters CSVs by method and dataset
        for stat in ['training_time', 'sampling_time', 'num_parameters']:
            stat_file = os.path.join(target_folder, stat, f"{stat}.csv")
            df_stat = pd.read_csv(stat_file)
            df_stat.sort_values(by=['method', 'dataset'], inplace=True)
            df_stat.to_csv(stat_file, index=False, float_format=f'%.{DIGITS}f')


def summarize_ablation_metrics(*, results_folder, target_folder):
    metrics_files = find_files(starting_folder=results_folder, pattern="*.json")
    metrics_folders = list(set([os.path.dirname(f) for f in metrics_files]))
    
    # exclude 'old' folders
    metrics_folders = [f for f in metrics_folders if 'old' not in f]
    
    os.makedirs(target_folder, exist_ok=True)
    
    rows = []

    stored_pairs = set() # to skip duplicates in the new protocol, where we have multiple json files for the same method/dataset/metric)
    
    k = len(results_folder.split('/'))
    for metric_folder in tqdm.tqdm(metrics_folders):
        subfolders = metric_folder.split('/')[k:]
        method = subfolders[2]
        metric_name = subfolders[-1]
        dataset = subfolders[0]

        # We only have one file per ablation method/dataset/metric, so we can directly compute the submetrics without worrying about duplicates

        rows.append({
            'dataset': dataset,
            'method': method,
            'metric': metric_name,
            'submetrics': get_mean_std_of_submetrics(metric_folder)
        })

        # if new_protocol:
        #     if "run" in dataset:
        #         dataset = dataset.split("_")[0]
        #     if (method, dataset, metric_name) in stored_pairs:
        #         continue
        #     stored_pairs.add((method, dataset, metric_name))
        #     submetrics = get_mean_std_of_submetrics_new_protocol(results_folder, metric_name, method, dataset)
        # else:
        #     submetrics = get_mean_std_of_submetrics(metric_folder)

        # rows.append({
        #     'dataset': dataset,
        #     'method': method,
        #     'metric': metric_name,
        #     'submetrics': submetrics
        # })

    df = pd.DataFrame(rows)
        
    # group by metric, and the save by splitting each submetric
    for metric_name, group in df.groupby('metric'):
        summary_folder = os.path.join(target_folder, metric_name)
        os.makedirs(summary_folder, exist_ok=True)
        subm_df = pd.DataFrame(group['submetrics'].to_list())
        for submetric_name in subm_df.keys():
            df_mean_std = pd.DataFrame(subm_df[submetric_name].to_list())
            df_out = pd.concat([group[['method','dataset']].reset_index(drop=True), df_mean_std.reset_index(drop=True)], axis=1)
            df_out.sort_values(by=['method', 'dataset'], inplace=True)

            df_out.to_csv(
                os.path.join(summary_folder, f"{submetric_name.replace('/','-').replace(' ','')}.csv"),
                index=False,
                float_format=f'%.{DIGITS}f'
            )


if __name__ == "__main__":
    # summarize_metrics(
    #     results_folder='artifacts/metrics_results',
    #     target_folder='artifacts/metrics_results_summary',
    # )

    summarize_metrics(
        results_folder='artifacts/new_metrics_results',
        target_folder='artifacts/new_metrics_results_summary',
        new_protocol=True,
    )

    # summarize_ablation_metrics(
    #     results_folder='artifacts/ablation/metric_results',
    #     target_folder='artifacts/ablation/metrics_results_summary',
    # )