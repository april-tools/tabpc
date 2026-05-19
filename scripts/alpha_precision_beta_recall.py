import sys
import os
import numpy as np
import pandas as pd
import json
import argparse
import fnmatch
from sklearn.preprocessing import OneHotEncoder
from synthcity.metrics import eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader

sys.path.append(".")

GENERATED_DATA_FOLDER = 'artifacts/new_generated_data'
ORIGINAL_DATA_FOLDER = 'data'
TARGET_FOLDER = 'artifacts/new_metrics_results'
NAMES = ['adult', 'default', 'shoppers', 'magic', 'beijing', 'news', 'diabetes']


def find_related_dataset(file_path: str) -> str:
    for name in NAMES:
        if name in file_path:
            return name
    raise ValueError(f"Unknown dataset for file: {file_path}")


def find_files(*, starting_folder: str = ".", pattern: str):
    """
    find all files that match the given pattern, starting from the given folder and going down the directory tree
    """
    matches = []
    for root, _, files in os.walk(starting_folder):
        for filename in files:
            full_name = os.path.join(root, filename)
            if fnmatch.fnmatch(full_name, pattern):
                matches.append(full_name)
    return matches


def result_exists(generated_data_path, generated_data_folder, target_folder, metric):
    result_dir = os.path.join(os.path.dirname(generated_data_path).replace(generated_data_folder, target_folder), metric.name())
    result_path = os.path.join(result_dir, os.path.basename(generated_data_path).replace('.csv', '.json'))
    return os.path.exists(result_path)


def main(real_path, syn_path, info_path):
    with open(info_path, 'r') as f:
        info = json.load(f)

    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)


    ''' Special treatment for default dataset and CoDi model '''

    real_data.columns = range(len(real_data.columns))
    syn_data.columns = range(len(syn_data.columns))

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']
    if info['task_type'] == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    num_real_data = real_data[num_col_idx]
    cat_real_data = real_data[cat_col_idx]

    num_real_data_np = num_real_data.to_numpy()
    cat_real_data_np = cat_real_data.to_numpy().astype('str')


    num_syn_data = syn_data[num_col_idx]
    cat_syn_data = syn_data[cat_col_idx]

    num_syn_data_np = num_syn_data.to_numpy()

    cat_syn_data_np = cat_syn_data.to_numpy().astype('str')

    def normalize_to_int(real, syn):
        out = syn.astype(str)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if out[i,j][-2:] == '.0':
                    out[i,j] = str(int(float(syn[i,j])))
        return out
    
    cat_syn_data_np = normalize_to_int(cat_real_data_np, cat_syn_data_np)
    
    # ensure categorical columns are strings and normalized
    def normalize_cat(arr):
        # arr is 2D numpy array of categorical columns
        out = arr.astype(str)
        # strip whitespace (and lowercase?) to avoid trivial mismatches
        for i in range(out.shape[1]):
            out[:, i] = np.char.strip(out[:, i])
            # out[:, i] = np.char.lower(out[:, i])
        return out
    
    cat_real_data_np = normalize_cat(cat_real_data_np)
    cat_syn_data_np = normalize_cat(cat_syn_data_np)

    # quick diagnostic: show unseen categories in synthetic data
    for col_idx in range(cat_real_data_np.shape[1]):
        real_uni = set(np.unique(cat_real_data_np[:, col_idx]))
        syn_uni = set(np.unique(cat_syn_data_np[:, col_idx]))
        unseen = syn_uni - real_uni
        if unseen:
            print(f"Warning: unseen categories in synthetic column {col_idx}: {sorted(list(unseen))}")
            print(f"Real categories: {sorted(list(real_uni))}")
            print(f"Synthetic categories: {sorted(list(syn_uni))}")

    encoder = OneHotEncoder()
    encoder.fit(cat_real_data_np)

    cat_real_data_oh = encoder.transform(cat_real_data_np).toarray()
    cat_syn_data_oh = encoder.transform(cat_syn_data_np).toarray()

    le_real_data = pd.DataFrame(np.concatenate((num_real_data_np, cat_real_data_oh), axis = 1)).astype(float)


    le_syn_data = pd.DataFrame(np.concatenate((num_syn_data_np, cat_syn_data_oh), axis = 1)).astype(float)

    # Check for nan
    if le_syn_data.isnull().values.any():
        nan_coordinate = np.isnan(le_syn_data.to_numpy()).nonzero()
        nan_row = np.unique(nan_coordinate[0])
        print(f"Synthetic data contains NaN at row {nan_row}: ")
        print(le_syn_data.iloc[nan_row])
        return None, None


    np.set_printoptions(precision=4)

    print('=========== All Features ===========')
    print('Data shape: ', le_syn_data.shape)

    X_syn_loader = GenericDataLoader(le_syn_data)
    X_real_loader = GenericDataLoader(le_real_data)

    quality_evaluator = eval_statistical.AlphaPrecision()
    qual_res = quality_evaluator.evaluate(X_real_loader, X_syn_loader)
    qual_res = {
        k: v for (k, v) in qual_res.items() if "naive" in k
    }  # use the naive implementation of AlphaPrecision

    print('alpha precision: {:.6f}, beta recall: {:.6f}'.format(qual_res['delta_precision_alpha_naive'], qual_res['delta_coverage_beta_naive'] ))

    Alpha_Precision_all = qual_res['delta_precision_alpha_naive']
    Beta_Recall_all = qual_res['delta_coverage_beta_naive']

    return Alpha_Precision_all, Beta_Recall_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate synthetic data quality using Alpha Precision and Beta Recall.')
    parser.add_argument('--pattern', type=str, default="*.csv", help='Pattern to match generated files')
    parser.add_argument('--target_folder', type=str, default=TARGET_FOLDER, help='Target folder for results')
    parser.add_argument('--generated_data_folder', type=str, default=GENERATED_DATA_FOLDER, help='Generated data folder')

    args = parser.parse_args()
    pattern = args.pattern
    target_folder = args.target_folder
    generated_data_folder = args.generated_data_folder

    os.makedirs(target_folder, exist_ok=True)
    generated_files = find_files(starting_folder=generated_data_folder, pattern=pattern)

    print("Pattern: ", pattern)
    print("Target folder: ", target_folder)
    print("Generated data folder: ", generated_data_folder)
    print("Generated files found: ", generated_files)

    for generated_file in generated_files:
        real_path = os.path.join(ORIGINAL_DATA_FOLDER, find_related_dataset(generated_file), 'train.csv')
        syn_path = generated_file
        info_path = os.path.join(ORIGINAL_DATA_FOLDER, find_related_dataset(generated_file), 'info.json')

        print(f"Processing {real_path}, {syn_path}, {info_path}")

        alpha_result_dir = os.path.join(os.path.dirname(generated_file).replace(generated_data_folder, target_folder), "alpha_precision")
        os.makedirs(alpha_result_dir, exist_ok=True)
        alpha_result_path = os.path.join(alpha_result_dir, os.path.basename(generated_file).replace('.csv', '.json'))

        beta_result_dir = os.path.join(os.path.dirname(generated_file).replace(generated_data_folder, target_folder), "beta_recall")
        os.makedirs(beta_result_dir, exist_ok=True)
        beta_result_path = os.path.join(beta_result_dir, os.path.basename(generated_file).replace('.csv', '.json'))

        # Check if these results already exist, and skip if they do (to avoid overwriting)
        if os.path.exists(alpha_result_path) and os.path.exists(beta_result_path):
            print(f"Results already exist for {generated_file}, skipping...")
            continue

        try:
            Alpha_Precision_all, Beta_Recall_all = main(real_path, syn_path, info_path)
        except Exception as e:
            print(f"Error processing {generated_file}: {e}")
            continue

        # Store results in target folder, in a parallel directory structure to the generated data
        # Store alpha precision and beta recall in separate directories
        # First store alpha precision
        
        with open(alpha_result_path, 'w') as f:
            json.dump({
                'alpha_precision': Alpha_Precision_all
            }, f, indent=4)

        # Then store beta recall
        with open(beta_result_path, 'w') as f:
            json.dump({
                'beta_recall': Beta_Recall_all
            }, f, indent=4)
