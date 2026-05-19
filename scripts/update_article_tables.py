import sys
import os
import pandas as pd

sys.path.append('.')
from src.util import find_files
from scripts.result_table_to_latex import latex_all_w_ranks
from scripts.rank_table import main as rank_table

TARGET_TABLES = {
    "sampling_time": None,
    "training_time": None,
    "num_parameters": None,
    "alpha_precision": None,
    "beta_recall": None,
    "shape": "*Shape.csv",
    "trend": "*Trend.csv",
    "lr_detection": "*logistic_regression(C2ST).csv",
    "xgb_detection": "*xgboost(C2ST).csv",
    "mle": "*mle.csv",
    "dcr002": "*dcr_fraction<0.02_test_quantile.csv",
    "dcr005": "*dcr_fraction<0.05_test_quantile.csv",
    "nmi_l1_weighted_complement": "*nmi_l1_weighted_complement.csv",
    "nmi_l1_complement": "*nmi_l1_complement.csv",
    # These are the NMI L1 (weighted) errors instead of similarities
    # "nmi_l1": "*nmi_l1.csv",
    # "nmi_l1_weighted": "*nmi_l1_weighted.csv",
    }


def update_rows(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    # This function takes in the existing dataframe and the new dataframe, and updates the existing dataframe with the new results
    # If there are new methods or datasets in the new dataframe, they will be added to the existing dataframe
    # If there are existing methods or datasets in the existing dataframe that are not in the new dataframe, they will be kept in the existing dataframe
    # If there are existing methods or datasets in the existing dataframe that are also in the new dataframe, their results will be updated with the new results

    for dataset, method in new_df[['dataset', 'method']].values: # iterate through each row of the new dataframe
        new_row = new_df[(new_df['dataset'] == dataset) & (new_df['method'] == method)]
        if len(new_row) != 1:
            print(f"Warning: Expected exactly one row for dataset {dataset} and method {method}, but found {len(new_row)}")
            print("Skipping this row.")
            continue
        new_row = new_row.iloc[0] # convert from dataframe to series

        # Check if this dataset and method already exist in the existing dataframe
        if ((existing_df['dataset'] == dataset) & (existing_df['method'] == method)).any():
            existing_df.loc[(existing_df['dataset'] == dataset) & (existing_df['method'] == method), ['mean', 'std']] = new_row[['mean', 'std']].values
        else: # Otherwise, add this new row to the existing dataframe
            existing_df = pd.concat([existing_df, new_row.to_frame().T], ignore_index=True)
    
    return existing_df


def update_article_tables(
        summary_folder: str,
        result_csvs_folder: str,
        ranked_result_csvs_folder: str,
        tables_folder: str,
        digits: int = 4,
    ) -> None:
    print("Updating raw CSV tables...")
    for table_name, file_pattern in TARGET_TABLES.items():
        # update raw table
        if file_pattern is not None:
            table_path = find_files(starting_folder=summary_folder, pattern=file_pattern)
            if len(table_path) != 1:
                print(f"Warning: Expected exactly one file for pattern {file_pattern}, but found {len(table_path)}")
                print("Skipping this table.")
                continue
            table_path = table_path[0] # get the single file path from the list

            # Update the existing table in the final results folder with the new results from the summary folder
            target_path = os.path.join(result_csvs_folder, f"{table_name}.csv")

            # Read the existing results
            if os.path.exists(target_path):
                existing_df = pd.read_csv(target_path)
            else:
                existing_df = pd.DataFrame()

            # Read the new results
            new_df = pd.read_csv(table_path)

            # Update the existing results with the new results
            updated_df = update_rows(existing_df, new_df)

            # Round the mean and std columns to the specified number of digits
            updated_df = updated_df.round({'mean': digits, 'std': digits})

            # Save the updated results back to the target path
            updated_df.to_csv(target_path, index=False, float_format=f'%.{digits}f')

    # Rerank tables after updating raw CSVs
    rank_table(
        summary_folder=result_csvs_folder,
        output_folder=ranked_result_csvs_folder,
        pattern="*.csv",
        digits=digits,
    )
    
    print("Generating LaTeX tables...")   
    latex_all_w_ranks(
        csv_folder=ranked_result_csvs_folder,
        pattern="*.csv",
        digits=digits,
    )
    
    os.system(f"mv {ranked_result_csvs_folder}/*.tex {tables_folder}/")


if __name__ == "__main__":
    pass
    # TODO: update CLI version of this function using argparse
