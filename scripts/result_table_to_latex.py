import pandas as pd
import os
import sys
import tqdm
import argparse

sys.path.append('.')
from src.util import find_files


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
    'ShallowMixture',
    'TabPC',
    'TabPC_conditional',
]


def fix_method_name(method: str) -> str:
    method_mapping = {
        'ctgan': 'CTGAN',
        'tvae': 'TVAE',
        'goggle': 'GOGGLE',
        'great': 'GReaT',
        'stasy': 'STaSy',
        'codi': 'CoDi',
        'tabddpm': 'TabDDPM',
        'tabsyn': 'TabSyn',
        'tabdiff': 'TabDiff',
        'fullyfactorizedpreprocessed': 'FF',
        'shallowmixture': 'SM',
        'tabpc': 'TabPC',
        'tabpc_conditional': 'TabPC_conditional'
    }
    return method_mapping.get(method.lower(), method)


def new_df_to_latex_table_w_ranks(df: pd.DataFrame, *, scientific: bool = False, times: bool = False) -> str:
    datasets = sorted([d for d in df['dataset'].unique() if d != 'Average Rank'])
    # has_average = 'Average' in df['dataset'].unique()
    has_average_rank = 'Average Rank' in df['dataset'].unique()
    if has_average_rank:
        datasets.append('Average Rank')
        # In this new format, we want to have three columns per dataset: mean, \pm std, and rank
        col_format = '\nl\n' + 'r@{}l@{}r\n' * (len(datasets) - 1) + 'r\n'
    elif scientific:
        col_format = '\nl\n' + 'r@{}l@{}l\n' * len(datasets)
    else:
        col_format = '\nl\n' + 'r@{}l\n' * len(datasets)
    methods = df['method'].unique()
    if has_average_rank:
        header = (
            r"\resizebox{\textwidth}{!}{" +
            f"\\begin{{tabular}}{{{col_format}}}\n"
            "    \\toprule[0.8pt]\n"
            "     \\textbf{Method} & " +
            " & ".join([f"\\multicolumn{{3}}{{c}}{{\\textbf{{{d.capitalize() if d != 'Average Rank' else d}}}}}\n" for d in datasets[:-1]]) +  " & \\textbf{Avg. Rank} \\\\\n"
            "    \\midrule \n"
        )
    elif scientific:
        header = (
            r"\resizebox{\textwidth}{!}{" +
            f"\\begin{{tabular}}{{{col_format}}}\n"
            "    \\toprule[0.8pt]\n"
            "     \\textbf{Method} & " +
            " & ".join([f"\\multicolumn{{3}}{{c}}{{\\textbf{{{d.capitalize()}}}}}\n" for d in datasets]) +  " \\\\\n"
            "    \\midrule \n"
        )
    else:
        header = (
            r"\resizebox{\textwidth}{!}{" +
            f"\\begin{{tabular}}{{{col_format}}}\n"
            "    \\toprule[0.8pt]\n"
            "     \\textbf{Method} & " +
            " & ".join([f"\\multicolumn{{2}}{{c}}{{\\textbf{{{d.capitalize()}}}}}\n" for d in datasets]) +  " \\\\\n"
            "    \\midrule \n"
        )
    
    def format_scientific(value):
        """Format a number in scientific notation for LaTeX"""
        if pd.isna(value):
            return "NaN"
        scientific_str = f"{value:.2e}"
        if 'e' in scientific_str:
            mantissa, exponent = scientific_str.split('e')
            exponent = int(exponent)
            return f"{mantissa} \\times 10^{{{exponent}}}"
        return scientific_str
    
    rows = []
    for method in SORTED_METHODS:
        if method not in methods:
            continue
        row = [fix_method_name(method)]
        for dataset in datasets:
            entry = df[(df['method'] == method) & (df['dataset'] == dataset)]
            if entry.empty:
                cell = "*"
            else:
                mean = entry.iloc[0]['mean']
                std = entry.iloc[0]['std']
                rank = entry.iloc[0]['rank'] if 'rank' in entry.columns else None
                if not has_average_rank:
                    rank = None  # If we don't have average rank, we shouldn't display ranks for any dataset
                if pd.isna(mean):
                    if rank is not None and not pd.isna(rank):
                        rank_fmt = f"({int(rank)})"
                    else:
                        rank_fmt = ""
                    num_cols = 3 if scientific else 2
                    cell = [f"\\multicolumn{{{num_cols}}}{{c}}{{*}}", "~~" + rank_fmt] if rank_fmt else [f"\\multicolumn{{{num_cols}}}{{c}}{{*}}"]
                else:
                    if scientific:
                        mean_fmt = format_scientific(mean)
                    elif mean < 1e-4:
                        mean_fmt = f"<{1e-4}"
                    elif dataset == 'Average Rank':
                        mean_fmt = f"{mean:.2f}"
                    elif times:
                        mean_fmt = f"{mean:.1f}"
                    else:
                        mean_fmt = f"{mean:.4f}"

                    if rank is not None and not pd.isna(rank):
                        # rank_fmt = f"\\textsuperscript{{({int(rank)})}}"
                        rank_fmt = f"({int(rank)})"
                    else:
                        rank_fmt = ""
                    
                    if dataset == 'Average Rank':
                        # Display only the average rank
                        cell = [f"{mean_fmt}"]
                    elif pd.isna(std) or std == 0.0:
                        # cell = f"{mean_fmt}(0){rank_fmt}" if rank_fmt else f"{mean_fmt}(0)"
                        cell = [mean_fmt, "\\footnotesize{$\pm{0.0000}$}", "~~" + rank_fmt] if rank_fmt else [mean_fmt, "\\footnotesize{$\pm{0.0000}$}"]

                        if mean_fmt.startswith("<"):
                            # cell = [mean_fmt, "", "~~" + rank_fmt] if rank_fmt else [mean_fmt, ""]
                            cell = ["\\multicolumn{2}{c}{<0.0001}", "~~" + rank_fmt] if rank_fmt else ["\\multicolumn{2}{c}{<0.0001}"]
                    else:
                        if scientific:
                            std_fmt = format_scientific(std)
                        elif times:
                            std_fmt = f"\\footnotesize{{$\\pm {std:.1f}$}}"
                        else:
                            std_fmt = f"\\footnotesize{{$\\pm {std:.4f}$}}"
                        # cell = f"${mean_fmt}$" + f"{{\\tiny${{\\pm {std_fmt}}}$}}"
                        # cell = f"{mean_fmt}({std_fmt}){rank_fmt}" if rank_fmt else f"{mean_fmt}({std_fmt})"
                        # Now we want these to be of the form [mean, std, rank]
                        cell = [mean_fmt, std_fmt, "~~" + rank_fmt + "\n"] if rank_fmt else [mean_fmt, std_fmt + "\n"]
            # row.append(cell)
            fallback = [cell, "", "\n"] if rank_fmt else [cell, "\n"]
            if scientific and (std == 0.0 or pd.isna(std)):
                if not pd.isna(mean):
                    cell = mean_fmt.split(" ")
                    for i in range(len(cell)):
                        cell[i] = f"${cell[i]}$"
                    cell[2] += "\n"
            elif scientific and std > 0.0:
                # i.e. the training time case
                cell = mean_fmt.split(" ")
                for i in range(len(cell)):
                    cell[i] = f"${cell[i]}$"

            row += cell if isinstance(cell, list) else fallback
        rows.append("    " + " & ".join(row) + " \\\\")
        if method.lower() == "tabdiff":
            rows.append("    \\midrule")
    footer = (
        "    \\bottomrule[1.0pt]\n"
        "\\end{tabular}}"
    )
    return header + "\n".join(rows) + "\n" + footer


def add_average_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add average rank rows (mean of method ranks) for every method as if it were a dataset.
    
    Args:
        df: DataFrame with columns 'method', 'dataset', 'mean', 'std', 'rank'
        
    Returns:
        DataFrame with average rank rows appended
    """
    avg_rank_rows = []
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        if not method_df.empty and 'rank' in method_df.columns:
            mean_rank = method_df['rank'].mean()
            avg_rank_rows.append({
                'method': method, 
                'dataset': 'Average Rank', 
                'mean': mean_rank, 
                'std': 0.0,
            })
    
    if avg_rank_rows:
        avg_rank_df = pd.DataFrame(avg_rank_rows)
        return pd.concat([df, avg_rank_df], axis=0).reset_index(drop=True)
    return df


def write_latex_table(df: pd.DataFrame, *, target_file: str, scientific: bool, times: bool = False):
    latex_table = new_df_to_latex_table_w_ranks(df, scientific=scientific, times=times)

    # Get metric name from target_file
    metric_name = os.path.basename(target_file).replace('.tex', '').replace('_', ' ')
    caption = f"Results for metric: {metric_name}"
    latex_table = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{tab:{metric_name}_updated}}\n"
        f"{latex_table}\n"
        "\\end{table}\n"
    )
    with open(target_file, 'w') as f:
        f.write(latex_table)


def df_to_latex_w_ranks(*, source_table: str, target_file: str, average_rank: bool = True, digits: int = 4, scientific: bool = False, times: bool = False):
    df = pd.read_csv(source_table)
    
    # excluding conditional generation
    df = df[~df['dataset'].astype(str).str.contains('cond', case=False, na=False)]
    
    if average_rank:
        df = add_average_ranks(df)
    df = df.round(digits)

    write_latex_table(df=df, target_file=target_file, scientific=scientific, times=times)


def latex_all_w_ranks(*, csv_folder, pattern="*.csv", digits: int, out_path=None):
    csv_files = find_files(starting_folder=csv_folder, pattern=pattern)
    
    for csv_file in tqdm.tqdm(csv_files):
        if out_path is not None:
            target_file = os.path.join(out_path, os.path.basename(csv_file).replace('csv', 'tex'))
        else:
            target_file = os.path.join(csv_file.replace('csv', 'tex'))
        metric_name = os.path.basename(csv_file).lower()
        average_rank = True
        if 'mle' in metric_name or 'parameters' in target_file.lower() or 'dcr' in metric_name:
            average_rank = False
        df_to_latex_w_ranks(source_table=csv_file,
                    target_file=target_file,
                    digits=digits,
                    average_rank=average_rank,
                    scientific='parameters' in target_file.lower(),
                    times='time' in target_file.lower()
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV result tables to LaTeX format with ranks.")
    parser.add_argument('--csv-folder', type=str, default='artifacts/new_metrics_results_summary_ranked', help='Folder containing CSV files to convert')
    parser.add_argument('--pattern', type=str, default='*.csv', help='Glob pattern to match CSV files in the folder')
    parser.add_argument('--out-path', type=str, default='artifacts/new_metrics_results_latex', help='Output folder for LaTeX files')
    parser.add_argument('--digits', type=int, default=4, help='Number of digits to round to in the LaTeX tables')
    args = parser.parse_args()
    
    latex_all_w_ranks(
        csv_folder=args.csv_folder,
        pattern=args.pattern,
        digits=args.digits,
        out_path=args.out_path,
    )

    # FIXME: Re-add ablation table generation in a more general way that can be integrated into the article pipeline
    # ablation_csv = 'artifacts/ablation/metrics_results_summary/C2ST/xgboost(C2ST).csv'
    # ablation_df = pd.read_csv(ablation_csv)
    # # Drop std column
    # ablation_df = ablation_df.drop(columns=['std'])
    # # Capitalize dataset names
    # ablation_df['dataset'] = ablation_df['dataset'].str.capitalize()
    # # Index by dataset
    # ablation_df = ablation_df.set_index('dataset').reset_index()
    # # Capitalize headers
    # ablation_df.columns = [col.capitalize() for col in ablation_df.columns]
    # # Rename Method column to "Preprocessing"
    # ablation_df = ablation_df.rename(columns={'Method': 'Preprocessing'})
    # # Sort by ablation method (Base, IV, QN, IV_QN)
    # ablation_df['Preprocessing'] = ablation_df['Preprocessing'].map({
    #     'Base': 0,
    #     'IV': 1,
    #     'QN': 2,
    #     'IV_QN': 3
    # })
    # # Sort by Dataset and then by Preprocessing
    # ablation_df = ablation_df.sort_values(by=['Dataset', 'Preprocessing'])
    # # Replace Preprocessing names with LaTeX-friendly versions
    # ablation_df['Preprocessing'] = ablation_df['Preprocessing'].map({
    #     0: 'Base',
    #     1: 'IV',
    #     2: 'QN',
    #     3: 'IV + QN'
    # })
    # ablation_df = ablation_df.rename(columns={'Preprocessing': '\\textbf{Preprocessing}', 'Mean': '\\textbf{C2ST (XGB)}', 'Dataset': '\\textbf{Dataset}'})
    # ablation_df.to_latex(os.path.join('artifacts/ablation/metrics_results_summary/C2ST/', 'xgboost(C2ST).tex'), index=False, label='tab:ablation_c2st_xgboost', caption='Ablation results for C2ST (XGB)', float_format="%.4f")