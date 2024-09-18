import argparse
import os
import json
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Callable

# Import necessary functions from your utils module
from utils import running_mean

# Set up matplotlib
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=14)
sns.set_style("whitegrid")


_to_array = lambda x: np.array(x)
COLUMN_TRANSFORM = {
    "component_potential_grad_norm": _to_array,
    "corrected_total_matrix": _to_array,
    "total_matrix": _to_array,
    "potential_matrix": _to_array,
}
def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots from experiment data.")
    parser.add_argument("--data_dir", required=True, help="Base directory containing the experiment data")
    parser.add_argument("--data_subdir", required=True, help="Subdirectory within data_dir containing the specific experiment data")
    parser.add_argument("--output_dir", required=True, help="Directory to save the generated plots")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing plots")
    parser.add_argument("--dry_run", action="store_true", help="Print filepaths without saving plots")
    return parser.parse_args()

def _read_json_file(filepath: str) -> Dict[str, Any]:
    if filepath.endswith(".json"):
        with open(filepath, "r") as f:
            return json.load(f)
    elif filepath.endswith(".json.gz"):
        with gzip.open(filepath, "rb") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

def extract_directory(dir_path: str) -> tuple:
    filelist = os.listdir(dir_path)
    print(f"Extracting directory: {dir_path}. Found {len(filelist)} files: \n   {filelist}")
    
    config_filename = "config.json" if "config.json" in filelist else "config.json.gz"
    info_filename = "info.json" if "info.json" in filelist else "info.json.gz"
    
    expt_config = _read_json_file(os.path.join(dir_path, config_filename))
    expt_info = _read_json_file(os.path.join(dir_path, info_filename))
    expt_properties = expt_info["expt_properties"]

    df_sgd = parse_sgd_logs(expt_info["sgd_logs"])
    df_gd = parse_gd_logs(expt_info["gd_logs"])
    stagewise_dfs = parse_stagewise_gd_logs(expt_info["stagewise_gd_logs"])
    
    return expt_config, expt_info, expt_properties, df_sgd, df_gd, stagewise_dfs

def parse_sgd_logs(logs):
    print("Parsing SGD logs...")
    eval_names = logs["eval_list"]
    other_cols = None
    data = []
    for checkpoint in logs["checkpoint_logs"]:
        keys = sorted(checkpoint.keys())
        keys.remove("evals")
        if other_cols is None:
            other_cols = keys
        else:
            assert keys == other_cols
        checkpoint_data = [
            checkpoint[key] for key in keys
        ] + checkpoint["evals"]
        data.append(checkpoint_data)
    columns = other_cols + eval_names
    df = pd.DataFrame(data, columns=columns)
    print(f"Data shape: {df.shape}")
    # apply COLUMN_TRANSFORM
    for k, v in COLUMN_TRANSFORM.items():
        if k in df.columns:
            df[k] = df[k].apply(v)
    return df

def parse_gd_logs(logs):
    print("Parsing GD logs...")
    eval_names = logs["eval_list"]
    other_cols = None
    data = []
    for checkpoint in logs["checkpoint_logs"]:
        keys = sorted(checkpoint.keys())
        keys.remove("evals")
        if other_cols is None:
            other_cols = keys
        else:
            assert keys == other_cols
        checkpoint_data = [
            checkpoint[key] for key in keys
        ] + checkpoint["evals"]
        data.append(checkpoint_data)
    columns = other_cols + eval_names
    df = pd.DataFrame(data, columns=columns)
    print(f"Data shape: {df.shape}")
    for k, v in COLUMN_TRANSFORM.items():
        if k in df.columns:
            df[k] = df[k].apply(v)
    return df



def parse_individual_stagewise_gd_logs(logs):
    eval_names = logs["eval_lists"]
    num_stages = logs["num_stages"]
    if num_stages != len(logs["stage_logs"]):
        print("Warning: num_stages does not match the number of stage logs.")
    other_cols = None
    data = []
    accumuated_time = 0
    for stage, stage_logs in enumerate(logs["stage_logs"]):
        for checkpoint in stage_logs:
            keys = sorted(checkpoint.keys())
            keys.remove("evals")
            if other_cols is None:
                other_cols = keys
            else:
                assert keys == other_cols
            checkpoint_data = (
                [checkpoint[key] for key in keys] 
                + checkpoint["evals"] 
                + [stage, num_stages, checkpoint["t"] + accumuated_time]
            )
            data.append(checkpoint_data)
        accumuated_time += stage_logs[-1]["t"]
    columns = other_cols + eval_names + ["stage", "num_stages", "total_time"]
    df = pd.DataFrame(data, columns=columns)
    print(f"Data shape: {df.shape}")
    for k, v in COLUMN_TRANSFORM.items():
        if k in df.columns:
            df[k] = df[k].apply(v)
    return df

def parse_stagewise_gd_logs(logs):
    print("Parsing Stagewise GD logs...")
    results = {}
    for potential_type, potential_logs in logs.items():
        print(f"Parsing GD logs for stagewise potential type: {potential_type}")
        results[potential_type] = parse_individual_stagewise_gd_logs(potential_logs)
    return results


def _generate_filepath(output_dir: str, data_subdir: str, name: str) -> str:
    # Construct the output filepath
    output_filepath = os.path.join(output_dir, data_subdir, name)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    print(f"Generated filepath: {output_filepath}")
    if os.path.exists(output_filepath):
        print(f"Warning: File already exists: {output_filepath}")

    return output_filepath

def create_savefig_fn(output_dir: str, data_subdir: str, overwrite: bool, dry_run: bool) -> Callable:
    def _savefig_fn(fig, name: str):
        filepath = _generate_filepath(output_dir, data_subdir, name)
        if os.path.exists(filepath) and not overwrite:
            print(f"File already exists and overwrite is False. Skipping: {filepath}")
            return
        
        if dry_run:
            print(f"Dry run: Would save figure to {filepath}")
        else:
            print(f"Saving figure: {filepath}")
            fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
    
    return _savefig_fn



def generate_plots(data_dir: str, data_subdir: str, output_dir: str, overwrite: bool, dry_run: bool):
    full_data_path = os.path.join(data_dir, data_subdir)
    expt_config, expt_info, expt_properties, df_sgd, df_gd, stagewise_dfs = extract_directory(full_data_path)
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Create savefig function
    savefig_fn = create_savefig_fn(output_dir, data_subdir, overwrite, dry_run)
    
    # Generate plots
    generate_sgd_plots(df_sgd, expt_config, expt_properties, savefig_fn)
    generate_gd_plots(df_gd, expt_properties, savefig_fn)
    generate_stagewise_plots(stagewise_dfs, savefig_fn)


def generate_sgd_plots(df: pd.DataFrame, config: Dict[str, Any], properties: Dict[str, Any], savefig_fn: Callable):
    do_llc_estimation = config["do_llc_estimation"]
    input_dim = config["model_config"]["input_dim"]
    output_dim = config["model_config"]["output_dim"]
    teacher_matrix = np.array(properties["teacher_matrix"])
    input_correlation_matrix = np.array(properties["input_correlation_matrix"])
    input_output_correlation_matrix = np.array(properties["input_output_cross_correlation_matrix"])
    U, S, V, Vhat, ChangeOfBasis = [
        np.array(properties["svd_matrices"][key]) for key in [
            "U", "S", "V", "Vhat", "ChangeOfBasis"
        ]
    ]

    df["corrected_total_matrix_diagonals"] = df["corrected_total_matrix"].apply(lambda x: np.diag(x))

    # Plot 1: Training Loss and LLC Estimation
    fig, axes = plt.subplots(4, 1, figsize=(10, 18), sharex=True)
    ax = axes[0]
    ax.plot(df["t"], df["train_loss"], label="Training Loss")
    ax.set_ylabel("Training Loss")
    ax.legend(loc="upper left")
    ax.set_yscale("log")

    if do_llc_estimation:
        ax = ax.twinx()
        clipped_llc = np.clip(df["lambdahat"], a_min=0, a_max=1e6)
        ax.plot(df["t"], clipped_llc, "kx", alpha=0.3, label="Estimated LLC, $\hat{\lambda}(w^*)$")
        yvals = running_mean(clipped_llc)
        ax.plot(df["t"], yvals, "g-")
        stage_llcs = np.array(df["stage_potential_llcs"].tolist())
        stage_llcs = np.clip(stage_llcs, a_min=0, a_max=100)
        for i in range(stage_llcs.shape[1]):
            yvals = running_mean(stage_llcs[:, i])
            ax.plot(df["t"], yvals, label=f"Stage {i + 1}", alpha=0.5)
        ax.set_ylabel("Estimated LLC, $\hat{\lambda}(w^*)$")
        ax.legend(loc="upper right")

    # Plot 2: Corrected Total Matrix Diagonals
    ax = axes[1]
    key = "corrected_total_matrix_diagonals"
    for i in range(min(input_dim, output_dim)):
        ax.plot(
            df["t"],
            df[key].apply(lambda x: x[i]), 
            label="$s_{" + str(i + 1) + "}$"
        )

    xmin, xmax = ax.get_xlim()
    singvals_true = S
    ax.hlines(singvals_true, xmin, xmax, color="k", linestyle="--", alpha=0.5)
    ax.set_ylabel("$i^{th}$ Diagonal of $U^T W_T V$")

    # Plot 3: Diagonal h_{i=j}(w_t)^2
    ax = axes[2]
    for i in range(input_output_correlation_matrix.shape[0]):
        yvals = df["potential_matrix"].apply(lambda x: np.array(x)[i, i]**2)
        ax.plot(df["t"], yvals, label=f"({i}, {i})")
    ax.set_ylabel("$h_{i=j}(w_t)^2$")
    ax.set_title("Diagonal $h_{i=j}(w_t)^2$")
    ax.legend()
    ax.set_yscale("log")

    # Plot 4: Off diagonal h_{i≠j}(w_t)^2
    ax = axes[3]
    for i in range(input_output_correlation_matrix.shape[0]):
        for j in range(input_output_correlation_matrix.shape[1]):
            if i == j: continue
            yvals = df["potential_matrix"].apply(lambda x: np.array(x)[i, j]**2)
            ax.plot(df["t"], yvals, label=f"({i}, {j})")
    ax.set_ylabel("$h_{i \\neq j}(w_t)^2$")
    ax.set_title("Off diagonal $h_{i \\neq j}(w_t)^2$")
    ax.set_yscale("log")

    for i, ax in enumerate(axes): 
        if i == len(axes) - 1:
            ax.set_xlabel("Num SGD Steps")

    savefig_fn(fig, "sgd_analysis.pdf")

    # Plot 5: Stage potentials
    cols = [col for col in df.columns if col.startswith("stage_potential")]
    ymin_bound = 1e-4
    for col in cols:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(df["t"], df["full_potential"], label="Full potential")
        values = np.array(df[col].tolist())
        name = col.split('=')[-1]
        for stage in range(values.shape[1]):
            ax.plot(df["t"], values[:, stage], label=f"Stage {stage + 1}")
        ax.set_xlabel("Num SGD Steps")
        ax.set_ylabel("Potential")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title(f"Stagewise potential: {name}")
        savefig_fn(fig, f"sgd_stagewise_potential_{name}.pdf")

def generate_gd_plots(df: pd.DataFrame, properties: Dict[str, Any], savefig_fn: Callable):
    input_output_correlation_matrix = np.array(properties["input_output_cross_correlation_matrix"])

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax = axes[0]
    ax.plot(df["t"], df["loss"], label="Training Loss")
    ax.set_ylabel("Training Loss")
    ax.legend(loc="upper left")
    ax.set_yscale("log")

    ax = axes[1]
    for i in range(input_output_correlation_matrix.shape[0]):
        yvals = df["potential_matrix"].apply(lambda x: np.array(x)[i, i]**2)
        ax.plot(df["t"], yvals, label=f"({i}, {i})")
    ax.set_ylabel("$h_{i=j}(w_t)^2$")
    ax.set_title("Diagonal $h_{i=j}(w_t)^2$")
    ax.legend()
    ax.set_yscale("log")

    ax = axes[2]
    for i in range(input_output_correlation_matrix.shape[0]):
        for j in range(input_output_correlation_matrix.shape[1]):
            if i == j: continue
            yvals = df["potential_matrix"].apply(lambda x: np.array(x)[i, j]**2)
            ax.plot(df["t"], yvals, label=f"({i}, {j})")
    ax.set_ylabel("$h_{i \\neq j}(w_t)^2$")
    ax.set_title("Off diagonal $h_{i \\neq j}(w_t)^2$")
    ax.set_yscale("log")

    for i, ax in enumerate(axes): 
        if i == len(axes) - 1:
            ax.set_xlabel("Num SGD Steps")

    savefig_fn(fig, "gd_analysis.pdf")

def generate_stagewise_plots(stagewise_dfs: Dict[str, pd.DataFrame], savefig_fn: Callable):
    ymin_bound = 1e-4
    for potential_type, df in stagewise_dfs.items():
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        cols = ["total_potential"] + [col for col in df.columns if col.startswith("stage_potential")]
        for col in cols:
            yvals = np.clip(df[col], a_min=ymin_bound, a_max=np.inf)
            ax.plot(df["total_time"], yvals, label=col)

        stage_boundaries = df.groupby("stage")["total_time"].min().values
        ymin, ymax = ax.get_ylim()
        ax.vlines(stage_boundaries, ymin, ymax, color="r", linestyle="dotted", alpha=0.8)

        ax.set_xlabel("Num Steps")
        ax.set_ylabel("Potential")

        ax.set_yscale("log")
        ax.legend()
        ax.set_title(f"`{potential_type}` Potential")

        savefig_fn(fig, f"stagewise_{potential_type}_potential.pdf")



if __name__ == "__main__":
    args = parse_args()
    generate_plots(args.data_dir, args.data_subdir, args.output_dir, args.overwrite, args.dry_run)