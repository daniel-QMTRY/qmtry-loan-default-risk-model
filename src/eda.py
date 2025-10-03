import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

def run_eda(csv_path: str, output_dir: str, plot_style: str = 'seaborn-v0_8-whitegrid') -> None:
    """
    Performs Exploratory Data Analysis on the loan dataset.

    This function loads the data, generates and saves a series of plots to
    visualize distributions, class balance, and feature correlations.

    Args:
        csv_path (str): The path to the input CSV data file.
        output_dir (str): The directory where plot images will be saved.
        plot_style (str): The matplotlib style to use for plots.
    """
    # --- 1. File Validation and Setup ---
    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at '{csv_path}'.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    plt.style.use(plot_style)
    print(f"--- Starting Exploratory Data Analysis ---")
    print(f"Info: Loading dataset from '{csv_path}'")
    
    df = pd.read_csv(csv_path)
    print(f"Info: Dataset loaded with shape: {df.shape}")

    # --- 2. Plot Generation ---
    plot_configs: Dict[str, Dict] = {
        "class_balance": {
            "title": "Class Balance (Loan Default vs. Fully Paid)",
            "xlabel": "Loan Status (0: Paid, 1: Default)",
            "ylabel": "Proportion",
        },
        "fico_distribution": {
            "title": "FICO Score Distribution",
            "xlabel": "FICO Score",
            "ylabel": "Frequency",
        },
        "interest_rate_distribution": {
            "title": "Interest Rate Distribution",
            "xlabel": "Interest Rate",
            "ylabel": "Frequency",
        },
        "correlation_heatmap": {
            "title": "Feature Correlation Heatmap",
        },
    }

    # Plot Class Balance
    fig, ax = plt.subplots(figsize=(8, 6))
    df["not.fully.paid"].value_counts(normalize=True).plot(kind="bar", ax=ax, color=["skyblue", "salmon"])
    ax.set_title(plot_configs["class_balance"]["title"])
    ax.set_xlabel(plot_configs["class_balance"]["xlabel"])
    ax.set_ylabel(plot_configs["class_balance"]["ylabel"])
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_plot(output_dir, "class_balance.png")

    # Plot FICO Score Distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df["fico"], bins=30, kde=True, ax=ax, color="darkslateblue")
    ax.set_title(plot_configs["fico_distribution"]["title"])
    ax.set_xlabel(plot_configs["fico_distribution"]["xlabel"])
    ax.set_ylabel(plot_configs["fico_distribution"]["ylabel"])
    plt.tight_layout()
    save_plot(output_dir, "fico_distribution.png")

    # Plot Interest Rate Distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df["int.rate"], bins=30, kde=True, ax=ax, color="teal")
    ax.set_title(plot_configs["interest_rate_distribution"]["title"])
    ax.set_xlabel(plot_configs["interest_rate_distribution"]["xlabel"])
    ax.set_ylabel(plot_configs["interest_rate_distribution"]["ylabel"])
    plt.tight_layout()
    save_plot(output_dir, "interest_rate_distribution.png")

    # Plot Correlation Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    # Only annotate if there are fewer than 20 features to avoid clutter
    should_annotate = len(corr.columns) < 20
    sns.heatmap(corr, annot=should_annotate, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(plot_configs["correlation_heatmap"]["title"])
    plt.tight_layout()
    save_plot(output_dir, "correlation_heatmap.png")

    print("\n--- EDA Finished ---")

def save_plot(output_dir: str, filename: str) -> None:
    """Saves the current matplotlib plot to a file and prints a confirmation."""
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close()
    print(f"Plot saved: {path}")

def create_dummy_csv(file_path: str) -> None:
    """Creates a dummy CSV file for demonstration purposes."""
    if os.path.exists(file_path):
        print(f"Info: Dummy data file '{file_path}' already exists.")
        return
    print(f"Info: Creating dummy data file at '{file_path}'.")
    data = {
        'credit.policy': [1]*5 + [0]*5,
        'purpose': ['debt_consolidation', 'credit_card', 'all_other', 'home_improvement', 'small_business']*2,
        'int.rate': [0.11, 0.13, 0.10, 0.15, 0.12, 0.14, 0.16, 0.11, 0.09, 0.12],
        'installment': [829.10, 228.22, 366.86, 162.34, 102.92, 400.00, 432.10, 120.50, 250.75, 310.45],
        'log.annual.inc': [11.35, 10.5, 11.0, 10.2, 11.5, 12.0, 11.2, 10.8, 11.1, 10.9],
        'dti': [19.9, 14.2, 11.3, 8.1, 15.0, 5.0, 20.1, 10.2, 13.3, 18.0],
        'fico': [737, 707, 682, 712, 667, 727, 692, 722, 687, 702],
        'days.with.cr.line': [5639, 2760, 4710, 2699, 4066, 3119, 4949, 1829, 3180, 2220],
        'revol.bal': [28854, 33623, 3511, 33667, 4740, 0, 7221, 24220, 6127, 21923],
        'revol.util': [52.1, 76.7, 25.6, 73.2, 39.5, 20.6, 68.6, 68.2, 70.1, 67.3],
        'inq.last.6mths': [0, 0, 1, 1, 0, 2, 0, 0, 1, 1],
        'delinq.2yrs': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'pub.rec': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        'not.fully.paid': [0, 0, 0, 0, 1, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    # Define paths for the script
    CSV_FILE_PATH = "data/loan_data.csv"
    OUTPUT_DIRECTORY = "plots/eda"

    # Create a dummy CSV file to make the script runnable
    create_dummy_csv(CSV_FILE_PATH)
    
    # Run the EDA function
    run_eda(csv_path=CSV_FILE_PATH, output_dir=OUTPUT_DIRECTORY)
