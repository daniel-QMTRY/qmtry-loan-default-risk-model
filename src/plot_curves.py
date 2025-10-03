import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

def plot_training_curves(log_path: str = "models/training_log.csv",
                         save_dir: str = "plots/training") -> None:
    """
    Reads a training log CSV and generates professional plots for loss,
    ROC-AUC, and learning rate over epochs.

    Args:
        log_path (str): The path to the training log CSV file.
        save_dir (str): The directory where plot images will be saved.
    """
    # --- 1. File Validation and Data Loading ---
    if not os.path.exists(log_path):
        print(f"Error: Training log not found at '{log_path}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Info: Loading training log from '{log_path}'...")
    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        print(f"Error: Failed to read or parse the CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(save_dir, exist_ok=True)
    print(f"Info: Plots will be saved in '{save_dir}'.")

    # --- 2. Plotting Loss Curve ---
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["epoch"], df["train_loss"], marker="o", linestyle='-', label="Train Loss", color="royalblue")
    ax.plot(df["epoch"], df["val_loss"], marker="s", linestyle='--', label="Validation Loss", color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Binary Cross-Entropy)")
    ax.set_title("Training & Validation Loss Over Epochs")
    ax.legend()
    plt.tight_layout()
    loss_path = os.path.join(save_dir, "loan_default_loss_curve.png")
    plt.savefig(loss_path)
    plt.close(fig)

    # --- 3. Plotting ROC-AUC Curve ---
    roc_path: Optional[str] = None
    best_epoch_data = None
    if "roc_auc" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["epoch"], df["roc_auc"], marker="^", linestyle='-', label="Validation ROC-AUC", color="forestgreen")

        # Highlight the best epoch
        best_epoch_data = df.loc[df["roc_auc"].idxmax()]
        best_epoch = int(best_epoch_data['epoch'])
        best_auc = best_epoch_data['roc_auc']
        
        ax.scatter(best_epoch, best_auc, color="red", s=120, zorder=5,
                   label=f"Best Epoch: {best_epoch} (AUC = {best_auc:.4f})")
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("ROC-AUC Score")
        ax.set_title("Validation ROC-AUC Over Epochs")
        ax.legend()
        plt.tight_layout()
        roc_path = os.path.join(save_dir, "loan_default_auc_curve.png")
        plt.savefig(roc_path)
        plt.close(fig)
    else:
        print("Info: 'roc_auc' column not found in log. Skipping ROC-AUC plot.")

    # --- 4. Plotting Learning Rate Schedule ---
    lr_path: Optional[str] = None
    if "lr" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["epoch"], df["lr"], marker="d", linestyle=':', label="Learning Rate", color="purple")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule Over Epochs")
        ax.legend()
        plt.tight_layout()
        lr_path = os.path.join(save_dir, "loan_default_lr_schedule.png")
        plt.savefig(lr_path)
        plt.close(fig)
    else:
        print("Info: 'lr' column not found in log. Skipping learning rate plot.")

    # --- 5. Final Summary ---
    print("\n--- Plot Generation Summary ---")
    print(f"  - Loss curve saved to: {loss_path}")
    if roc_path:
        print(f"  - ROC-AUC curve saved to: {roc_path}")
    if lr_path:
        print(f"  - Learning rate curve saved to: {lr_path}")
    if best_epoch_data is not None:
        print(f"\nAnalysis: Best model performance was observed at epoch {int(best_epoch_data['epoch'])} "
              f"with a validation ROC-AUC of {best_epoch_data['roc_auc']:.4f}.")

def create_dummy_log_file(log_path: str):
    """Creates a dummy training log CSV to make the script runnable."""
    if os.path.exists(log_path):
        print(f"Info: Dummy log file '{log_path}' already exists.")
        return
    
    print(f"Info: Creating dummy log file at '{log_path}'.")
    header = ['epoch', 'train_loss', 'val_loss', 'roc_auc', 'lr']
    data = [
        [1, 0.65, 0.62, 0.68, 0.001],
        [2, 0.58, 0.55, 0.75, 0.001],
        [3, 0.52, 0.51, 0.79, 0.001],
        [4, 0.48, 0.49, 0.82, 0.001],
        [5, 0.45, 0.48, 0.84, 0.0005],
        [6, 0.43, 0.47, 0.85, 0.0005],
        [7, 0.41, 0.47, 0.85, 0.0005]
    ]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    pd.DataFrame(data, columns=header).to_csv(log_path, index=False)


if __name__ == "__main__":
    LOG_FILE_PATH = "models/training_log.csv"
    SAVE_DIRECTORY = "plots/training"
    
    # Create a dummy log file for demonstration purposes
    create_dummy_log_file(LOG_FILE_PATH)
    
    # Run the main plotting function
    plot_training_curves(
        log_path=LOG_FILE_PATH,
        save_dir=SAVE_DIRECTORY
    )
    print("\n--- Script Finished ---")

