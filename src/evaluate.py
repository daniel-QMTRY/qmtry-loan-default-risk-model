import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)

# --- Placeholder Data and Model Definitions ---
# In a real project, these would be in a separate 'model.py' and imported.
# They are included here to make this script self-contained and runnable.

class LoanDataset(torch.utils.data.Dataset):
    """
    A placeholder Dataset that simulates loading loan application data.
    """
    def __init__(self, csv_path: str):
        """Initializes the dataset."""
        print(f"Info: Initializing dataset (using placeholder data for '{csv_path}').")
        num_samples, num_features = 2000, 15
        self.X: torch.Tensor = torch.randn(num_samples, num_features)
        # Simulate an imbalanced dataset (~15% positive class)
        self.y: torch.Tensor = (torch.rand(num_samples) > 0.85).float().view(-1, 1)

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves a single sample from the dataset."""
        return self.X[idx], self.y[idx]

class LoanNet(nn.Module):
    """
    A placeholder neural network. Its architecture must match the one used for
    training the model being loaded.
    """
    def __init__(self, input_features: int, dropout_rate: float = 0.4):
        super(LoanNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model."""
        return self.network(x)


# --- Core Evaluation Function ---

def evaluate_model(csv_path: str, model_path: str, plots_dir: str) -> None:
    """
    Loads a trained model to evaluate its performance on the full dataset.

    This function calculates and prints key classification metrics and generates
    and saves plots for the ROC and Precision-Recall curves.
    """
    print("--- Starting Model Evaluation ---")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Info: Using compute device: {device}")
    os.makedirs(plots_dir, exist_ok=True)

    # --- Data and Model Loading ---
    dataset = LoanDataset(csv_path)
    model = LoanNet(dataset.X.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Prediction on Full Dataset ---
    X_all, y_true_tensor = dataset.X.to(device), dataset.y
    with torch.no_grad():
        y_pred_probs = model(X_all).cpu().numpy()

    y_true = y_true_tensor.cpu().numpy()
    y_pred_labels = (y_pred_probs > 0.5).astype(int)

    # --- Metrics Calculation and Display ---
    auc_score = roc_auc_score(y_true, y_pred_probs)
    print(f"\nOverall ROC-AUC Score: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_labels, target_names=["Not Default", "Default"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred_labels))

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')

    # ROC Curve
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(plots_dir, "loan_default_roc_curve.png")
    plt.savefig(roc_path)
    print(f"\nPlot saved: ROC curve at '{roc_path}'")
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='royalblue', lw=2, label='Precision-Recall Curve')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    plt.tight_layout()
    pr_path = os.path.join(plots_dir, "loan_default_pr_curve.png")
    plt.savefig(pr_path)
    print(f"Plot saved: Precision-Recall curve at '{pr_path}'")
    plt.show()

# --- Utility Functions for Standalone Execution ---

def create_dummy_files(csv_path: str, model_path: str) -> None:
    """Creates dummy data and model files to ensure the script is runnable."""
    # Create dummy CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        print(f"Info: Creating dummy data file at '{csv_path}'.")
        pd.DataFrame({'feature': [0]}).to_csv(csv_path, index=False)

    # Create dummy model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        print(f"Info: Creating dummy model file at '{model_path}'.")
        dummy_model = LoanNet(input_features=15)
        torch.save(dummy_model.state_dict(), model_path)


# --- Main Execution Block ---

if __name__ == "__main__":
    # Define file paths
    CSV_PATH = "data/loan_data_processed.csv"
    MODEL_PATH = "models/loan_model.pt"
    PLOTS_DIR = "plots/evaluation"

    # Create dummy files for demonstration purposes
    create_dummy_files(csv_path=CSV_PATH, model_path=MODEL_PATH)

    # Run the main evaluation function
    evaluate_model(
        csv_path=CSV_PATH,
        model_path=MODEL_PATH,
        plots_dir=PLOTS_DIR
    )
    print("\n--- Evaluation Script Finished ---")

