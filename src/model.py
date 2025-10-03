import os
import csv
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Dataset
from sklearn.metrics import roc_auc_score

# --- Data and Model Definitions ---

class LoanDataset(Dataset):
    """
    A placeholder Dataset class. In a real project, this would handle loading
    and preprocessing of loan data from a CSV file.
    """
    def __init__(self, csv_path):
        """Initializes the dataset."""

        print(f"Info: Initializing dataset (using placeholder data). Path: {csv_path}")
        num_samples, num_features = 2000, 15
        self.X = torch.randn(num_samples, num_features)

        self.y = (torch.rand(num_samples) > 0.85).float().view(-1, 1)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.y)

    def __getitem__(self, idx):
        """Retrieves a sample from the dataset at the given index."""
        return self.X[idx], self.y[idx]

class LoanNet(nn.Module):
    """
    Final neural network architecture for loan default prediction.
    Includes BatchNorm + Dropout for regularization and stability.
    """
    def __init__(self, input_features, dropout_rate=0.4):
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

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# --- Training Function ---
def train_model(config):
    """
    Executes the model training and validation loop based on the provided configuration.
    
    Args:
        config (dict): A dictionary containing all training hyperparameters and paths.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Info: Using compute device: {device}")

    dataset = LoanDataset(config["csv_path"])

    # --- Data Splitting and Loading ---
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # --- Weighted Sampler for Class Imbalance ---
    train_indices = train_dataset.indices
    train_labels = dataset.y[train_indices].squeeze().long()
    class_counts = torch.bincount(train_labels)
    class_weights = 1.0 / (class_counts.float() + 1e-6) # Epsilon for stability
    sample_weights = class_weights[train_labels]
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # --- Model, Optimizer, and Scheduler Setup ---
    model = LoanNet(dataset.X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.BCELoss()

    scheduler_type = config.get("scheduler_type", "plateau")
    if scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=config["patience"]//2)
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["learning_rate"]/100)
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    else:
        print(f"Error: Invalid scheduler type '{scheduler_type}'. Aborting.", file=sys.stderr)
        return None
    print(f"Info: Using learning rate scheduler: {scheduler_type}")

    # --- Logging and Early Stopping Initialization ---
    log_file = config["log_path"]
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "roc_auc", "lr"])

    best_auc = 0.0
    patience_counter = 0

    # --- Training and Validation Loop ---
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss, y_true, y_pred_probs = 0.0, [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                y_true.extend(y_batch.cpu().numpy())
                y_pred_probs.extend(outputs.cpu().numpy())
        val_loss /= len(val_loader)
        
        roc_auc = roc_auc_score(y_true, y_pred_probs)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:02d}/{config['epochs']} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val AUC: {roc_auc:.4f} | LR: {current_lr:.6f}")

        with open(log_file, "a", newline="") as f:
            writer.writerow([epoch, train_loss, val_loss, roc_auc, current_lr])

        # --- Scheduler and Early Stopping Logic ---
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(roc_auc)
        else:
            scheduler.step()

        if roc_auc > best_auc:
            best_auc = roc_auc
            patience_counter = 0
            torch.save(model.state_dict(), config["model_path"])
            print(f"  -> Validation AUC improved. Model saved to {config['model_path']}")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"  -> Early stopping triggered after {config['patience']} epochs with no improvement.")
                break
    
    print("\nTraining finished.")
    return model

# --- Main Execution Block ---

if __name__ == '__main__':
    # --- Centralized Training Configuration ---
    TRAINING_CONFIG = {
        "csv_path": "data/loan_data_processed.csv",
        "model_path": "models/loan_model.pt",
        "log_path": "models/training_log.csv",
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "patience": 10,
        "scheduler_type": "cosine"  # Options: "plateau", "cosine", "step"
    }

    print("--- Starting Model Training ---")
    train_model(TRAINING_CONFIG)
    print("--- Training Script Finished ---")
