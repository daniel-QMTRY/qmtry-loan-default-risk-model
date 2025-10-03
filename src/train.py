import os
import csv
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Dataset
from sklearn.metrics import roc_auc_score

# --- Placeholder Classes (to make the script runnable) ---
# In a real project, these would likely be in a separate `src.model` file.
class LoanDataset(Dataset):
    """Placeholder for the loan dataset."""
    def __init__(self, csv_path):
        print(f"Info: Loading data from '{csv_path}'.")
        num_samples, num_features = 2000, 15
        self.X = torch.randn(num_samples, num_features)
        # Create an imbalanced dataset (~20% positive class)
        self.y = (torch.rand(num_samples) > 0.8).float().view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LoanNet(nn.Module):
    """Placeholder for the neural network model."""
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
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)


# --- Main Training Function ---
def train_model(csv_path, model_path, log_path, epochs, batch_size, lr, patience, scheduler_type="plateau"):
    """
    Trains the LoanNet model with options for different LR schedulers.
    """
    dataset = LoanDataset(csv_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Info: Using device: {device}")

    # --- Data Splitting and Sampling ---
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_indices = train_dataset.indices
    train_labels = dataset.y[train_indices].squeeze().long()
    class_counts = torch.bincount(train_labels)
    class_weights = 1.0 / (class_counts.float() + 1e-6)
    sample_weights = class_weights[train_labels]
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Model, Optimizer, and Scheduler Initialization ---
    model = LoanNet(dataset.X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience//2)
        print("Info: Using ReduceLROnPlateau scheduler.")
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        print("Info: Using CosineAnnealingLR scheduler.")
    else:
        print(f"Error: Unknown scheduler type '{scheduler_type}'. Exiting.", file=sys.stderr)
        sys.exit(1)

    # --- Logging and Early Stopping Setup ---
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "roc_auc", "lr"])

    best_auc = 0
    best_epoch = 0
    patience_counter = 0

    # --- Training Loop ---
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss, y_true, y_pred = 0.0, [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                y_true.extend(y.cpu().numpy())
                y_pred.extend(outputs.cpu().numpy())
        val_loss /= len(val_loader)
        roc_auc = roc_auc_score(y_true, y_pred)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Scheduler Step (conditional on type)
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(roc_auc)
        else:
            scheduler.step()

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | ROC-AUC: {roc_auc:.4f} | LR: {current_lr:.6f}")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, roc_auc, current_lr])

        # Early stopping and model saving
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"  -> Model improved and saved (AUC: {roc_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} (AUC: {best_auc:.4f})")
                break
    
    print("-" * 60)
    print(f"Training log saved to: {log_path}")
    print(f"Best model from epoch {best_epoch} saved to: {model_path}")
    return model

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting training script...")

    # --- Training Configuration ---
    CSV_PATH = "data/loan_data_processed.csv"
    MODEL_PATH = "models/loan_model.pt"
    LOG_PATH = "models/training_log.csv"
    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    PATIENCE = 10
    # Choose scheduler: "plateau" or "cosine"
    SCHEDULER_TYPE = "cosine"

    # --- Run Training ---
    train_model(
        csv_path=CSV_PATH,
        model_path=MODEL_PATH,
        log_path=LOG_PATH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        patience=PATIENCE,
        scheduler_type=SCHEDULER_TYPE
    )

    print("\nTraining run complete.")
    print("Next step: run `python -m src.evaluate` to analyze the results.")
