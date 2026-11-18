# Bonus
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch
import copy
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from task2_feature_extraction import train_test

# here we have 6 outputs (sigmas) not 1 (promoter vs non promoter)
# We will use CrossEntropyLoss, which takes the six raw logits and calculates the multiclass loss
class SigmaCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=4,
            out_channels=32,
            kernel_size=8
        )

        self.relu = nn.ReLU()

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.dropout = nn.Dropout(0.4)

        self.fc = nn.Linear(32, 6)

    def forward(self, x):

        x = self.conv(x)

        x = self.relu(x)

        x = self.pool(x)

        x = x.squeeze(-1)

        x = self.dropout(x)

        x = self.fc(x)

        return x


def one_hot_encode(sequences):
    """
    Convert DNA sequences into tensors with shape:

        (number_of_sequences, 4, sequence_length)

    Channel order:
        A = 0
        C = 1
        G = 2
        T = 3
    """

    mapping = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3
    }

    encoded = []

    for seq in sequences:
        seq = seq.upper()

        matrix = np.zeros((4, len(seq)), dtype=np.float32)

        for position, nucleotide in enumerate(seq):
            if nucleotide in mapping:
                matrix[mapping[nucleotide], position] = 1.0

        encoded.append(matrix)

    return np.array(encoded, dtype=np.float32)

def class_weights(y_train):
    classes = np.unique(y_train)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    )

    print("Class weights:", class_weights)
    return class_weights

def evaluation(y_test_encoded, y_pred, label_encoder):

    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )

    print(classification_report(
        y_test_encoded,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    print("Accuracy:", accuracy_score(y_test_encoded, y_pred))

    print("Macro Precision:", precision_score(
        y_test_encoded, y_pred,
        average="macro",
        zero_division=0
    ))

    print("Macro Recall:", recall_score(
        y_test_encoded, y_pred,
        average="macro",
        zero_division=0
    ))

    print("Macro F1:", f1_score(
        y_test_encoded, y_pred,
        average="macro",
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test_encoded, y_pred))


if __name__ == "__main__":
    train_seq, train_sigma, test_seq, test_sigma = train_test()
    X_train = one_hot_encode(train_seq)
    X_test = one_hot_encode(test_seq)

    # PyTorch needs integer labels from 0 to 5.
    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(train_sigma)
    y_test_encoded = label_encoder.transform(test_sigma)

    # use 10% of the training data for validation
    X_train_part, X_val, y_train_part, y_val = train_test_split(
        X_train,
        y_train_encoded,
        test_size=0.10,
        stratify=y_train_encoded, # preserves the class proportions in both subsets as much as possible.
        random_state=42
    )

    # weights are calculated using only the training portion
    weights = class_weights(y_train_part)

    #---------------------------
    # Training and early stopping
    #---------------------------
    # Use weighted loss for cross entropy
    # errors on underrepresented classes contribute more to the loss.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SigmaCNN().to(device)

    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )
    # Training and early stopping
    train_dataset = TensorDataset(
        torch.tensor(X_train_part),
        torch.tensor(y_train_part, dtype=torch.long)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val),
        torch.tensor(y_val, dtype=torch.long)
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test),
        torch.tensor(y_test_encoded, dtype=torch.long)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    # ---------------------------
    # Training
    # ---------------------------

    EPOCHS = 100
    PATIENCE = 10

    best_val_loss = float("inf")
    patience_counter = 0

    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):

        # -----------------------
        # Training phase
        # -----------------------

        model.train()

        running_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            loss.backward()

            optimizer.step()

            running_train_loss += loss.item() * X_batch.size(0)

        epoch_train_loss = (
                running_train_loss / len(train_loader.dataset)
        )

        # -----------------------
        # Validation phase
        # -----------------------

        model.eval()

        running_val_loss = 0.0

        with torch.no_grad():

            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)

                loss = criterion(outputs, y_batch)

                running_val_loss += loss.item() * X_batch.size(0)

        epoch_val_loss = (
                running_val_loss / len(val_loader.dataset)
        )

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {epoch_train_loss:.4f} "
            f"Val Loss: {epoch_val_loss:.4f}"
        )

        # -----------------------
        # Early stopping
        # -----------------------

        if epoch_val_loss < best_val_loss:

            best_val_loss = epoch_val_loss

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    # Restore best model
    model.load_state_dict(best_model_state)

    print("Best validation loss:", best_val_loss)

    # ---------------------------
    # Plot training and validation loss
    # ---------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        range(1, len(val_losses) + 1),
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Sigma CNN Training and Validation Loss")

    plt.legend()
    plt.tight_layout()
    plt.savefig("task2_results/task2_sigma_cnn_loss_bonus.png", dpi=300)
    plt.show()
    plt.close()

    # ---------------------------
    # Test evaluation
    # ---------------------------

    model.eval()

    y_pred = []

    with torch.no_grad():

        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)

            outputs = model(X_batch)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

    y_pred = np.array(y_pred)

    evaluation(
        y_test_encoded,
        y_pred,
        label_encoder
    )
