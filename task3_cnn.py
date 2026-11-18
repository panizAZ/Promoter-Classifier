import os
import copy
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from task1_feature_extraction import train_test
import logomaker


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ============================================================
# 2. SETTINGS
# ============================================================

DATA_PATH = "data/task1_promoter.csv"

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001

KERNEL_SIZE = 10
NUM_FILTERS = 32
DROPOUT = 0.5

PATIENCE = 10

TSS_INDEX = 60       # position 61 in 1-based numbering


# ============================================================
# 3. ONE-HOT ENCODING
# ============================================================

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


# ============================================================
# 4. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

sequences = df["sequence"].str.upper().values
labels = df["label"].values.astype(np.float32)

print("\nDataset:")
print("Number of sequences:", len(sequences))
print("Sequence length:", len(sequences[0]))
print("Promoters:", np.sum(labels == 1))
print("Non-promoters:", np.sum(labels == 0))


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

# already have a fixed Task 1 train/test split,
# replace this with Task 1 split so that the CNN uses
# exactly the same test set as Task 1.

X_train_seq, y_train, X_test_seq, y_test = train_test()

# ============================================================
# 6. TRAIN / VALIDATION SPLIT
# ============================================================

# 10% of TRAINING data becomes validation data.

X_train_seq, X_val_seq, y_train, y_val = train_test_split(
    X_train_seq,
    y_train,
    test_size=0.10,
    stratify=y_train,
    random_state=SEED
)


print("\nSplits:")
print("Training:", len(X_train_seq))
print("Validation:", len(X_val_seq))
print("Test:", len(X_test_seq))


# ============================================================
# 7. ONE-HOT ENCODE
# ============================================================

X_train = one_hot_encode(X_train_seq)
X_val = one_hot_encode(X_val_seq)
X_test = one_hot_encode(X_test_seq)

print("\nOne-hot shape:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# Expected:
#
# (number_of_sequences, 4, 81)
#
# Example:
#
# (4117, 4, 81)


# ============================================================
# 8. CREATE PYTORCH DATASETS
# ============================================================

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train.to_numpy().ravel(), dtype=torch.float32)
)

val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val.to_numpy().ravel(), dtype=torch.float32)
)

test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test.to_numpy().ravel(), dtype=torch.float32)
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# 9. CNN MODEL
# ============================================================

class PromoterCNN(nn.Module):

    def __init__(
        self,
        num_filters=32,
        kernel_size=15,
        dropout=0.5
    ):

        super().__init__()

        # Input:
        # (batch, 4, 81)

        self.conv = nn.Conv1d(
            in_channels=4,
            out_channels=num_filters,
            kernel_size=kernel_size
        )

        self.relu = nn.ReLU()

        # Global max pooling
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            num_filters,
            1
        )

    def forward(self, x, return_activations=False):

        # Convolution
        x = self.conv(x)

        # ReLU
        x = self.relu(x)

        # Save activation map before pooling
        activations = x

        # Global max pooling
        x = self.global_pool(x)

        # Remove final dimension
        x = x.squeeze(-1)

        # Dropout
        x = self.dropout(x)

        # Dense output
        x = self.fc(x)

        # Output shape:
        # (batch, 1)

        if return_activations:
            return x, activations

        return x


model = PromoterCNN(
    num_filters=NUM_FILTERS,
    kernel_size=KERNEL_SIZE,
    dropout=DROPOUT
).to(device)


print("\nModel:")
print(model)


# ============================================================
# 10. LOSS + OPTIMIZER
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 11. TRAINING WITH EARLY STOPPING
# ============================================================

train_losses = []
val_losses = []

best_val_loss = float("inf")
best_model_state = None

patience_counter = 0


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    total_train_loss = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(
            outputs.squeeze(1),
            y_batch
        )

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item() * len(X_batch)

    train_loss = total_train_loss / len(train_loader.dataset)


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    total_val_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)

            loss = criterion(
                outputs.squeeze(1),
                y_batch
            )

            total_val_loss += loss.item() * len(X_batch)

    val_loss = total_val_loss / len(val_loader.dataset)


    train_losses.append(train_loss)
    val_losses.append(val_loss)


    print(
        f"Epoch {epoch + 1:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        patience_counter = 0

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print("\nEarly stopping.")

            break


# Restore best model
model.load_state_dict(best_model_state)


# ============================================================
# 12. PLOT TRAINING / VALIDATION LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    train_losses,
    label="Training loss"
)

plt.plot(
    val_losses,
    label="Validation loss"
)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("CNN Training and Validation Loss")

plt.legend()
plt.tight_layout()

plt.savefig(
    "task3_filter_results/task3_loss_curve.png",
    dpi=300
)

plt.show()


# ============================================================
# 13. PREDICTION FUNCTION
# ============================================================

def predict_probabilities(model, loader):

    model.eval()

    probabilities = []
    true_labels = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)

            logits = model(X_batch)

            probs = torch.sigmoid(logits)

            probabilities.extend(
                probs.squeeze(1).cpu().numpy()
            )

            true_labels.extend(
                y_batch.numpy()
            )

    return (
        np.array(probabilities),
        np.array(true_labels)
    )


# ============================================================
# 14. TASK 1 METRICS
# ============================================================

test_probabilities, test_labels = predict_probabilities(
    model,
    test_loader
)

test_predictions = (
    test_probabilities >= 0.5
).astype(int)


accuracy = accuracy_score(
    test_labels,
    test_predictions
)

precision = precision_score(
    test_labels,
    test_predictions
)

recall = recall_score(
    test_labels,
    test_predictions
)

f1 = f1_score(
    test_labels,
    test_predictions
)

mcc = matthews_corrcoef(
    test_labels,
    test_predictions
)

roc_auc = roc_auc_score(
    test_labels,
    test_probabilities
)


print("\n==============================")
print("CNN TEST RESULTS")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}")
print(f"MCC      : {mcc:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(
    test_labels,
    test_predictions
)

print("\nConfusion Matrix:")
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(6, 5))

plt.imshow(cm, interpolation="nearest")
plt.title("Task 3 CNN - Confusion Matrix")
plt.colorbar()

plt.xticks([0, 1], ["Non-Promoter", "Promoter"])
plt.yticks([0, 1], ["Non-Promoter", "Promoter"])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Add numbers inside the cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.savefig(
    "task3_filter_results/task3_confusion_matrix.png",
    dpi=300
)
plt.show()


# ============================================================
# 16. ROC CURVE
# ============================================================

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(
    test_labels,
    test_probabilities
)

plt.figure(figsize=(6, 6))

plt.plot(
    fpr,
    tpr,
    label=f"CNN (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("CNN ROC Curve")

plt.legend()
plt.tight_layout()

plt.savefig(
    "task3_filter_results/task3_roc_curve.png",
    dpi=300
)

plt.show()


# ============================================================
# 17. SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "task3_filter_results/task3_cnn_model.pt"
)


# ============================================================
# 18. FILTER ACTIVATION EXTRACTION
# ============================================================

def get_filter_activations(
    model,
    sequences,
    labels
):

    X = one_hot_encode(sequences)

    dataset = TensorDataset(
        torch.tensor(X),
        torch.tensor(labels)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model.eval()

    all_activations = []

    with torch.no_grad():

        for X_batch, _ in loader:

            X_batch = X_batch.to(device)

            _, activations = model(
                X_batch,
                return_activations=True
            )

            # Shape:
            # (batch, filters, positions)

            all_activations.append(
                activations.cpu().numpy()
            )

    return np.concatenate(
        all_activations,
        axis=0
    )


# ============================================================
# 19. USE TEST PROMOTERS ONLY FOR MOTIF DISCOVERY
# ============================================================

test_promoter_mask = (
    test_labels == 1
)

test_promoter_sequences = (
    X_test_seq[test_promoter_mask]
)

test_promoter_labels = (
    test_labels[test_promoter_mask]
)


activations = get_filter_activations(
    model,
    test_promoter_sequences,
    test_promoter_labels
)


print("\nActivation shape:")
print(activations.shape)


# Expected:
#
# (number_of_test_promoters,
#  32,
#  81 - 15 + 1)
#
# = (N, 32, 67)
#
# because:
#
# 81 - 15 + 1 = 67


# ============================================================
# 20. CREATE PFM FOR EACH FILTER
# ============================================================

nucleotides = ["A", "C", "G", "T"]


def get_filter_windows(
    sequences,
    activation_matrix,
    filter_index,
    kernel_size,
    threshold_fraction=0.5
):

    # activation_matrix shape:
    #
    # (number_of_sequences,
    #  number_of_filters,
    #  activation_positions)

    filter_activation = (
        activation_matrix[:, filter_index, :]
    )

    max_activation = filter_activation.max()

    threshold = (
        threshold_fraction *
        max_activation
    )

    windows = []
    activation_positions = []
    activation_values = []

    for seq_index, seq in enumerate(sequences):

        for position in range(
            filter_activation.shape[1]
        ):

            activation = filter_activation[
                seq_index,
                position
            ]

            if activation >= threshold:

                window = seq[
                    position:
                    position + kernel_size
                ]

                if len(window) == kernel_size:

                    windows.append(window)

                    activation_positions.append(
                        position
                    )

                    activation_values.append(
                        activation
                    )

    return (
        windows,
        activation_positions,
        activation_values,
        max_activation
    )


def create_pfm(windows, kernel_size):

    counts = np.zeros(
        (kernel_size, 4),
        dtype=int
    )

    nucleotide_to_index = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3
    }

    for window in windows:

        for position, nucleotide in enumerate(window):

            if nucleotide in nucleotide_to_index:

                index = nucleotide_to_index[nucleotide]

                counts[position, index] += 1

    return counts


# ============================================================
# 21. PFM → INFORMATION CONTENT
# ============================================================

def pfm_to_information_content(pfm):

    # Add pseudocount to avoid log(0)

    pseudocount = 1e-6

    probabilities = (
        pfm + pseudocount
    ) / (
        pfm.sum(axis=1, keepdims=True)
        + 4 * pseudocount
    )

    # Information content:
    #
    # IC = 2 - H
    #
    # where H is entropy for DNA

    entropy = -np.sum(
        probabilities *
        np.log2(probabilities),
        axis=1
    )

    information = 2 - entropy

    logo_matrix = (
        probabilities *
        information[:, None]
    )

    return pd.DataFrame(
        logo_matrix,
        columns=nucleotides
    )


# ============================================================
# 22. DRAW SEQUENCE LOGO
# ============================================================

def plot_sequence_logo(
    logo_matrix,
    filter_index,
    number_of_windows
):

    fig, ax = plt.subplots(figsize=(10, 4))

    logomaker.Logo(
        logo_matrix,
        ax=ax
    )

    ax.set_title(
        f"Filter {filter_index + 1} "
        f"Sequence Logo "
        f"(n={number_of_windows})"
    )

    ax.set_xlabel("Position")
    ax.set_ylabel("Information (bits)")

    fig.tight_layout()

    filename = (
        f"filter_{filter_index + 1}_logo.png"
    )

    os.makedirs(
        "task3_filter_results/logo",
        exist_ok=True
    )

    fig.savefig(
        f"task3_filter_results/logo/{filename}",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 23. CREATE LOGOS FOR ALL 32 FILTERS
# ============================================================

filter_results = {}


for filter_index in range(NUM_FILTERS):

    (
        windows,
        positions,
        values,
        max_activation
    ) = get_filter_windows(
        test_promoter_sequences,
        activations,
        filter_index,
        KERNEL_SIZE,
        threshold_fraction=0.5
    )

    if len(windows) == 0:

        print(
            f"Filter {filter_index + 1}: "
            f"No windows above threshold."
        )

        continue

    pfm = create_pfm(
        windows,
        KERNEL_SIZE
    )

    logo_matrix = (
        pfm_to_information_content(pfm)
    )

    filter_results[filter_index] = {
        "windows": windows,
        "positions": positions,
        "activation_values": values,
        "max_activation": max_activation,
        "pfm": pfm,
        "logo_matrix": logo_matrix
    }

    print(
        f"Filter {filter_index + 1}: "
        f"{len(windows)} windows, "
        f"max activation = {max_activation:.4f}"
    )


# ============================================================
# 24. SAVE PFMs
# ============================================================

os.makedirs(
    "task3_filter_results",
    exist_ok=True
)
os.makedirs(
    "task3_filter_results/PFM",
    exist_ok=True
)

for filter_index, result in filter_results.items():

    result["pfm_df"] = pd.DataFrame(
        result["pfm"],
        columns=nucleotides
    )

    result["pfm_df"].to_csv(
        f"task3_filter_results/PFM/"
        f"filter_{filter_index + 1}_PFM.csv",
        index=False
    )


# ============================================================
# 25. RANK FILTERS BY INFORMATION CONTENT
# ============================================================

filter_scores = []

for filter_index, result in filter_results.items():

    logo = result["logo_matrix"]

    total_information = logo.values.sum()

    filter_scores.append({
        "Filter": filter_index,
        "Total Information": total_information,
        "Number of windows": len(
            result["windows"]
        ),
        "Maximum activation": result[
            "max_activation"
        ]
    })


filter_scores_df = pd.DataFrame(
    filter_scores
)

filter_scores_df = filter_scores_df.sort_values(
    "Total Information",
    ascending=False
)

print("\nFilters ranked by information content:")
print(filter_scores_df)


filter_scores_df.to_csv(
    "task3_filter_results/filter_ranking.csv",
    index=False
)


# ============================================================
# 26. PLOT THE THREE CLEAREST FILTERS
# ============================================================

top_filters = (
    filter_scores_df
    .head(3)["Filter"]
    .tolist()
)

print(
    "\nTop 3 filters based on information content:",
    [x + 1 for x in top_filters]
)

for filter_index, result in filter_results.items():

    plot_sequence_logo(
        result["logo_matrix"],
        filter_index,
        len(result["windows"])
    )


# ============================================================
# 27. WHERE DOES EACH FILTER ACTIVATE STRONGLY?
# ============================================================

def plot_activation_positions(
    sequences,
    activation_matrix,
    filter_index
):

    filter_activation = activation_matrix[
        :,
        filter_index,
        :
    ]

    # Maximum activation of this filter
    max_activation = filter_activation.max()

    # Keep all activations >= 50% of the filter maximum
    threshold = 0.5 * max_activation

    activation_positions = []

    for seq_index in range(
        filter_activation.shape[0]
    ):

        for position in range(
            filter_activation.shape[1]
        ):

            activation = filter_activation[
                seq_index,
                position
            ]

            if activation >= threshold:
                activation_positions.append(
                    position
                )

    activation_positions = np.array(
        activation_positions
    )

    # Convert window-start positions
    # to positions relative to TSS
    relative_positions = (
        activation_positions - TSS_INDEX
    )

    plt.figure(figsize=(9, 5))

    plt.hist(
        relative_positions,
        bins=np.arange(
            relative_positions.min() - 0.5,
            relative_positions.max() + 1.5,
            1
        )
    )

    plt.axvline(
        -35,
        linestyle="--",
        label="-35"
    )

    plt.axvline(
        -10,
        linestyle="--",
        label="-10"
    )

    plt.axvline(
        0,
        linestyle="--",
        label="TSS"
    )

    plt.xlabel(
        "Strong filter activation window start relative to TSS"
    )

    plt.ylabel(
        "Number of strongly activating windows"
    )

    plt.title(
        f"Filter {filter_index + 1} "
        f"Strong Activation Positions"
    )

    plt.legend()

    plt.tight_layout()

    os.makedirs(
        "task3_filter_results/filter_position",
        exist_ok=True
    )

    plt.savefig(
        f"task3_filter_results/filter_position/"
        f"filter_{filter_index + 1}_strong_positions.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

for filter_index in filter_results:

    plot_activation_positions(
        test_promoter_sequences,
        activations,
        filter_index
    )


# ============================================================
# 28. IN-SILICO MUTAGENESIS
# ============================================================

def predict_single_sequence(
    model,
    sequence
):

    X = one_hot_encode(
        [sequence]
    )

    X = torch.tensor(
        X,
        dtype=torch.float32
    ).to(device)

    model.eval()

    with torch.no_grad():

        logits = model(X)

        probability = torch.sigmoid(
            logits
        ).item()

    return probability


# ============================================================
# 29. SELECT 20 HIGH-CONFIDENCE CORRECTLY
#     CLASSIFIED TEST PROMOTERS
# ============================================================

# For promoters, "correctly classified with high
# confidence" means:
#
# true label = 1
# predicted probability is high.

promoter_indices = np.where(
    test_labels == 1
)[0]

correct_promoter_indices = (
    promoter_indices[
        test_probabilities[
            promoter_indices
        ] >= 0.90
    ]
)

# Sort by confidence

correct_promoter_indices = sorted(
    correct_promoter_indices,
    key=lambda i: test_probabilities[i],
    reverse=True
)

selected_indices = (
    correct_promoter_indices[:20]
)

print(
    "\nSelected sequences for mutagenesis:",
    len(selected_indices)
)

if len(selected_indices) < 20:

    print(
        "WARNING: Fewer than 20 promoters "
        "have confidence >= 0.90."
    )

    # Use the 20 highest-confidence
    # correctly classified promoters instead.

    selected_indices = sorted(
        promoter_indices,
        key=lambda i: test_probabilities[i],
        reverse=True
    )[:20]


# ============================================================
# 30. IN-SILICO MUTAGENESIS
# ============================================================

nucleotide_list = [
    "A",
    "C",
    "G",
    "T"
]

importance_matrix = []
test_promoter_sequences = np.array(test_promoter_sequences)

for index in selected_indices:

    original_sequence = test_promoter_sequences[
        np.where(
            promoter_indices == index
        )[0][0]
    ]

    original_probability = (
        test_probabilities[index]
    )

    sequence_importance = np.zeros(
        len(original_sequence)
    )

    for position in range(
        len(original_sequence)
    ):

        original_nucleotide = (
            original_sequence[position]
        )

        mutation_drops = []

        for nucleotide in nucleotide_list:

            if nucleotide == original_nucleotide:
                continue

            mutated_sequence = list(
                original_sequence
            )

            mutated_sequence[position] = (
                nucleotide
            )

            mutated_sequence = "".join(
                mutated_sequence
            )

            mutated_probability = (
                predict_single_sequence(
                    model,
                    mutated_sequence
                )
            )

            drop = (
                original_probability
                - mutated_probability
            )

            mutation_drops.append(
                drop
            )

        # Average effect of the three
        # possible mutations.

        sequence_importance[position] = (
            np.mean(mutation_drops)
        )

    importance_matrix.append(
        sequence_importance
    )


importance_matrix = np.array(
    importance_matrix
)


# Average across 20 sequences

mean_importance = (
    importance_matrix.mean(axis=0)
)


std_importance = (
    importance_matrix.std(axis=0)
)


# ============================================================
# 31. PLOT MUTAGENESIS IMPORTANCE
# ============================================================

positions = np.arange(
    1,
    len(mean_importance) + 1
)

relative_positions = (
    positions - 1 - TSS_INDEX
)


plt.figure(figsize=(12, 5))

plt.plot(
    relative_positions,
    mean_importance
)

plt.axvline(
    -35,
    linestyle="--",
    label="-35"
)

plt.axvline(
    -10,
    linestyle="--",
    label="-10"
)

plt.axvline(
    0,
    linestyle="--",
    label="TSS"
)

plt.xlabel(
    "Position relative to TSS"
)

plt.ylabel(
    "Average drop in promoter probability"
)

plt.title(
    "In-silico Mutagenesis Importance"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "task3_filter_results/task3_mutagenesis.png",
    dpi=300
)

plt.show()


# ============================================================
# 32. SAVE MUTAGENESIS RESULTS
# ============================================================

mutagenesis_df = pd.DataFrame({
    "Position": positions,
    "Relative_to_TSS": relative_positions,
    "Mean_Importance": mean_importance,
    "Std_Importance": std_importance
})

mutagenesis_df.to_csv(
    "task3_filter_results/task3_mutagenesis_results.csv",
    index=False
)


print("\nTask 3 completed.")