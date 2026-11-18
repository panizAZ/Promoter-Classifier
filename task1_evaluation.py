import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

from task1_feature_extraction import train_test, create_feature_matrix


def evaluate_models():

    # Load the saved models and CV results
    best_models = joblib.load("task1_results/best_models.pkl")
    results_df = pd.read_csv("task1_results/cv_results.csv")

    # Load the test data
    train_seq, train_label, test_seq, test_label = train_test()

    # Select the best k for each algorithm based on CV F1
    best_k_per_model = (
        results_df.loc[
            results_df.groupby("Model")["Best CV F1"].idxmax()
        ]
    )

    metrics_results = []

    # Store information for plotting
    confusion_matrices = {}
    roc_data = {}

    for _, row in best_k_per_model.iterrows():

        model_name = row["Model"]
        k = int(row["k"])

        print(f"\nEvaluating {model_name} with k={k}")

        # Extract test features
        X_test = create_feature_matrix(test_seq, k)

        # Retrieve the corresponding trained model
        model = best_models[(model_name, k)]

        # Generate predictions
        y_pred = model.predict(X_test)

        # Get prediction scores for ROC and AUC
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(test_label, y_pred)
        precision = precision_score(test_label, y_pred, zero_division=0)
        recall = recall_score(test_label, y_pred, zero_division=0)
        f1 = f1_score(test_label, y_pred, zero_division=0)
        mcc = matthews_corrcoef(test_label, y_pred)
        auc = roc_auc_score(test_label, y_score)

        metrics_results.append({
            "Model": model_name,
            "k": k,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "MCC": mcc,
            "ROC-AUC": auc
        })

        # Store confusion matrix
        confusion_matrices[model_name] = confusion_matrix(
            test_label, y_pred
        )

        # Store ROC curve data
        fpr, tpr, thresholds = roc_curve(test_label, y_score)

        roc_data[model_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc,
            "k": k
        }

    # Create metrics table
    metrics_df = pd.DataFrame(metrics_results)

    print("\nTest Set Evaluation Results:")
    print(metrics_df.round(4).to_string(index=False))

    metrics_df.to_csv("task1_results/test_results.csv", index=False)

    # Plot confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, (model_name, cm) in zip(axes, confusion_matrices.items()):

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Non-promoter (0)", "Promoter (1)"]
        )

        display.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(model_name)

    plt.tight_layout()
    plt.savefig("task1_results/confusion_matrices.png", dpi=300)
    plt.show()

    # Plot ROC curves of all four models in one figure
    plt.figure(figsize=(8, 6))

    for model_name, data in roc_data.items():

        plt.plot(
            data["fpr"],
            data["tpr"],
            label=f'{model_name} (k={data["k"]}, AUC={data["auc"]:.3f})'
        )

    # Random classifier reference line
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Test Set")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("task1_results/roc_curves.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    evaluate_models()