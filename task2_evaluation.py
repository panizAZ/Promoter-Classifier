import numpy as np
import pandas as pd
import joblib
import os
from task2_feature_extraction import create_feature_matrix, train_test

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_models(test_seq, test_label, best_models, label_encoder):
    os.makedirs("task2_results/confusion_matrices", exist_ok=True)
    class_names = label_encoder.classes_
    class_labels = np.arange(len(class_names))

    results = []
    class_reports = []

    # Cache test features so we don't extract them repeatedly
    test_features = {
        k: create_feature_matrix(test_seq, k)
        for k in [2, 3, 4]
    }

    for (model_name, k, imbalance), model in best_models.items():

        X_test = test_features[k]
        y_pred = model.predict(X_test)

        # Overall metrics
        accuracy = accuracy_score(test_label, y_pred)

        macro_precision, macro_recall, macro_f1, _ = (
            precision_recall_fscore_support(
                test_label,
                y_pred,
                labels=class_labels,
                average="macro",
                zero_division=0
            )
        )

        results.append({
            "Model": model_name,
            "k": k,
            "Imbalance": imbalance,
            "Accuracy": accuracy,
            "Macro Precision": macro_precision,
            "Macro Recall": macro_recall,
            "Macro F1": macro_f1
        })

        # Per-class metrics
        report = classification_report(
            test_label,
            y_pred,
            labels=class_labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

        for sigma in class_names:
            class_reports.append({
                "Model": model_name,
                "k": k,
                "Imbalance": imbalance,
                "Sigma factor": sigma,
                "Precision": report[sigma]["precision"],
                "Recall": report[sigma]["recall"],
                "F1-score": report[sigma]["f1-score"],
                "Support": report[sigma]["support"] # how many data we have of that sigma
            })

        # 4 algorithms * 3 k values * 2 imbalance settings = 24 Confusion matrix
        cm = confusion_matrix(
            test_label,
            y_pred,
            labels=class_labels
        )

        # Plot confusion matrix
        plt.figure(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_
        )

        plt.title(f"{model_name} - k={k} - {imbalance}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        # Create a unique filename
        safe_model_name = model_name.replace(" ", "_")

        filename = (
            f"task2_results/confusion_matrices/"
            f"{safe_model_name}_k{k}_{imbalance}_confusion_matrix.png"
        )

        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

    results_df = pd.DataFrame(results)
    class_reports_df = pd.DataFrame(class_reports)

    return results_df, class_reports_df

if __name__=="__main__":
    train_seq, train_sigma, test_seq, test_sigma = train_test()
    best_models = joblib.load("task2_results/best_models2.pkl")
    label_encoder = joblib.load(
        "task2_results/label_encoder.pkl"
    )

    test_sigma = label_encoder.transform(test_sigma)

    results_df, class_reports_df = evaluate_models(
        test_seq,
        test_sigma,
        best_models,
        label_encoder
    )

    results_df.to_csv(
        "task2_results/test_results.csv",
        index=False
    )

    class_reports_df.to_csv(
        "task2_results/per_class_results.csv",
        index=False
    )