from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

from task2_feature_extraction import create_feature_matrix
from task2_feature_extraction import train_test

import pandas as pd

import joblib


def tune_models(train_seq, train_label):

    # Stratified CV preserves the proportions of the six sigma-factor classes
    # in each fold.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []
    best_models = {}

    # Compare k = 2, 3, 4 for kmer feature extraction
    for k in [2, 3, 4]:

        print(f"\n{'='*50}")
        print(f"Evaluating k = {k}")
        print(f"{'='*50}")

        train_feature_matrix = create_feature_matrix(train_seq, k)

        ### Train every model twice:
        ### 1. without imbalance handling
        ### 2. with class imbalance handling
        for imbalance in ["imbalanced", "balanced"]:

            # =========================================================
            # MODEL CONFIGURATIONS
            # =========================================================

            if imbalance == "imbalanced":
                # Define models and their hyperparameter grids
                model_configs = {

                    "Logistic Regression": {
                        "model": LogisticRegression(
                            max_iter=2000,
                            random_state=42,
                            class_weight=None
                        ),
                        "params": {
                            "classifier__C": [0.1, 1, 10],
                            ### changed liblinear to saga because it doesn't support multi-labeling
                            "classifier__solver": ["saga", "lbfgs"]
                        },
                        "scaling": True
                    },

                    "SVM": {
                        "model": SVC(
                            class_weight=None
                        ),
                        "params": [
                            {
                                "classifier__kernel": ["linear"],
                                "classifier__C": [0.1, 1, 10]
                            },
                            {
                                "classifier__kernel": ["rbf"],
                                "classifier__C": [0.1, 1, 10],
                                "classifier__gamma": ["scale", "auto"]
                            }
                        ],
                        "scaling": True
                    },

                    "Random Forest": {
                        "model": RandomForestClassifier(
                            random_state=42,
                            n_jobs=1,
                            class_weight=None
                        ),
                        "params": {
                            "classifier__n_estimators": [200, 500],
                            "classifier__max_depth": [None, 10, 20],
                            "classifier__min_samples_split": [2, 5],
                            "classifier__max_features": ["sqrt"]
                        },
                        "scaling": False
                    },

                    "XGBoost": {
                        "model": XGBClassifier(
                            random_state=42,
                            eval_metric="mlogloss",
                            n_jobs=1
                        ),
                        "params": {
                            "classifier__n_estimators": [200, 400],
                            "classifier__max_depth": [3, 5],
                            "classifier__learning_rate": [0.03, 0.1],
                            "classifier__subsample": [0.8, 1.0],
                            "classifier__colsample_bytree": [0.8, 1.0]
                        },
                        "scaling": False
                    }
                }
                sample_weight = None

            # =========================================================
            # BALANCED VERSION
            # =========================================================
            else:
                model_configs = {

                    "Logistic Regression": {
                        "model": LogisticRegression(
                            max_iter=2000,
                            random_state=42,
                            class_weight="balanced"
                        ),
                        "params": {
                            "classifier__C": [0.1, 1, 10],
                            "classifier__solver": ["saga", "lbfgs"]
                        },
                        "scaling": True
                    },

                    "SVM": {
                        "model": SVC(
                            class_weight="balanced"
                        ),
                        "params": [
                            {
                                "classifier__kernel": ["linear"],
                                "classifier__C": [0.1, 1, 10]
                            },
                            {
                                "classifier__kernel": ["rbf"],
                                "classifier__C": [0.1, 1, 10],
                                "classifier__gamma": ["scale", "auto"]
                            }
                        ],
                        "scaling": True
                    },

                    "Random Forest": {
                        "model": RandomForestClassifier(
                            random_state=42,
                            n_jobs=1,
                            class_weight="balanced"
                        ),
                        "params": {
                            "classifier__n_estimators": [200, 500],
                            "classifier__max_depth": [None, 10, 20],
                            "classifier__min_samples_split": [2, 5],
                            "classifier__max_features": ["sqrt"]
                        },
                        "scaling": False
                    },

                    "XGBoost": {
                        "model": XGBClassifier(
                            random_state=42,
                            eval_metric="mlogloss",
                            n_jobs=1
                        ),
                        "params": {
                            "classifier__n_estimators": [200, 400],
                            "classifier__max_depth": [3, 5],
                            "classifier__learning_rate": [0.03, 0.1],
                            "classifier__subsample": [0.8, 1.0],
                            "classifier__colsample_bytree": [0.8, 1.0]
                        },
                        "scaling": False
                    }
                }
                # Balanced sample weights for XGBoost , we calculate the weights ourselves
                sample_weight = compute_sample_weight(
                    class_weight="balanced",
                    y=train_label
                )

            # =========================================================
            # TRAIN ALL FOUR MODELS
            # =========================================================
            for model_name, config in model_configs.items():
                # Create pipeline
                if config["scaling"]:
                    pipeline = Pipeline([
                        ("scaler", StandardScaler()),
                        ("classifier", config["model"])
                    ])
                else:
                    pipeline = Pipeline([
                        ("classifier", config["model"])
                    ])

                # Grid search with cross-validation
                # finds the best parameters for each model with cross validation using f1 macro metric
                grid_search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=config["params"],
                    cv=cv, # 5-fold cross-validation for each hyperparameter combination
                    scoring="f1_macro",  # for multi class alpha instead of f1 (for binary promoter vs non-promoter)(not micro_f1 and normal f1)
                    n_jobs=-1, # use all available CPU cores for parallel model fitting
                    refit=True # آموزش مجدد بهترین مدل روی کل training set
                )

                # -----------------------------------------------------
                # XGBoost balanced version needs sample weights
                # -----------------------------------------------------
                if (
                        model_name == "XGBoost"
                        and imbalance == "balanced"
                ):

                    grid_search.fit(
                        train_feature_matrix,
                        train_label,
                        classifier__sample_weight=sample_weight
                    )

                else:

                    grid_search.fit(
                        train_feature_matrix,
                        train_label
                    )

                # Save results
                results.append({
                    "Model": model_name,
                    "k": k,
                    "Imbalance": imbalance,
                    "Best CV F1 Macro": grid_search.best_score_,
                    "Best Parameters": grid_search.best_params_
                })

                # Save fitted model
                best_models[
                    (model_name, k, imbalance)
                ] = grid_search.best_estimator_

                print(f"\n{model_name}")
                print(
                    "Best CV F1 Macro:",
                    round(grid_search.best_score_, 4)
                )
                print(
                    "Best parameters:",
                    grid_search.best_params_
                )

    results_df = pd.DataFrame(results)
    return results_df, best_models


if __name__ == "__main__":
    train_seq, train_sigma, test_seq, test_sigma = train_test()

    # Convert sigma-factor labels into integers (0 to 5) for XGBoost's sample weights
    # no need for seed
    label_encoder = LabelEncoder()

    train_sigma = label_encoder.fit_transform(train_sigma)

    # Transform test labels using the same encoder
    test_sigma = label_encoder.transform(test_sigma)

    print("Class mapping:")
    for i, sigma in enumerate(label_encoder.classes_):
        print(f"{sigma} -> {i}")

    joblib.dump(
        label_encoder,
        "task2_results/label_encoder.pkl"
    )

    results_df, best_models = tune_models(
        train_seq,
        train_sigma
    )

    # shows all 24 trained models
    # (4 algorithms × 3 k values × 2 imbalance settings)
    ##### other than 5 cross validations for each model and different parameters for each
    print("\nAll CV results:")
    print(
        results_df.sort_values(
            by="Best CV F1 Macro",
            ascending=False
        )
    )

    results_df.to_csv(
        "task2_results/cv_results2.csv",
        index=False
    )

    # Saves all 24 fitted models
    joblib.dump(
        best_models,
        "task2_results/best_models2.pkl"
    )

