from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from task1_feature_extraction import create_feature_matrix
from task1_feature_extraction import train_test

import pandas as pd

import joblib

def tune_models(train_seq, train_label):

    # stratified to balance out the promoter vs non-promoter classes
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Define models and their hyperparameter grids
    model_configs = {

        "Logistic Regression": {
            "model": LogisticRegression(
                max_iter=2000,
                random_state=42
            ),
            "params": {
                "classifier__C": [0.1, 1, 10],
                "classifier__solver": ["liblinear", "lbfgs"]
            },
            "scaling": True
        },

        "SVM": {
            "model": SVC(),
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
                n_jobs=1
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
                eval_metric="logloss",
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

    results = []
    best_models = {}

    # Compare k = 2, 3, 4 for kmer feature extraction
    for k in [2, 3, 4]:

        print(f"\n{'='*50}")
        print(f"Evaluating k = {k}")
        print(f"{'='*50}")

        train_feature_matrix = create_feature_matrix(train_seq, k)

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
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=config["params"],
                cv=cv, # 5-fold cross-validation for each hyperparameter combination
                scoring="f1",  # بنابراین F1 به‌صورت پیش‌فرض برای کلاس مثبت یعنی promoter محاسبه می‌شود
                n_jobs=-1, #  # use all available CPU cores for parallel model fitting
                refit=True # آموزش مجدد بهترین مدل روی کل training set
            )

            grid_search.fit(train_feature_matrix, train_label)

            # Save results
            results.append({
                "Model": model_name,
                "k": k,
                "Best CV F1": grid_search.best_score_,
                "Best Parameters": grid_search.best_params_
            })

            best_models[(model_name, k)] = grid_search.best_estimator_

            print(f"\n{model_name}")
            print("Best CV F1:", round(grid_search.best_score_, 4))
            print("Best parameters:", grid_search.best_params_)

    results_df = pd.DataFrame(results)

    return results_df, best_models

if __name__ == "__main__":
    train_seq, train_label, test_seq, test_label = train_test()
    results_df, best_models = tune_models(train_seq, train_label)

    # shows all 12 trained models (4 algorithms × 3 k values)
    print("\nAll CV results:")
    print(
        results_df.sort_values(
            by="Best CV F1",
            ascending=False
        )
    )
    results_df.to_csv("task1_results/cv_results.csv", index=False)

    # saves all 12 trained models (4 algorithms × 3 k values) in one file
    joblib.dump(best_models, "task1_results/best_models.pkl")