"""
3_gradientboosting_xgboost.py

This script demonstrates:
- Gradient Boosting using either XGBoost (if installed) or sklearn's GradientBoostingClassifier

Dataset:
- Wine dataset (multi-class classification)

Goal:
Show boosting in action, perform simple hyperparameter tuning,
and inspect feature importances.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def main():
    # Load dataset
    data = load_wine()
    X, y = data.data, data.target
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Use XGBoost if available, otherwise sklearn
    if HAS_XGB:
        print("Using XGBoost (XGBClassifier)")
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
    else:
        print("XGBoost not found. Using sklearn.GradientBoostingClassifier")
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }

    # Grid Search to tune hyperparameters
    grid = GridSearchCV(model, param_grid, cv=4, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    print("Best parameters:", grid.best_params_)

    # Predictions
    y_pred = best.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Feature importances
    if hasattr(best, "feature_importances_"):
        importances = best.feature_importances_
        idx = np.argsort(importances)[::-1]

        plt.figure(figsize=(7, 4))
        plt.bar([feature_names[i] for i in idx], importances[idx])
        plt.xticks(rotation=45)
        plt.title("Feature importances (Gradient Boosting)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("Running 3_gradientboosting_xgboost.py ...\n")
    main()
