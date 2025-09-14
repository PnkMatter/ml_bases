"""
2_decisiontree_randomforest.py

This script demonstrates:
- Decision Tree Classifier
- Random Forest Classifier

Dataset:
- Iris dataset (3 flower classes)

Goal:
Compare single decision tree vs ensemble (random forest),
examine feature importances, and visualize decision rules.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def main():
    # Load dataset
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Initialize models
    dt = DecisionTreeClassifier(random_state=42, max_depth=4)
    rf = RandomForestClassifier(random_state=42, n_estimators=100)

    # Train
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Predictions
    y_pred_dt = dt.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    print("===== Decision Tree =====")
    print("Accuracy:", accuracy_score(y_test, y_pred_dt))
    print(classification_report(y_test, y_pred_dt))

    print("===== Random Forest =====")
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))

    # Feature importance (Random Forest)
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(6, 4))
    plt.bar([feature_names[i] for i in idx], importances[idx])
    plt.title("Random Forest - Feature Importances")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Show decision rules (text output)
    print("\nDecision Tree rules (top levels):")
    print(export_text(dt, feature_names=feature_names, max_depth=3))


if __name__ == "__main__":
    print("Running 2_decisiontree_randomforest.py ...\n")
    main()
