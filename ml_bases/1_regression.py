"""
1_regression.py

This script demonstrates:
- Linear Regression (continuous target prediction)
- Logistic Regression (binary classification)

Datasets:
- Diabetes dataset (regression)
- Breast Cancer dataset (classification)

Goal:
Show how regression models are trained, evaluated, and visualized.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)


def run_linear_regression():
    """Linear Regression example on the Diabetes dataset."""
    data = load_diabetes()
    X, y = data.data, data.target  # Features and target

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create pipeline: StandardScaler (normalization) + Linear Regression
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    pipe.fit(X_train, y_train)

    # Make predictions
    y_pred = pipe.predict(X_test)

    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("===== Linear Regression (Diabetes dataset) =====")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    # Visualize predicted vs actual values
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle='--',
        color="red"
    )
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Linear Regression - True vs Predicted")
    plt.tight_layout()
    plt.show()


def run_logistic_regression():
    """Logistic Regression example on the Breast Cancer dataset."""
    data = load_breast_cancer()
    X, y = data.data, data.target  # Binary target: 0 = malignant, 1 = benign

    # Stratify=y ensures class balance between train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: scale features + Logistic Regression
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=2000))
    ])
    pipe.fit(X_train, y_train)

    # Predictions
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]  # Probability for ROC

    print("\n===== Logistic Regression (Breast Cancer dataset) =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Logistic Regression")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Running 1_regression.py ...\n")
    run_linear_regression()
    run_logistic_regression()
