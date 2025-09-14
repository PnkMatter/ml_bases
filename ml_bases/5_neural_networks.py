"""
5_mlp.py

This script demonstrates:
- Neural Network (Multi-Layer Perceptron - MLP)

Dataset:
- Digits dataset (same as KNN for comparison)

Goal:
Train a simple neural network for digit recognition,
compare performance, and visualize the learning curve.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score


def main():
    digits = load_digits()
    X, y = digits.data, digits.target

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: scale features + MLP
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(max_iter=500, early_stopping=True, random_state=42))
    ])

    # Small grid search (hidden layers and regularization)
    param_grid = {
        'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'mlp__alpha': [0.0001, 0.001]  # L2 regularization
    }
    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Best params (MLP):", grid.best_params_)
    best = grid.best_estimator_

    # Predictions
    y_pred = best.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Plot loss curve (how training error decreases over iterations)
    mlp_model = best.named_steps['mlp']
    if hasattr(mlp_model, "loss_curve_"):
        plt.figure(figsize=(6, 4))
        plt.plot(mlp_model.loss_curve_)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Loss Curve - MLP")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("Running 5_mlp.py ...\n")
    main()
