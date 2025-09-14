"""
4_knn.py

This script demonstrates:
- K-Nearest Neighbors (KNN)

Dataset:
- Digits dataset (0-9 handwritten numbers, 8x8 pixel images)

Goal:
Perform digit recognition using KNN,
tune hyperparameters, and visualize predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score


def main():
    digits = load_digits()
    X, y = digits.data, digits.target
    images = digits.images  # original 8x8 images

    # Split keeping stratification (class balance)
    X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
        X, y, images, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: scale features + KNN
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    # Hyperparameter tuning: k neighbors and weight strategy
    param_grid = {
        'knn__n_neighbors': [3, 5, 7],
        'knn__weights': ['uniform', 'distance']
    }
    grid = GridSearchCV(pipe, param_grid, cv=4, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Best params (KNN):", grid.best_params_)
    best = grid.best_estimator_

    # Predictions
    y_pred = best.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Visualize some predictions
    n_show = 12
    plt.figure(figsize=(9, 5))
    for i in range(n_show):
        plt.subplot(3, 4, i + 1)
        plt.imshow(img_test[i], cmap='gray')
        plt.title(f"True: {y_test[i]} / Pred: {y_pred[i]}")
        plt.axis('off')
    plt.suptitle("KNN Predictions - Examples")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Running 4_knn.py ...\n")
    main()
