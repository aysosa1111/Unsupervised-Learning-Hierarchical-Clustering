# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 18:07:49 2024

Author: Owner
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")  # To ignore warnings for cleaner output

def load_olivetti_faces():
    """
    Load the Olivetti faces dataset.
    Returns:
        X: numpy array of shape (400, 64, 64)
        y: numpy array of shape (400,)
    """
    data = fetch_olivetti_faces()
    X = data.images
    y = data.target
    return X, y

def split_dataset(X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    """
    Split the dataset into training, validation, and test sets using stratified sampling.
    Args:
        X: numpy array of features
        y: numpy array of labels
        train_size: float, proportion of the dataset to include in the train split
        val_size: float, proportion for validation
        test_size: float, proportion for test
        random_state: int, random seed
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    assert train_size + val_size + test_size == 1.0, "Sum of splits must be 1.0"
    
    sss_train_val = StratifiedShuffleSplit(n_splits=1, test_size=(val_size + test_size), random_state=random_state)
    for train_index, temp_index in sss_train_val.split(X, y):
        X_train, X_temp = X[train_index], X[temp_index]
        y_train, y_temp = y[train_index], y[temp_index]
    
    relative_val_size = val_size / (val_size + test_size)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=(1 - relative_val_size), random_state=random_state)
    for val_index, test_index in sss_val.split(X_temp, y_temp):
        X_val, X_test = X_temp[val_index], X_temp[test_index]
        y_val, y_test = y_temp[val_index], y_temp[test_index]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def flatten_images(X):
    """
    Flatten the image data.
    Args:
        X: numpy array of shape (n_samples, height, width)
    Returns:
        X_flat: numpy array of shape (n_samples, height*width)
    """
    n_samples = X.shape[0]
    return X.reshape((n_samples, -1))

def compute_cluster_centroids(X, labels, n_clusters):
    """
    Compute centroids of clusters.
    Args:
        X: numpy array of features
        labels: numpy array of cluster labels
        n_clusters: int, number of clusters
    Returns:
        centroids: numpy array of shape (n_clusters, n_features)
    """
    centroids = np.zeros((n_clusters, X.shape[1]))
    for k in range(n_clusters):
        if np.sum(labels == k) == 0:
            centroids[k] = 0
        else:
            centroids[k] = X[labels == k].mean(axis=0)
    return centroids

def assign_clusters(X, centroids, similarity_measure, p=None):
    """
    Assign clusters to new data based on nearest centroids.
    Args:
        X: numpy array of features
        centroids: numpy array of cluster centroids
        similarity_measure: str, 'euclidean', 'minkowski', or 'cosine'
        p: int, parameter for minkowski distance
    Returns:
        labels: numpy array of assigned cluster labels
    """
    if similarity_measure == 'minkowski' and p is not None:
        # Compute Minkowski distance
        distances = np.sum(np.abs(X[:, np.newaxis] - centroids) ** p, axis=2) ** (1/p)
    elif similarity_measure == 'cosine':
        # Normalize the data for cosine similarity
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        centroids_norm = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        distances = 1 - np.dot(X_norm, centroids_norm.T)
    else:  # 'euclidean'
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    
    labels = np.argmin(distances, axis=1)
    return labels

def train_evaluate_classifier(X_train, y_train, X_val, y_val, n_splits=5, random_state=42):
    """
    Train a classifier using k-fold cross-validation and evaluate on the validation set.
    Args:
        X_train: numpy array of training features
        y_train: numpy array of training labels
        X_val: numpy array of validation features
        y_val: numpy array of validation labels
        n_splits: int, number of folds
        random_state: int, random seed
    Returns:
        avg_cv_accuracy: float, average cross-validation accuracy
        val_accuracy: float, accuracy on validation set
    """
    scaler = StandardScaler()
    classifier = SVC(kernel='linear', random_state=random_state)
    pipeline = Pipeline([
        ('scaler', scaler),
        ('svc', classifier)
    ])
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_accuracies = []
    
    for train_idx, test_idx in skf.split(X_train, y_train):
        X_tr, X_te = X_train[train_idx], X_train[test_idx]
        y_tr, y_te = y_train[train_idx], y_train[test_idx]
        
        pipeline.fit(X_tr, y_tr)
        acc = pipeline.score(X_te, y_te)
        cv_accuracies.append(acc)
    
    avg_cv_accuracy = np.mean(cv_accuracies)
    
    # Train on full training set and evaluate on validation
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    return avg_cv_accuracy, val_accuracy

def perform_clustering(X, similarity_measure, linkage='average', max_clusters=20, random_state=42):
    """
    Perform Agglomerative Hierarchical Clustering with specified similarity measure.
    Args:
        X: numpy array of features
        similarity_measure: str, 'euclidean', 'minkowski', or 'cosine'
        linkage: str, linkage method
        max_clusters: int, maximum number of clusters to try
        random_state: int, random seed
    Returns:
        best_clusters: int, number of clusters with highest silhouette score
        best_labels: numpy array of cluster labels
        silhouette_scores: dict, mapping from number of clusters to silhouette score
    """
    silhouette_scores = {}
    cluster_labels = {}
    
    for n_clusters in range(2, max_clusters + 1):
        try:
            if similarity_measure == 'minkowski':
                # Compute pairwise Minkowski distance with p=3
                distance_matrix = squareform(pdist(X, metric='minkowski', p=3))
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric='precomputed',
                    linkage=linkage
                )
                labels = clustering.fit_predict(distance_matrix)
                # Compute silhouette score using Minkowski distance
                silhouette = silhouette_score(X, labels, metric='minkowski', p=3)
            else:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric=similarity_measure,
                    linkage=linkage
                )
                labels = clustering.fit_predict(X)
                silhouette = silhouette_score(X, labels, metric=similarity_measure)
            
            silhouette_scores[n_clusters] = silhouette
            cluster_labels[n_clusters] = labels
        except Exception as e:
            print(f"Clustering failed for n_clusters={n_clusters} with error: {e}")
            silhouette_scores[n_clusters] = -1
            cluster_labels[n_clusters] = np.zeros(X.shape[0])
    
    # Select number of clusters with highest silhouette score
    best_clusters = max(silhouette_scores, key=silhouette_scores.get)
    best_labels = cluster_labels[best_clusters]
    
    return best_clusters, best_labels, silhouette_scores

def plot_silhouette_scores(silhouette_scores, measure):
    """
    Plot Silhouette Scores for different numbers of clusters.
    Args:
        silhouette_scores: dict, mapping from number of clusters to silhouette score
        measure: str, similarity measure name
    """
    plt.figure(figsize=(10, 6))
    plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o')
    plt.title(f'Silhouette Scores for {measure.capitalize()} Distance')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()

def reduce_dimensionality_via_clustering(X_train, X_val, X_test, labels, n_clusters, similarity_measure, p=None):
    """
    Reduce dimensionality by one-hot encoding cluster labels.
    Args:
        X_train: numpy array of training features
        X_val: numpy array of validation features
        X_test: numpy array of test features
        labels: numpy array of cluster labels for training data
        n_clusters: int, number of clusters
        similarity_measure: str, similarity measure used
        p: int, parameter for minkowski distance
    Returns:
        X_train_reduced, X_val_reduced, X_test_reduced: numpy arrays of reduced features
    """
    # Compute cluster centroids based on training data
    centroids = compute_cluster_centroids(X_train, labels, n_clusters)
    
    # Assign clusters to validation and test sets based on nearest centroid
    labels_val = assign_clusters(X_val, centroids, similarity_measure, p)
    labels_test = assign_clusters(X_test, centroids, similarity_measure, p)
    
    # One-Hot Encode cluster labels
    encoder = OneHotEncoder(categories='auto', sparse_output=False)
    
    # Fit encoder on training labels and transform all sets
    labels_train_reshaped = labels.reshape(-1, 1)
    X_train_reduced = encoder.fit_transform(labels_train_reshaped)
    
    labels_val_reshaped = labels_val.reshape(-1, 1)
    X_val_reduced = encoder.transform(labels_val_reshaped)
    
    labels_test_reshaped = labels_test.reshape(-1, 1)
    X_test_reduced = encoder.transform(labels_test_reshaped)
    
    return X_train_reduced, X_val_reduced, X_test_reduced

def main():
    # Step 1: Load dataset
    print("Loading Olivetti faces dataset...")
    X, y = load_olivetti_faces()
    X_flat = flatten_images(X)
    print(f"Dataset shape: {X_flat.shape}, Labels shape: {y.shape}\n")
    
    # Step 2: Split dataset
    print("Splitting dataset into training, validation, and test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X_flat, y)
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}\n")
    
    # Step 3: Scale the data
    print("Scaling the data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print(f"Scaled training set: {X_train_scaled.shape}")
    print(f"Scaled validation set: {X_val_scaled.shape}")
    print(f"Scaled test set: {X_test_scaled.shape}\n")
    
    # Step 4: Train and evaluate initial classifier
    print("Training and evaluating classifier with k-fold cross-validation...")
    avg_cv_acc, val_acc = train_evaluate_classifier(X_train, y_train, X_val, y_val)
    print(f"Average CV Accuracy: {avg_cv_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}\n")
    
    # Step 5: Dimensionality Reduction via Clustering
    similarity_measures = {
        'euclidean': {'metric': 'euclidean', 'p': None},
        'minkowski': {'metric': 'minkowski', 'p': 3},
        'cosine': {'metric': 'cosine', 'p': None}
    }
    
    reduced_datasets = {}
    cluster_info = {}
    
    print("Performing Agglomerative Hierarchical Clustering for dimensionality reduction...")
    for measure, params in similarity_measures.items():
        print(f"\nSimilarity Measure: {measure.capitalize()}")
        similarity_measure = params['metric']
        p = params['p']
        
        if measure == 'minkowski':
            X_train_used = X_train_scaled
        else:
            X_train_used = X_train_scaled  # Even for 'euclidean' and 'cosine', use scaled data
        
        best_k, labels, silhouette = perform_clustering(
            X_train_used, similarity_measure=measure, linkage='average', max_clusters=20, random_state=42
        )
        print(f"Optimal number of clusters: {best_k}")
        print(f"Silhouette Score: {silhouette[best_k]:.4f}")
        
        # Plot silhouette scores
        plot_silhouette_scores(silhouette, measure)
        
        # Reduce dimensionality by encoding cluster assignments
        X_train_reduced, X_val_reduced, X_test_reduced = reduce_dimensionality_via_clustering(
            X_train_scaled, X_val_scaled, X_test_scaled, labels, best_k, measure, p
        )
        
        reduced_datasets[measure] = {
            'train': X_train_reduced,
            'val': X_val_reduced,
            'test': X_test_reduced
        }
        
        cluster_info[measure] = best_k
    
    # Step 6: Discuss discrepancies
    print("\nDiscrepancies between similarity measures based on number of clusters chosen:")
    for measure, k in cluster_info.items():
        print(f"- {measure.capitalize()} Distance: {k} clusters")
    
    # Step 7: Train and evaluate classifiers on reduced datasets
    print("\nTraining and evaluating classifiers on reduced datasets...")
    for measure in similarity_measures:
        print(f"\nSimilarity Measure: {measure.capitalize()}")
        X_tr_reduced = reduced_datasets[measure]['train']
        X_va_reduced = reduced_datasets[measure]['val']
        
        # Debugging: Print shapes to ensure consistency
        print(f"Shape of X_train_reduced: {X_tr_reduced.shape}")
        print(f"Shape of y_train: {y_train.shape}")
        print(f"Shape of X_val_reduced: {X_va_reduced.shape}")
        print(f"Shape of y_val: {y_val.shape}")
        
        try:
            avg_cv_acc_red, val_acc_red = train_evaluate_classifier(
                X_tr_reduced, y_train, X_va_reduced, y_val
            )
            print(f"Average CV Accuracy: {avg_cv_acc_red:.4f}")
            print(f"Validation Accuracy: {val_acc_red:.4f}")
        except ValueError as ve:
            print(f"Error during training/evaluation: {ve}")
    
    print("\nProject Completed.")

if __name__ == "__main__":
    main()
