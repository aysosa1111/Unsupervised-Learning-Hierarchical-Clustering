# Olivetti Faces Dataset Analysis

This project explores the Olivetti faces dataset to perform facial recognition using support vector machines (SVMs) and hierarchical clustering. The goal is to understand the impact of dimensionality reduction via clustering on SVM classification performance.

## Project Setup

### Dependencies
To run this project, ensure the following Python libraries are installed:
- NumPy
- pandas
- scikit-learn
- matplotlib
- scipy

You can install these packages using pip:
```bash
pip install numpy pandas scikit-learn matplotlib scipy
```

# Olivetti Faces Dataset Analysis Project

## Dataset
The dataset used is the Olivetti faces dataset, which can be directly fetched from the `sklearn.datasets` module using the `fetch_olivetti_faces()` function.

## Usage
To execute the main program, navigate to the project directory and run:
```bash
python main.py
```

Project Structure
Code Overview
Loading the Dataset:

The Olivetti faces dataset is loaded and its structure is described.
Data Splitting:

The dataset is divided into training, validation, and test sets using stratified sampling to ensure even representation across all sets.
Data Scaling:

Data is scaled using StandardScaler to normalize feature scales, preparing it for effective machine learning processing.
Classifier Training and Evaluation:

An SVM classifier is trained using k-fold cross-validation and evaluated on the validation set.
Dimensionality Reduction via Clustering:

Agglomerative Hierarchical Clustering is applied using Euclidean, Minkowski, and Cosine distances. The optimal number of clusters is determined by silhouette scores.
Training on Reduced Datasets:

Classifiers are re-trained on datasets reduced via clustering to evaluate the impact of dimensionality reduction on classifier performance.
Functions
load_olivetti_faces(): Fetches and loads the Olivetti faces dataset.
split_dataset(): Splits the dataset into training, validation, and test sets.
flatten_images(): Flattens the image data from 2D to 1D per sample.
train_evaluate_classifier(): Trains the SVM classifier and evaluates it.
perform_clustering(): Applies hierarchical clustering on the dataset.
reduce_dimensionality_via_clustering(): Reduces dataset dimensionality by encoding cluster labels.
main(): Orchestrates the loading, processing, clustering, and classification tasks.
Conclusions
This project illustrates the use of SVMs and hierarchical clustering in facial recognition. While SVMs demonstrated high accuracy on original data, the dimensionality reduction via clustering significantly impacted classification performance, emphasizing the need for careful consideration of the methods used in reducing data complexity for image recognition tasks.
