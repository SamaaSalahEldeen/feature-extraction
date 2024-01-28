# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:28:36 2023

@author: DELL
"""

import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score 
import joblib

import warnings

# Suppress warnings related to iCCP and sRGB profiles
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def load_images_from_folder(folder):
    images = []
    labels = []
    valid_image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

    for product_folder in os.listdir(folder):
        product_path = os.path.join(folder, product_folder)
        if os.path.isdir(product_path):
            # Concatenate both training and validation sets
            for subset in ['Train', 'Validation']:
                subset_path = os.path.join(product_path, subset)
                if os.path.isdir(subset_path):
                    for filename in os.listdir(subset_path):
                        img_path = os.path.join(subset_path, filename)

                        # Skip non-image files and files with .db extension
                        if not img_path.lower().endswith(valid_image_extensions) or img_path.lower().endswith('.db'):
                            continue

                        img = cv2.imread(img_path)
                        if img is not None:
                            images.append(img)
                            labels.append(product_folder)

    return images, labels

# Extract features using SIFT
def sift_extraction(folder):
    sift = cv2.SIFT_create()
    descriptors_list = []

    for product_folder in os.listdir(folder):
        product_path = os.path.join(folder, product_folder)
        if os.path.isdir(product_path):
            print(f"Processing product: {product_folder}")

            # Concatenate both training and validation sets
            for subset in ['Train', 'Validation']:
                subset_path = os.path.join(product_path, subset)
                if os.path.isdir(subset_path):
                    print(f"Processing subset: {subset}")

                    for image_file in os.listdir(subset_path):
                        image_path = os.path.join(subset_path, image_file)
                        print(f"Processing image: {image_path}")

                        # Read the image
                        img = cv2.imread(image_path)

                        # Check if the image is loaded
                        if img is None:
                            print(f"Error: Unable to load {image_path}")
                            continue

                        # Check the image depth
                        if img.dtype != 'uint8':
                            print(f"Error: Incorrect depth for {image_path}")
                            continue

                        # Extract SIFT features
                        _, des = sift.detectAndCompute(img, None)

                        # If features are extracted, add to the list
                        if des is not None and len(des) > 0:
                            descriptors_list.append(des)
                        else:
                            print(f"Warning: No SIFT features extracted for {image_path}")

    return descriptors_list

# Cluster using KMeans
def clusters(descriptors_list, k):
    all_descriptors = np.vstack(descriptors_list)
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(all_descriptors)
    visual_words = kmeans.cluster_centers_

    return visual_words

# Create visual word descriptor
def visual_word_descriptor(descriptor_list, visual_words):
    assignments_list = []
    for descriptors in descriptor_list:
        distance = np.linalg.norm(descriptors[:, None] - visual_words, axis=2)
        assignment = np.argmin(distance, axis=1)
        assignments_list.append(assignment)

    return assignments_list

# Generate histogram
def generate_histogram(assignments_list, k):
    histograms_list = []
    for assignment in assignments_list:
        histo, _ = np.histogram(assignment, bins=np.arange(k + 1), density=True)
        histograms_list.append(histo)

    return np.array(histograms_list)


# Define path to the dataset folder
dataset_folder = r"D:\material\CV\project\Data\project data\Product Classification"

# Load images and labels
training_images, training_labels = load_images_from_folder(dataset_folder)
validation_images, validation_labels = load_images_from_folder(dataset_folder)

# Extract SIFT features
training_descriptor_list = sift_extraction(dataset_folder)
validation_descriptor_list = sift_extraction(dataset_folder)

# Cluster using KMeans
visual_words = clusters(training_descriptor_list, k=500)

# Create visual word descriptors
training_assignments_list = visual_word_descriptor(training_descriptor_list, visual_words)
validation_assignments_list = visual_word_descriptor(validation_descriptor_list, visual_words)

# Generate histograms
training_histograms = generate_histogram(training_assignments_list, k=500)
validation_histograms = generate_histogram(validation_assignments_list, k=500)

# Create an SVM classifier pipeline
svm_classifier = make_pipeline(StandardScaler(), SVC(C=1, kernel='sigmoid', gamma='scale'))





# Perform cross-validation for validation accuracy
cv_scores = cross_val_score(svm_classifier, validation_histograms, validation_labels, cv=5, scoring='accuracy')

# Print cross-validation scores for validation accuracy
print("Cross-Validation Scores for Validation Accuracy:", cv_scores)

# Fit the model on the entire training set
svm_classifier.fit(training_histograms, training_labels)

# Evaluate on training set
training_accuracy = accuracy_score(training_labels, svm_classifier.predict(training_histograms))
print(f"Training Accuracy: {training_accuracy}")

# Evaluate on validation set
validation_accuracy = np.mean(cv_scores)
print(f"Validation Accuracy: {validation_accuracy}")

# Print classification report
print("Training Classification Report:")
print(classification_report(training_labels, svm_classifier.predict(training_histograms)))

print("Validation Classification Report:")
print(classification_report(validation_labels, svm_classifier.predict(validation_histograms)))
from sklearn.model_selection import GridSearchCV

# Define the hyperparameter search space
param_grid = {'svc__C': [0.1, 1, 10, 100, 1000], 'svc__kernel': ['linear', 'rbf', 'sigmoid']}

# Create the SVM classifier pipeline
svm_classifier = make_pipeline(StandardScaler(), SVC(gamma='scale'))
# Save the trained SVM classifier to a file
model_filename = "svm_classifier_model.joblib"
joblib.dump(svm_classifier, model_filename)
print(f"Trained SVM classifier saved to {model_filename}")

# Initialize GridSearchCV
grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(training_histograms, training_labels)

# Print the best parameters from GridSearchCV
print("Best Parameters (GridSearchCV):", grid_search.best_params_)

# Perform cross-validation for validation accuracy using the best parameters
cv_scores_grid = cross_val_score(grid_search.best_estimator_, validation_histograms, validation_labels, cv=5, scoring='accuracy')

# Print cross-validation scores for validation accuracy
print("Cross-Validation Scores for Validation Accuracy (GridSearchCV):", cv_scores_grid)

# Fit the model on the entire training set using the best parameters from GridSearchCV
grid_search.best_estimator_.fit(training_histograms, training_labels)

# Evaluate on the training set using the best parameters from GridSearchCV
training_accuracy_grid = accuracy_score(training_labels, grid_search.best_estimator_.predict(training_histograms))
print(f"Training Accuracy (GridSearchCV): {training_accuracy_grid}")

# Evaluate on the validation set using the best parameters from GridSearchCV
validation_accuracy_grid = np.mean(cv_scores_grid)
print(f"Validation Accuracy (GridSearchCV): {validation_accuracy_grid}")

# Print the classification report for the training set using the best parameters from GridSearchCV
print("Training Classification Report (GridSearchCV):")
print(classification_report(training_labels, grid_search.best_estimator_.predict(training_histograms)))

# Print the classification report for the validation set using the best parameters from GridSearchCV
print("Validation Classification Report (GridSearchCV):")
print(classification_report(validation_labels, grid_search.best_estimator_.predict(validation_histograms)))
import time

# Measure training time
start_time = time.time()
svm_classifier.fit(training_histograms, training_labels)
training_time = time.time() - start_time
print(f"Training Time OF SVMS: {training_time:.2f} seconds")

# Measure testing time
start_time = time.time()
validation_predictions = svm_classifier.predict(validation_histograms)
testing_time = time.time() - start_time
print(f"Testing Time OF SVM : {testing_time:.2f} seconds")
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix of SVM')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Use the trained model to make predictions on the validation set
validation_predictions = grid_search.best_estimator_.predict(validation_histograms)

# Get unique class labels
class_labels = np.unique(np.concatenate([training_labels, validation_labels]))

# Plot the confusion matrix
plot_confusion_matrix(validation_labels, validation_predictions, class_labels)


