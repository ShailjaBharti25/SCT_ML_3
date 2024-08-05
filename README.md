# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the dataset
train_dir = 'path_to_train_directory'
test_dir = 'path_to_test_directory'

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(224, 224),
                                                  batch_size=32,
                                                  class_mode='binary')

# Extract features from the images
train_features = []
train_labels = []
for batch in train_generator:
    features, labels = batch
    features = features.reshape(-1, 224*224*3)
    train_features.extend(features)
    train_labels.extend(labels)

test_features = []
test_labels = []
for batch in test_generator:
    features, labels = batch
    features = features.reshape(-1, 224*224*3)
    test_features.extend(features)
    test_labels.extend(labels)

# Convert lists to numpy arrays
train_features = np.array(train_features)
train_labels = np.array(train_labels)
test_features = np.array(test_features)
test_labels = np.array(test_labels)

# Scale the features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Train the SVM model
svm_model = svm.SVC(kernel='rbf', C=1)
svm_model.fit(train_features, train_labels)

# Make predictions
predictions = svm_model.predict(test_features)

# Evaluate the model
accuracy = accuracy_score(test_labels, predictions)
print('Accuracy:', accuracy)
print('Classification Report:')
print(classification_report(test_labels, predictions))
print('Confusion Matrix:')
print(confusion_matrix(test_labels, predictions))
