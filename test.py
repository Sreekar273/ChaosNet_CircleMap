import numpy as np
from load_data import get_data
from Codes import chaosnet
import os
import ChaosFEX.feature_extractor as CFX

import numpy as np
from load_data import get_data
from Codes import chaosnet
import ChaosFEX.feature_extractor as CFX  # Ensure feature_extractor is imported
import os

# Load the best parameters from the training phase
with open("./concentric_circle_max_best_params.txt", "r") as f:
    best_params = eval(f.read())  # Read and evaluate the dictionary

print("Best Parameters: ", best_params)

# Set the data name
DATA_NAME = "concentric_circle"

# Load training and testing data
traindata, trainlabel, testdata, testlabel = get_data(DATA_NAME)

# Extract the best parameters
INA = best_params['INITIAL_NEURAL_ACTIVITY']
OMEGA = best_params['OMEGA']
K = best_params['K']

# Prepare to run predictions
# First, transform both the training and test datasets using the best parameters

# Feature extraction for the training data
FEATURE_MATRIX_TRAIN = CFX.transform(traindata, INA, OMEGA, K, 10000, 0.01, 0)  # Use best INA, OMEGA, and K
# Feature extraction for the test data
FEATURE_MATRIX_TEST = CFX.transform(testdata, INA, OMEGA, K, 10000, 0.01, 0)  # Use best INA, OMEGA, and K

# Run predictions on the test set
mean_each_class, Y_PRED = chaosnet(FEATURE_MATRIX_TRAIN, trainlabel, FEATURE_MATRIX_TEST, OMEGA, K)  # No need to pass OMEGA and K here

# Evaluate the predictions
from sklearn.metrics import accuracy_score, f1_score
accuracy = accuracy_score(testlabel, Y_PRED)
f1 = f1_score(testlabel, Y_PRED, average='macro')

# Output the evaluation metrics
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test F1 Score: {f1:.2f}")
