import numpy as np
from load_data import get_data
from Codes import k_cross_validation
import os
from itertools import product

# Set the data name
DATA_NAME = "concentric_circle_noise"

# Load training and testing data
traindata, trainlabel, testdata, testlabel = get_data(DATA_NAME)

# Parameters for tuning - Define ranges for Omega, K, and Q (INITIAL_NEURAL_ACTIVITY)
OMEGA_RANGE = np.linspace(0.1, 0.99, 10)  # Omega values between 0.1 and 1.0, 10 steps
K_RANGE = np.linspace(0.5, 5.0, 10)  # K values between 0.5 and 5.0, 10 steps
INITIAL_NEURAL_ACTIVITY_RANGE = np.linspace(0.1, 0.99, 10)  # Q values between 0.1 and 1.0, 10 steps

# Noise intensity (fixed range)
EPSILON = np.arange(0.001, 1.001, 0.01)

# Number of folds for cross-validation
FOLD_NO = 5

# Initialize variables to store the best results
best_fscore = 0
best_params = {}

# Grid search over Omega, K, and INITIAL_NEURAL_ACTIVITY
for INA, Omega, K in product(INITIAL_NEURAL_ACTIVITY_RANGE, OMEGA_RANGE, K_RANGE):
    # Perform cross-validation for the current combination of parameters
    print(f"INA, Omega, K: ", INA, Omega, K)
    FSCORE, Q, OMEGA_MATRIX, K_MATRIX, EPS, _ = k_cross_validation(
        FOLD_NO,
        traindata,
        trainlabel,
        testdata,
        testlabel,
        [INA],  # Single value for current INA
        [Omega],  # Single value for current Omega
        [K],  # Single value for current K
        EPSILON,
        DATA_NAME
    )

    # Find the best F1-score for the current parameter combination
    best_fscore_for_params = np.max(FSCORE)  # Use max F1-score across folds and noise levels

    # Update the overall best parameters if this combination is better
    if best_fscore_for_params > best_fscore:
        best_fscore = best_fscore_for_params
        best_params = {
            'INITIAL_NEURAL_ACTIVITY': INA,
            'OMEGA': Omega,
            'K': K,
            'F1_SCORE': best_fscore
        }

    print(f"Tested INA={INA}, Omega={Omega}, K={K} -> F1-Score: {best_fscore_for_params:.4f}")
    # I only need these values for the best parameters. So store these in an array and then save it to a file

# Save the best parameters to a file
with open(f"{DATA_NAME}_best_params2.txt", "w") as f:
    f.write(str(best_params))

print("Best parameters found:")
print(best_params)
