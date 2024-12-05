import matplotlib.pyplot as plt
import numpy as np
from load_data import get_data
from Codes import k_cross_validation
import os

# Load training and testing data
DATA_NAME = "concentric_circle"
traindata, trainlabel, testdata, testlabel = get_data(DATA_NAME)
FOLD_NO = 5

# Parameters for the circle map
INITIAL_NEURAL_ACTIVITY = [0.1988888888888889]  # Q
OMEGA = [0.1]  # Natural frequency of the circle map
K_VALUES = [0.5]  # Coupling strength
EPSILON = np.arange(0.001, 1.001, 0.001)

# Call k_cross_validation with Omega and K
FSCORE, Q, OMEGA, K, EPS, EPSILON = k_cross_validation(
    FOLD_NO, 
    traindata, 
    trainlabel, 
    testdata, 
    testlabel, 
    INITIAL_NEURAL_ACTIVITY, 
    OMEGA,  # Use Omega
    K_VALUES,  # Use K
    EPSILON, 
    DATA_NAME
)

# Plot results
PATH = os.getcwd()
RESULT_PATH = PATH + '/SR-PLOTS/' + DATA_NAME + '/NEUROCHAOS-RESULTS/'
plt.figure(figsize=(10, 10))
plt.plot(EPSILON, FSCORE[0, 0, :], '-*k', markersize=10)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.xlabel('Noise intensity', fontsize=20)
plt.ylabel('Average F1-score', fontsize=20)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(RESULT_PATH + "/Chaosnet-" + DATA_NAME + "-SR_plot.jpg", format='jpg', dpi=200)
plt.show()
