import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from load_data import get_data
from Codes import k_cross_validation

# Load training and testing data
DATA_NAME = "single_variable_classification"
traindata, trainlabel, testdata, testlabel = get_data(DATA_NAME)

# Parameters for ChaosNet
FOLD_NO = 5
INITIAL_NEURAL_ACTIVITY = [0.1]  # Q
OMEGA = [0.1]  # Natural frequency of the circle map
K_VALUES = [0.5]  # Coupling strength
EPSILON = np.arange(0.001, 1.001, 0.01)

# Initialize arrays to store F1 scores for traditional models
FSCORE_XGB = np.zeros(len(EPSILON))
FSCORE_SVM = np.zeros(len(EPSILON))
FSCORE_MLP = np.zeros(len(EPSILON))

print("Starting model training and evaluation...")

# Iterate through all epsilon values
for i, epsilon in enumerate(tqdm(EPSILON, desc="Epsilon Loop")):
    # Adding Gaussian noise to the data
    noise_train = np.random.normal(0, epsilon, traindata.shape)
    noise_test = np.random.normal(0, epsilon, testdata.shape)
    X_train_noisy = traindata + noise_train
    X_test_noisy = testdata + noise_test

    # Standardize the noisy data
    scaler = StandardScaler()
    X_train_noisy = scaler.fit_transform(X_train_noisy)
    X_test_noisy = scaler.transform(X_test_noisy)

    # **XGBoost**
    xgb_model = XGBClassifier(eval_metric="logloss")
    xgb_model.fit(X_train_noisy, trainlabel.ravel())
    y_pred_xgb = xgb_model.predict(X_test_noisy)
    FSCORE_XGB[i] = f1_score(testlabel, y_pred_xgb)
    print(f"XGBoost: Epsilon={epsilon:.4f}, F1-Score={FSCORE_XGB[i]:.4f}")

    # **SVM**
    svm_model = SVC(kernel='rbf', probability=True)
    svm_model.fit(X_train_noisy, trainlabel.ravel())
    y_pred_svm = svm_model.predict(X_test_noisy)
    FSCORE_SVM[i] = f1_score(testlabel, y_pred_svm)
    print(f"SVM: Epsilon={epsilon:.4f}, F1-Score={FSCORE_SVM[i]:.4f}")

    # **MLP**
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    mlp_model.fit(X_train_noisy, trainlabel.ravel())
    y_pred_mlp = mlp_model.predict(X_test_noisy)
    FSCORE_MLP[i] = f1_score(testlabel, y_pred_mlp)
    print(f"MLP: Epsilon={epsilon:.4f}, F1-Score={FSCORE_MLP[i]:.4f}")

# **ChaosNet**
FSCORE_CHAOSNET, _, _, _, _, _ = k_cross_validation(
    FOLD_NO,
    traindata,
    trainlabel,
    testdata,
    testlabel,
    INITIAL_NEURAL_ACTIVITY,
    OMEGA,
    K_VALUES,
    EPSILON,
    DATA_NAME,
)

# Plot results
PATH = os.getcwd()
RESULT_PATH = PATH + '/SR-PLOTS/' + DATA_NAME + '/ALL-RESULTS/'
os.makedirs(RESULT_PATH, exist_ok=True)

plt.figure(figsize=(10, 10))
plt.plot(EPSILON, FSCORE_XGB, '-*k', markersize=10, label="XGBoost")
plt.plot(EPSILON, FSCORE_SVM, '-^b', markersize=10, label="SVM")
plt.plot(EPSILON, FSCORE_MLP, '-or', markersize=10, label="MLP")
plt.plot(EPSILON, FSCORE_CHAOSNET[0, 0, :], '-sg', markersize=10, label="ChaosNet")  # ChaosNet results
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.xlabel('Noise intensity', fontsize=20)
plt.ylabel('Average F1-score', fontsize=20)
plt.ylim(0, 1)
plt.legend(fontsize=20)
plt.tight_layout()

# Save plot
plt.savefig(RESULT_PATH + f"Chaosnet-{DATA_NAME}-SR_plot.jpg", format='jpg', dpi=200)
plt.show()

print("Process completed. Results saved.")
