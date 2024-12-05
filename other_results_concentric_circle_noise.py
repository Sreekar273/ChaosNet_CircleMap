import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from load_data import get_data
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Load training and testing data
DATA_NAME = "concentric_circle_noise"
traindata, trainlabel, testdata, testlabel = get_data(DATA_NAME)

# Define epsilon range
EPSILON = np.arange(0.001, 1.001, 0.01)

# Initialize arrays to store results
FSCORE_XGB = np.zeros(len(EPSILON))
FSCORE_SVM = np.zeros(len(EPSILON))
FSCORE_MLP = np.zeros(len(EPSILON))

# Add noise to the training and testing data based on epsilon values
print("Starting model training and evaluation...")
for i, epsilon in enumerate(tqdm(EPSILON, desc="Epsilon Loop")):
    # Adding Gaussian noise to the data based on epsilon
    noise_train = np.random.normal(0, epsilon, traindata.shape)
    noise_test = np.random.normal(0, epsilon, testdata.shape)
    
    # Apply noise to the data
    X_train_noisy = traindata + noise_train
    X_test_noisy = testdata + noise_test

    # Standardize the data
    scaler = StandardScaler()
    X_train_noisy = scaler.fit_transform(X_train_noisy)
    X_test_noisy = scaler.transform(X_test_noisy)

    # Initialize models
    xgb_model = XGBClassifier(eval_metric="logloss")
    svm_model = SVC(kernel='rbf', probability=True)
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)

    # Train and evaluate XGBoost
    xgb_model.fit(X_train_noisy, trainlabel.ravel())
    y_pred_xgb = xgb_model.predict(X_test_noisy)
    FSCORE_XGB[i] = f1_score(testlabel, y_pred_xgb)
    print(f"XGBoost: Epsilon={epsilon:.4f}, F1-Score={FSCORE_XGB[i]:.4f}")

    # Train and evaluate SVM
    svm_model.fit(X_train_noisy, trainlabel.ravel())
    y_pred_svm = svm_model.predict(X_test_noisy)
    FSCORE_SVM[i] = f1_score(testlabel, y_pred_svm)
    print(f"SVM: Epsilon={epsilon:.4f}, F1-Score={FSCORE_SVM[i]:.4f}")

    # Train and evaluate MLP
    mlp_model.fit(X_train_noisy, trainlabel.ravel())
    y_pred_mlp = mlp_model.predict(X_test_noisy)
    FSCORE_MLP[i] = f1_score(testlabel, y_pred_mlp)
    print(f"MLP: Epsilon={epsilon:.4f}, F1-Score={FSCORE_MLP[i]:.4f}")

# Plot results
PATH = os.getcwd()
RESULT_PATH = PATH + '/SR-PLOTS/' + DATA_NAME + '/OTHER-RESULTS/'
os.makedirs(RESULT_PATH, exist_ok=True)

plt.figure(figsize=(10, 10))
plt.plot(EPSILON, FSCORE_XGB, '-*k', markersize=10, label="XGBoost")
plt.plot(EPSILON, FSCORE_SVM, '-^b', markersize=10, label="SVM")
plt.plot(EPSILON, FSCORE_MLP, '-or', markersize=10, label="MLP")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.xlabel('Noise intensity', fontsize=20)
plt.ylabel('Average F1-score', fontsize=20)
plt.ylim(0, 1)
plt.legend(fontsize=20)
plt.tight_layout()

# Save plot
plt.savefig(RESULT_PATH + f"Chaosnet-{DATA_NAME}-SR_plot2.jpg", format='jpg', dpi=200)
plt.show()

print("Process completed. Results saved.")
