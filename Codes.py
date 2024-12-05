# -*- coding: utf-8 -*-
"""
Author: Harikrishnan NB
Dtd: 22 Dec. 2020
ChaosNet decision function
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix as cm
import os
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.svm import LinearSVC

import ChaosFEX.feature_extractor as CFX


def chaosnet(traindata, trainlabel, testdata, Omega, K):
    """
    Parameters
    ----------
    traindata : TYPE - Numpy 2D array
        DESCRIPTION - traindata
    trainlabel : TYPE - Numpy 2D array
        DESCRIPTION - Trainlabel
    testdata : TYPE - Numpy 2D array
        DESCRIPTION - testdata
    Omega : scalar, float64
        Natural frequency of the oscillator.
    K : scalar, float64
        Coupling strength.

    Returns
    -------
    mean_each_class : Numpy 2D array
        DESCRIPTION - mean representation vector of each class
    predicted_label : TYPE - numpy 1D array
        DESCRIPTION - predicted label
    """
    from sklearn.metrics.pairwise import cosine_similarity
    NUM_FEATURES = traindata.shape[1]
    NUM_CLASSES = len(np.unique(trainlabel))
    mean_each_class = np.zeros((NUM_CLASSES, NUM_FEATURES))
    
    for label in range(0, NUM_CLASSES):
        mean_each_class[label, :] = np.mean(traindata[(trainlabel == label)[:, 0], :], axis=0)
    
    predicted_label = np.argmax(cosine_similarity(testdata, mean_each_class), axis=1)
    return mean_each_class, predicted_label




def k_cross_validation(FOLD_NO, traindata, trainlabel, testdata, testlabel, INITIAL_NEURAL_ACTIVITY, Omega, K_VALUES, EPSILON, DATA_NAME):
    """
    Cross-validation function for circle map dynamics.

    Parameters
    ----------
    FOLD_NO : Integer
        Number of folds for cross-validation.
    traindata : Numpy array
    trainlabel : Numpy array
    testdata : Numpy array
    testlabel : Numpy array
    INITIAL_NEURAL_ACTIVITY : Numpy 1D array
        Initial value for the chaotic circle map (Q).
    Omega : Numpy 1D array
        Natural frequency of the circle map.
    K_VALUES : Numpy 1D array
        Coupling strength of the circle map.
    EPSILON : Numpy 1D array
        Noise intensity for chaotic map.
    DATA_NAME : String
        Name of the dataset.
    """
    ACCURACY = np.zeros((len(K_VALUES), len(INITIAL_NEURAL_ACTIVITY), len(EPSILON)))
    FSCORE = np.zeros((len(K_VALUES), len(INITIAL_NEURAL_ACTIVITY), len(EPSILON)))
    Q = np.zeros((len(K_VALUES), len(INITIAL_NEURAL_ACTIVITY), len(EPSILON)))
    OMEGA_MATRIX = np.zeros((len(K_VALUES), len(INITIAL_NEURAL_ACTIVITY), len(EPSILON)))
    K_MATRIX = np.zeros((len(K_VALUES), len(INITIAL_NEURAL_ACTIVITY), len(EPSILON)))
    EPS = np.zeros((len(K_VALUES), len(INITIAL_NEURAL_ACTIVITY), len(EPSILON)))

    KF = KFold(n_splits=FOLD_NO, random_state=42, shuffle=True)  # Define the split
    print(KF)
    
    # ROW = -1
    # for DT in DISCRIMINATION_THRESHOLD:
    #     ROW += 1
    #     COL = -1
    #     for INA in INITIAL_NEURAL_ACTIVITY:
    #         COL += 1
    #         WIDTH = -1
    #         for EPSILON_1 in EPSILON:
    #             WIDTH += 1
                
    #             ACC_TEMP = []
    #             FSCORE_TEMP = []
            
    #             for TRAIN_INDEX, VAL_INDEX in KF.split(traindata):
    #                 X_TRAIN, X_VAL = traindata[TRAIN_INDEX], traindata[VAL_INDEX]
    #                 Y_TRAIN, Y_VAL = trainlabel[TRAIN_INDEX], trainlabel[VAL_INDEX]
                    
    #                 # Extract features using the updated function
    #                 FEATURE_MATRIX_TRAIN = CFX.transform(X_TRAIN, INA, Omega, K, 10000, EPSILON_1, DT)
    #                 FEATURE_MATRIX_VAL = CFX.transform(X_VAL, INA, Omega, K, 10000, EPSILON_1, DT) 
                
    #                 mean_each_class, Y_PRED = chaosnet(FEATURE_MATRIX_TRAIN, Y_TRAIN, FEATURE_MATRIX_VAL, Omega, K)
                    
    #                 ACC = accuracy_score(Y_VAL, Y_PRED) * 100
    #                 RECALL = recall_score(Y_VAL, Y_PRED, average="macro")
    #                 PRECISION = precision_score(Y_VAL, Y_PRED, average="macro")
    #                 F1SCORE = f1_score(Y_VAL, Y_PRED, average="macro")
                    
    #                 ACC_TEMP.append(ACC)
    #                 FSCORE_TEMP.append(F1SCORE)
                    
    #             Q[ROW, COL, WIDTH] = INA
    #             B[ROW, COL, WIDTH] = DT
    #             EPS[ROW, COL, WIDTH] = EPSILON_1
    #             ACCURACY[ROW, COL, WIDTH] = np.mean(ACC_TEMP)
    #             FSCORE[ROW, COL, WIDTH] = np.mean(FSCORE_TEMP)
    #             print("Mean F1-Score for Q =", Q[ROW, COL, WIDTH], "B =", B[ROW, COL, WIDTH], "EPSILON =", EPS[ROW, COL, WIDTH], "is =", np.mean(FSCORE_TEMP))

    # Array to store logs
    log_messages = []

    for idx_k, K in enumerate(K_VALUES):
        for idx_q, INA in enumerate(INITIAL_NEURAL_ACTIVITY):
            for idx_eps, EPSILON_1 in enumerate(EPSILON):
                ACC_TEMP = []
                FSCORE_TEMP = []

                for TRAIN_INDEX, VAL_INDEX in KF.split(traindata):
                    X_TRAIN, X_VAL = traindata[TRAIN_INDEX], traindata[VAL_INDEX]
                    Y_TRAIN, Y_VAL = trainlabel[TRAIN_INDEX], trainlabel[VAL_INDEX]

                    # Extract features with Omega and K
                    FEATURE_MATRIX_TRAIN = CFX.transform(X_TRAIN, INA, Omega[0], K, 10000, EPSILON_1, 0)
                    FEATURE_MATRIX_VAL = CFX.transform(X_VAL, INA, Omega[0], K, 10000, EPSILON_1, 0)

                    mean_each_class, Y_PRED = chaosnet(FEATURE_MATRIX_TRAIN, Y_TRAIN, FEATURE_MATRIX_VAL, Omega[0], K)

                    ACC = accuracy_score(Y_VAL, Y_PRED) * 100
                    F1SCORE = f1_score(Y_VAL, Y_PRED, average="macro")

                    ACC_TEMP.append(ACC)
                    FSCORE_TEMP.append(F1SCORE)

                Q[idx_k, idx_q, idx_eps] = INA
                OMEGA_MATRIX[idx_k, idx_q, idx_eps] = Omega[0]
                K_MATRIX[idx_k, idx_q, idx_eps] = K
                ACCURACY[idx_k, idx_q, idx_eps] = np.mean(ACC_TEMP)
                FSCORE[idx_k, idx_q, idx_eps] = np.mean(FSCORE_TEMP)

                # Log message
                mean_f1_score = np.mean(FSCORE_TEMP)
                log_message = f"Mean F1-Score for Q = {INA}, Omega = {Omega[0]}, K = {K}, EPSILON = {EPSILON_1} is = {mean_f1_score:.4f}"
                log_messages.append(log_message)
                print(log_message)

    # Save all logs to a text file at the end
    log_file_path = f"{DATA_NAME}_results.log"
    with open(log_file_path, "w") as log_file:
        log_file.write("\n".join(log_messages))

    print("Logs have been saved to", log_file_path)

    # Saving Hyperparameter Tuning Results
    print("Saving Hyperparameter Tuning Results")
    PATH = os.getcwd()
    RESULT_PATH = PATH + '/SR-PLOTS/' + DATA_NAME + '/NEUROCHAOS-RESULTS/'

    try:
        os.makedirs(RESULT_PATH)
    except OSError:
        print("Creation of the result directory %s failed" % RESULT_PATH)
    else:
        print("Successfully created the result directory %s" % RESULT_PATH)

    # Save the results
    np.save(RESULT_PATH + "/h_fscore.npy", FSCORE)
    np.save(RESULT_PATH + "/h_accuracy.npy", ACCURACY)
    np.save(RESULT_PATH + "/h_Q.npy", Q)
    np.save(RESULT_PATH + "/h_OMEGA.npy", OMEGA_MATRIX)  # Save Omega results
    np.save(RESULT_PATH + "/h_K.npy", K_MATRIX)  # Save K results
    np.save(RESULT_PATH + "/h_EPS.npy", EPS)

    # Finding the best results
    MAX_FSCORE = np.max(FSCORE)
    Q_MAX = []
    OMEGA_MAX = []
    K_MAX = []
    EPSILON_MAX = []

    for ROW in range(len(INITIAL_NEURAL_ACTIVITY)):
        for COL in range(len(K_VALUES)):
            for WID in range(len(EPSILON)):
                if FSCORE[COL, ROW, WID] == MAX_FSCORE:
                    Q_MAX.append(Q[COL, ROW, WID])
                    OMEGA_MAX.append(OMEGA_MATRIX[COL, ROW, WID])
                    K_MAX.append(K_MATRIX[COL, ROW, WID])
                    EPSILON_MAX.append(EPS[COL, ROW, WID])

    print("BEST F1SCORE", MAX_FSCORE)
    print("BEST INITIAL NEURAL ACTIVITY =", Q_MAX)
    print("BEST OMEGA =", OMEGA_MAX)
    print("BEST K =", K_MAX)
    print("BEST EPSILON =", EPSILON_MAX)

    return FSCORE, Q, OMEGA_MATRIX, K_MATRIX, EPS, EPSILON


    

