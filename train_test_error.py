### Experiments to reproduce right panel of Figure 2.
###
from quantum_methods import get_functions
from quantum_methods import kernel_matrix_classic

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")  # to ignore sparse matrix warning in Kernel Ridge Regression

noise_level = 0.01  # level of label noise
samplesize = 200
max_qubits = 7
runs = 10

full_train = np.zeros(max_qubits)
full_test = np.zeros(max_qubits)
bias_train = np.zeros(max_qubits)
bias_test = np.zeros(max_qubits)
gauss_train = np.zeros(max_qubits)
gauss_test = np.zeros(max_qubits)
bias_2_train = np.zeros(max_qubits)
bias_2_test = np.zeros(max_qubits)

for seed in tqdm(range(runs)):
    np.random.seed(seed)
    for i in range(max_qubits):
        qubits = i+1
        _, _, f, kernel_matrix_bias, kernel_second_qubit = get_functions(qubits=qubits, seed=seed, second_qubit=True)

        # generate data
        X = np.random.uniform([0]*qubits, [2 * np.pi]*qubits, size=(samplesize, qubits))
        noise = np.random.normal(0, noise_level, samplesize)
        f_X = np.array([f(X[i]) for i in range(samplesize)])
        f_X = f_X / np.sqrt(np.var(f_X))
        y = f_X + noise
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)

        # fit mean separately
        mean = np.mean(y_train)
        y_train = y_train - mean
        y_test = y_test - mean

        krr_full = KernelRidge(alpha=0.001, kernel="precomputed")
        K_train_train_full = kernel_matrix_classic(X_train, X_train)
        krr_full.fit(K_train_train_full, y_train)
        # compute training loss
        y_train_pred = krr_full.predict(K_train_train_full)
        full_train[i] += mean_squared_error(y_train_pred, y_train) / runs
        # compute test loss
        y_full_pred = krr_full.predict(kernel_matrix_classic(X_test, X_train))
        full_test[i] += mean_squared_error(y_test, y_full_pred) / runs

        krr_bias = KernelRidge(alpha=0., kernel="precomputed")
        K_train_train_bias = kernel_matrix_bias(X_train, X_train)
        krr_bias.fit(K_train_train_bias, y_train)
        # compute training loss
        y_train_pred = krr_bias.predict(K_train_train_bias)
        bias_train[i] += mean_squared_error(y_train_pred, y_train) / runs
        # compute test loss
        y_bias_pred = krr_bias.predict(kernel_matrix_bias(X_test, X_train))
        bias_test[i] += mean_squared_error(y_test, y_bias_pred) / runs

        krr_gauss = KernelRidge(alpha=0.001, kernel="rbf", gamma=1/2.)
        krr_gauss.fit(X_train, y_train)
        # training error
        y_train_pred = krr_gauss.predict(X_train)
        gauss_train[i] += mean_squared_error(y_train_pred, y_train) / runs
        # test error
        y_gauss_pred = krr_gauss.predict(X_test)
        gauss_test[i] += mean_squared_error(y_gauss_pred, y_test) / runs

        # do with reduced density of second qubit
        krr_bias_2 = KernelRidge(alpha=0., kernel="precomputed")
        K_2_train_train = kernel_second_qubit(X_train, X_train)
        krr_bias_2.fit(K_2_train_train, y_train)
        # compute training loss
        y_train_pred = krr_bias_2.predict(K_2_train_train)
        bias_2_train[i] += mean_squared_error(y_train_pred, y_train) / runs
        # compute test loss
        y_bias_pred = krr_bias_2.predict(kernel_second_qubit(X_test, X_train))
        bias_2_test[i] += mean_squared_error(y_test, y_bias_pred) / runs


qubits = [i for i in range(1, max_qubits+1)]
np.save("data/loss_10_runs_alpha_e-3", (qubits, full_train, full_test, bias_train, bias_test,
                                        gauss_train, gauss_test, bias_2_train, bias_2_test))
