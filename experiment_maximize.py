from  product_methods import get_functions
from product_methods import kernel_matrix_classic

import warnings
warnings.filterwarnings("ignore")

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
noise_level = 0.01 # level of label noise
samplesize = 200
max_qubits = 7
full_train = np.zeros(max_qubits)
full_test = np.zeros(max_qubits)
bias_train = np.zeros(max_qubits)
bias_test = np.zeros(max_qubits)
gauss_train = np.zeros(max_qubits)
gauss_test = np.zeros(max_qubits)
bias_2_train = np.zeros(max_qubits)
bias_2_test = np.zeros(max_qubits)

runs = 1
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

        # fit mean seperately
        mean = np.mean(y_train)
        y_train = y_train - mean
        y_test = y_test - mean

        # full kernel
        # unfair optimization over all lambda to show that the behaviour is not due to poor choice of regularization
        full_test[i] = 1e6
        full_train[i] = 1e6
        for alpha in np.logspace(-6,4, num=15):
            krr_full = KernelRidge(alpha=alpha, kernel="precomputed")
            K_train_train_full = kernel_matrix_classic(X_train, X_train)
            krr_full.fit(K_train_train_full, y_train)
            # compute training loss
            y_train_pred = krr_full.predict(K_train_train_full)
            training_err = mean_squared_error(y_train_pred, y_train)
            # compute test loss
            y_full_pred = krr_full.predict(kernel_matrix_classic(X_test, X_train))
            test_err = mean_squared_error(y_test, y_full_pred)
            if test_err < full_test[i]:
                full_test[i] = test_err
                full_train[i] = training_err

        # biased kernel. RKHS only contains four functions (constant function + 3 equally weighted functions, therefore we dont regularize)
        krr_bias = KernelRidge(alpha=0., kernel="precomputed")
        K_train_train_bias = kernel_matrix_bias(X_train, X_train)
        krr_bias.fit(K_train_train_bias, y_train)
        # compute training loss
        y_train_pred = krr_bias.predict(K_train_train_bias)
        bias_train[i] = mean_squared_error(y_train_pred, y_train)
        # compute test loss
        y_bias_pred = krr_bias.predict(kernel_matrix_bias(X_test, X_train))
        bias_test[i] = mean_squared_error(y_test, y_bias_pred)

        # gaussian kernel as another reference.
        gauss_test[i] = 1e6
        gauss_train[i] = 1e6
        for alpha in np.logspace(-6, 4, num=15):
            krr_gauss = KernelRidge(alpha=alpha, kernel="rbf", gamma=1/2.)  # gamma chosen to match the choice in alignment exp
            krr_gauss.fit(X_train, y_train)
            # training error
            y_train_pred = krr_gauss.predict(X_train)
            training_err = mean_squared_error(y_train_pred, y_train)
            # test error
            y_gauss_pred = krr_gauss.predict(X_test)
            test_err = mean_squared_error(y_gauss_pred, y_test)
            if test_err < gauss_train[i]:
                gauss_test[i] = test_err
                gauss_train[i] = training_err

        # do with reduced denisity of second qubit
        krr_bias_2 = KernelRidge(alpha=0.0, kernel="precomputed")
        K_2_train_train = kernel_second_qubit(X_train, X_train)
        krr_bias_2.fit(K_2_train_train, y_train)
        # compute training loss
        y_train_pred = krr_bias_2.predict(K_2_train_train)
        bias_2_train[i] += mean_squared_error(y_train_pred, y_train) / runs
        # compute test loss
        y_bias_pred = krr_bias_2.predict(kernel_second_qubit(X_test, X_train))
        bias_2_test[i] += mean_squared_error(y_test, y_bias_pred) / runs


qubits = [i for i in range(1,max_qubits+1)]
np.save("data/loss_optimized_15_steps", (qubits, full_train, full_test, bias_train, bias_test,
                                         gauss_train, gauss_test, bias_2_train, bias_2_test))
# errors = [full_train, full_test, bias_train, bias_test, gauss_train, gauss_test]
# labels = ["full_train", "full_test", "bias_train", "bias_test", "rbf_train", "rbf_test"]
# styles = ["dashed", "solid", "dashed", "-", "dashed", "solid"]
# colors = ['red', 'red', 'blue', 'blue', "green", "green"]
# for i in range(6):
#     plt.plot(qubits, errors[i], label=labels[i], ls=styles[i], color=colors[i])
# plt.xlabel("Qubits")
# plt.ylabel("MSE")
# plt.yscale("log")
# plt.legend()
# plt.savefig("exp_optimized_gaussian_0.01.pdf")
