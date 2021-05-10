from  product_methods import get_functions
from product_methods import kernel_matrix_classic

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
seed = 1
noise_level = 0.1 # level of label noise
np.random.seed(seed)
samplesize = 100
max_qubits = 8
full_train = np.zeros(max_qubits)
full_test = np.zeros(max_qubits)
bias_train = np.zeros(max_qubits)
bias_test = np.zeros(max_qubits)

for i in range(max_qubits):
    qubits = i+1
    _, _, f, kernel_matrix_bias = get_functions(qubits=qubits, seed=seed)

    # generate data
    X = np.random.uniform([0]*qubits, [2 * np.pi]*qubits, size=(2*samplesize, qubits))
    noise = np.random.normal(0, noise_level, 2*samplesize)
    y = [f(X[i]) + noise[i] for i in range(2*samplesize)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)

    krr_full = KernelRidge(alpha=0.1, kernel="precomputed")
    K_train_train_full = kernel_matrix_classic(X_train, X_train)
    krr_full.fit(K_train_train_full, y_train)
    # compute training loss
    y_train_pred = krr_full.predict(K_train_train_full)
    full_train[i] = mean_squared_error(y_train_pred, y_train)
    # compute test loss
    y_full_pred = krr_full.predict(kernel_matrix_classic(X_test, X_train))
    full_test[i] = mean_squared_error(y_test, y_full_pred)

    krr_bias = KernelRidge(alpha=0, kernel="precomputed")
    K_train_train_bias = kernel_matrix_bias(X_train, X_train)
    krr_bias.fit(K_train_train_bias, y_train)
    # compute training loss
    y_train_pred = krr_bias.predict(K_train_train_bias)
    bias_train[i] = mean_squared_error(y_train_pred, y_train)
    # compute test loss
    y_bias_pred = krr_bias.predict(kernel_matrix_bias(X_test, X_train))
    bias_test[i] = mean_squared_error(y_test, y_bias_pred)

qubits = [i for i in range(1,max_qubits+1)]
errors = [full_train, full_test, bias_train, bias_test]
labels = ["full_train", "full_test", "bias_train", "bias_test"]
styles = ["dashed", "solid", "dashed", "-"]
colors = ['red', 'red', 'blue', 'blue']
for i in range(4):
    plt.plot(qubits, errors[i], label=labels[i], ls=styles[i], color=colors[i])
plt.xlabel("Qubits")
plt.ylabel("MSE")
plt.legend()
plt.savefig("exp.pdf")
