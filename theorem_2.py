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
samplesize = 100
max_qubits = 6
runs = 10

eigenvalues = np.zeros((max_qubits, runs, 4))


for run in tqdm(range(runs)):
    seed=run
    np.random.seed(seed)
    for i in range(max_qubits):
        qubits = i+2
        _, _, _, kernel_matrix_bias, _ = get_functions(qubits=qubits, seed=seed, second_qubit=True)

        # generate data
        X = np.random.uniform([0]*qubits, [2 * np.pi]*qubits, size=(samplesize, qubits))
        K_XX = kernel_matrix_bias(X, X)
        eigenvalues[i, run] = 1 /samplesize * np.linalg.eigvalsh(K_XX)[-4:]
np.save("data/thm2_data", eigenvalues)

# mean = np.mean(eigenvalues, axis=1)
# print(np.shape(mean))
# qubits = [j + 2 for j in range(max_qubits)]
# print(qubits)
# for i in range(4):
#     plt.plot(qubits, mean[:,i])
# plt.yscale("log")
# plt.show()

