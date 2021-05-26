from product_methods import get_functions
from product_methods import kernel_matrix_classic

import warnings
warnings.filterwarnings("ignore")

from sklearn.gaussian_process.kernels import RBF
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def center(K):
    n = len(K)
    K = 1/ n* np.array(K)
    one_n = 1 / n * np.ones((n,n))
    K_cent = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    return K_cent

def tm_alignment(K, y):
    K = np.array(K)
    w, v = np.linalg.eigh(K)
    C = (y @ v)**2
    C = np.flip(C)
    sum = np.sum(C)
    C_n = [np.sum(C[:i+1]) / sum for i in range(len(C))]
    # print(np.linalg.norm(K,ord="fro"), y @ K @ y, y@y)
    return C_n


def task_model(qubits, samplesize=100, runs=1):
    # full = np.zeros(runs)
    # bias = np.zeros(runs)
    # gaussian = np.zeros(runs)
    for i in tqdm(range(runs)):
        seed = i
        samplesize = 100
        np.random.seed(seed)
        _, _, f, kernel_matrix_bias, kernel_matrix_bias_wrong =  get_functions(qubits=qubits, seed=seed, second_qubit=True)
        # generate data
        X = np.random.uniform([0] * qubits, [2 * np.pi] * qubits, size=(2 * samplesize, qubits))
        f_X = np.array([f(X[i]) for i in range(2 * samplesize)])
        y = f_X / np.var(f_X)
        y = y - np.mean(y)

        K_full = center(kernel_matrix_classic(X, X))
        K_bias = center(kernel_matrix_bias(X, X))
        K_gauss = center(RBF(1.0)(X))
        K_bias_wrong = center(kernel_matrix_bias_wrong(X, X))


    np.save("data/tm_alignment_zero_mean_{0}_{1}_{2}".format(qubits, samplesize, runs),
            (tm_alignment(K_bias, y), tm_alignment(K_full, y), tm_alignment(K_gauss, y), (tm_alignment(K_bias_wrong, y))))

if __name__ == '__main__':
    for qubits in range(7, 8):
        print("current number of qubits: {}".format(qubits))
        task_model(qubits, 100, 1)