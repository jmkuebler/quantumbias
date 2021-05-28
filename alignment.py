### Experiments to reproduce the kernel target alignment
###
from product_methods import get_functions
from product_methods import kernel_matrix_classic

from tqdm import tqdm
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import warnings
warnings.filterwarnings("ignore")  # to ignore sparse matrix warning in Kernel Ridge Regression


def alignment(K, y):
    # compute the  empirical kernel target alignment.
    K = np.array(K)
    alignment = y @ K @ y / (np.linalg.norm(K, ord="fro") * y@y)
    return alignment


def center(K):
    # centers the kernel matrix
    n = len(K)
    K = 1 / n * np.array(K)
    one_n = 1 / n * np.ones((n, n))
    K_cent = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    return K_cent


def histograms(qubits, samplesize=100, runs=100):
    full = np.zeros(runs)
    bias = np.zeros(runs)
    gaussian = np.zeros(runs)
    bias_wrong = np.zeros(runs)
    for i in tqdm(range(runs)):
        seed = i
        np.random.seed(seed)
        _, _, f, kernel_matrix_bias, kernel_matrix_bias_wrong =  get_functions(qubits=qubits, seed=seed, second_qubit=True)
        # generate data
        X = np.random.uniform([0] * qubits, [2 * np.pi] * qubits, size=(2 * samplesize, qubits))
        f_X = np.array([f(X[i]) for i in range(2 * samplesize)])
        y = f_X / np.var(f_X)

        K_full_c = center(kernel_matrix_classic(X, X))
        K_bias_c = center(kernel_matrix_bias(X, X))

        K_gauss_c = center(RBF(1.0)(X))
        K_bias_wrong = center(kernel_matrix_bias_wrong(X, X))

        bias[i] = alignment(K_bias_c, y)
        full[i] = alignment(K_full_c, y)
        gaussian[i] = alignment(K_gauss_c, y)
        bias_wrong[i] = alignment(K_bias_wrong, y)

    np.save("data/hist_{0}_{1}_{2}".format(qubits, samplesize, runs), (bias,full, gaussian, bias_wrong))


if __name__ == '__main__':
    for qubits in range(1, 7):
        print("current number of qubits: {}".format(qubits))
        histograms(qubits, 100, 50)
