from product_methods import get_functions
from product_methods import kernel_matrix_classic

import warnings
warnings.filterwarnings("ignore")

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def alignment(K, y):
    K = np.array(K)
    alignment = y @ K @ y / (np.linalg.norm(K, ord="fro") * y@y)
    # print(np.linalg.norm(K,ord="fro"), y @ K @ y, y@y)
    return alignment

def center(K):
    n = len(K)
    K = 1/ n* np.array(K)
    one_n = 1 / n * np.ones((n,n))
    K_cent = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    return K_cent


def histograms(qubits, samplesize=100, runs=100):
    full = np.zeros(runs)
    bias = np.zeros(runs)
    gaussian = np.zeros(runs)
    for i in tqdm(range(runs)):
        seed = i
        samplesize = 100
        np.random.seed(seed)
        _, _, f, kernel_matrix_bias = get_functions(qubits=qubits, seed=seed)
        # generate data
        X = np.random.uniform([0] * qubits, [2 * np.pi] * qubits, size=(2 * samplesize, qubits))
        f_X = np.array([f(X[i]) for i in range(2 * samplesize)])
        y = f_X / np.var(f_X)

        K_full_c = center(kernel_matrix_classic(X, X))
        K_bias_c = center(kernel_matrix_bias(X, X))

        K_gauss_c = center(RBF(1.0)(X))


        bias[i] = alignment(K_bias_c, y)
        full[i] = alignment(K_full_c, y)
        gaussian[i] = alignment(K_gauss_c, y)

    np.save("data/hist_gauss_{0}_{1}_{2}".format(qubits, samplesize, runs), (bias,full, gaussian))
    # plt.hist(full, label="full kernel")
    # plt.hist(bias, label="biased kernel")
    # plt.title("alignment for " + str(qubits) + " qubits")
    # plt.legend()
    # plt.savefig("hist_{0}_{1}_{2}.png".format(qubits, samplesize, runs))
    # plt.close()


# noise_level = 0.1 # level of label noise
#
#
# qubits = 7
# seed = 1
# # qubits = 2
# # seed = 5
# _, _, f, kernel_matrix_bias = get_functions(qubits=qubits, seed=seed)
# # x = np.arange(0,2*np.pi, 0.05)
# # y = [f([x,0]) for x in x]
# # plt.plot(x,y)
# # plt.show()
#
# samplesize = 50
# np.random.seed(seed)
# # generate data
# X = np.random.uniform([0]*qubits, [2*np.pi]*qubits, size=(2*samplesize, qubits))
# noise = np.random.normal(0, noise_level, 2*samplesize)
# f_X = np.array([f(X[i]) for i in range(2*samplesize)])
# f_X = f_X / np.var(f_X)
# y = np.array(f_X + noise) - np.mean(f_X)
# # #
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)
# #
# # krr_full = KernelRidge(alpha=0.05, kernel="precomputed")
# # K_train_train_full = kernel_matrix_classic(X_train, X_train)
# # krr_full.fit(K_train_train_full, y_train)
# # # compute test loss
# # y_full_pred = krr_full.predict(kernel_matrix_classic(X_test, X_train))
# # print("full test loss= ", mean_squared_error(y_test, y_full_pred))
# # #
# # krr_bias = KernelRidge(alpha=0., kernel="precomputed")
# # K_train_train_bias = kernel_matrix_bias(X_train, X_train)
# # krr_bias.fit(K_train_train_bias, y_train)
# # # compute test loss
# # y_bias_pred = krr_bias.predict(kernel_matrix_bias(X_test, X_train))
# # print("biased test loss= ", mean_squared_error(y_test, y_bias_pred))
#
#
#
# K_full_c = center(kernel_matrix_classic(X, X))
#
# K_bias_c = center(kernel_matrix_bias(X, X))
# # print(np.linalg.eigvalsh(K_full_c))
# w,v = np.linalg.eigh(K_full_c)
# # print(np.linalg.eigvalsh())
#
# y_norm = y / np.sqrt(y@y)
# print("three leading evalues", w[-9:])
# print("inner product with 3 leading ev", [(y_norm@v) for v in np.transpose(v)[-9:]])
#
# print("linear combination", np.linalg.solve(v, y_norm)[-9:])
# #
# # print("nominator larger ", y@K_bias_c@y > y@K_full_c@y)
# #
# # print("denominator smaller", np.linalg.norm(K_bias_c, ord="fro") < np.linalg.norm(K_full_c, ord="fro"))
#
# # print(K_bias, y)
# #
# print("expectation of y at full kernel - normalized with root of sum of squared ev (equals alignment)", y_norm @ K_full_c @ y_norm / np.sqrt(np.sum(np.linalg.eigvalsh(K_full_c)**2)))
#
# print("centered alignment:   full: ", alignment(K_full_c, y), "   biased: ", alignment(K_bias_c, y))
# #
# #
# # print(K_bias_c / np.linalg.norm(K_bias_c, ord="fro"))
# # print(K_full_c / np.linalg.norm(K_full_c, ord="fro"))


if __name__ == '__main__':
    for qubits in range(1, 8):
        print("current number of qubits: {}".format(qubits))
        histograms(qubits, 100, 200)