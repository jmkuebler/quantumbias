"""
This module contains all the functions that define the kernels
"""
import pennylane as qml
from pennylane.templates.layers import RandomLayers
import numpy as np
import matplotlib.pyplot as plt
import torch


@qml.template
def product_encoding(x, wires=[0]):
    assert len(x) == len(wires), 'number of parameters does not match number of qubits'
    # encode data via single qubit rotation of each dimension seperately
    for i in wires:
        # data encoding
        qml.RX(x[i], wires=i)


def circuit_function(x, data_encoding, qubits, seed, return_reduced_state=False, M="Z", reduced_state=0):
    np.random.seed(seed)
    wires = [i for i in range(qubits)]
    # First encode the data
    data_encoding(x, wires=wires)

    # Here we define how the random unitary matrix V is created
    layers = qubits ** 2  # we use qubits^2 rotations
    # create random angles corresponding to drawing V
    weights = np.random.uniform(0, 2*np.pi, size=(layers, qubits))
    ratio_imprim = 0.5 # use as many CNOTs as rotation gates
    RandomLayers(weights=weights, ratio_imprim=ratio_imprim, wires=wires, seed=seed)

    if return_reduced_state:
        # return the reduced state of the single qubit that defines the biased kernel.
        return qml.density_matrix(reduced_state)
    else:  # return the function f(x)
        if M=="Z":
            #  the observable that defines the target function (f^*) is arbitrarily chosen as the pauli-Z on the first qubit.
            return qml.expval(qml.PauliZ(0))
        if M=="0":
            #  you can also chooose the observable |0><0| or implement anything else.
            return qml.expval(qml.Hermitian([[1, 0], [0, 0]], wires=0))


def full_kernel_fct(x, y, data_encoding):
    """
    defines the full kernel via the simulation of the quantum circuit. for the simple kernel we consider, we directly
    evaluate its classical description
    :param x: arg 1
    :param y: arg 2
    :param data_encoding: Define the data encoding unitary
    :return:
    """
    # initial state is all zero
    qubits = len(x)
    wires = [i for i in range(qubits)]  # qubits to work on
    # encode first argument
    data_encoding(x, wires=wires)
    #  inversely encode second argument
    qml.inv(data_encoding(y, wires=wires))

    # project back onto the all 0 state
    projector = np.zeros((2 ** qubits, 2 ** qubits))
    projector[0, 0] = 1
    return qml.expval(qml.Hermitian(projector, wires=range(qubits)))


def full_kernel_classic(x, y):
    """
    Classical evaluation of the full kernel
    :param x: arg 1
    :param y: arg 2
    :return:
    """
    k = 1
    for i in range(len(x)):
        k = k * np.cos(1/2*(x[i]-y[i]))**2
    return k


def kernel_matrix_classic(X, Y):
    """
    compute the kenel matrix of the full kernel
    :param X: vector of samples
    :param Y: vector of samples
    :return:
    """
    K = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            K[i, j] = full_kernel_classic(X[i], Y[j])
    return K


def kernel_matrix_classic_torch(X, Y):
    """
    compute the kenel matrix of the full kernel potentially utilizing some parallel processing
    :param X: vector of samples
    :param Y: vector of samples
    :return:
    """
    if type(X) is np.ndarray:
        X = torch.from_numpy(X)
    if type(Y) is np.ndarray:
        Y = torch.from_numpy(Y)
    # create tensor with entry i x j x k equal to x_ik - y_jk
    X = X.unsqueeze(1).expand(-1, Y.size(0), -1)
    Y = Y.unsqueeze(0).expand(X.size(0), -1, -1)
    K = X - Y
    K = torch.cos(K / 2) ** 2
    K = torch.prod(K, 2)
    return K


def biased_kernel_fct(x, y, reduced_state):
    """
    Compute the biased kernel function with the reduced density matrix rho(x) = reduced_state(x)
    :param x:
    :param y:
    :param reduced_state: function that takes a single argument and returns the a reduced denisty operator
    :return:
    """
    # works with the reduced first qubit density operators
    rho_x = reduced_state(x)
    rho_y = reduced_state(y)
    k = np.real(np.trace(rho_x @ rho_y))
    return k


def biased_kernel_matrix(X, Y, reduced_state):
    """
    Compute the kernel matrix of the biased kernel with torch
    :param X: input vector of data
    :param Y: input vector of data
    :param reduced_state: function that takes a single argument and returns the a reduced denisty operator
    :return: kernel matrix
    """
    # works with the reduced density operators
    rho_X = torch.tensor([np.array(reduced_state(x)) for x in X]) # compute reduced density operator for all inputs
    rho_Y = torch.tensor([np.array(reduced_state(y)) for y in Y])
    K = np.real(torch.einsum('aik,bki -> ab', rho_X, rho_Y)) # trace[rho(x)rho(y)
    return K # formula for the swap test


def get_functions(qubits, seed, M='Z', second_qubit=False):
    """
    Utility function that creates all the functionalities for a experimental setting
    :param qubits: number of qubits involved == dimensionality of data
    :param seed: random seed to generate V
    :param M: observable on qubit 1 that defines the target functioon
    :param second_qubit: If trueallso returns functions to compute the kernel matrix of q_w (2nd qubit)
    :return:
    """
    dev = qml.device('default.qubit', wires=qubits)
    # define the full kernel
    k_prod_fct = lambda x, y: full_kernel_fct(x, y, data_encoding=product_encoding)
    k_prod = qml.QNode(k_prod_fct, dev)

    # make everything work with the pennylane interfaces

    # function that returns reduced state of first qubit
    reduced_state_fct = lambda x: circuit_function(x, product_encoding, qubits, return_reduced_state=True, seed=seed, reduced_state=0)
    reduced_state = qml.QNode(reduced_state_fct, dev)
    # kernel matrix for biased kernel of qubit 1
    k_prod_bias = lambda x, y: biased_kernel_fct(x, y, reduced_state)
    kernel_matrix_bias = lambda X, Y: biased_kernel_matrix(X, Y, reduced_state)

    # target function
    f_fct = lambda x: circuit_function(x, product_encoding, qubits, return_reduced_state=False, seed=seed, M=M)
    f = qml.QNode(f_fct, dev)

    # potentially add second qubit biased kernel functionality
    if second_qubit==True and qubits > 1:
        reduced_state_fct_2 = lambda x: circuit_function(x, product_encoding, qubits, return_reduced_state=1, seed=seed, reduced_state=1)
        reduced_state_2 = qml.QNode(reduced_state_fct_2, dev)
        k_prod_bias_2 = lambda x, y: biased_kernel_fct(x, y, reduced_state_2)
        kernel_matrix_bias_2 = lambda X, Y: biased_kernel_matrix(X, Y, reduced_state_2)
        return k_prod, k_prod_bias, f, kernel_matrix_bias, kernel_matrix_bias_2
    else:
        return k_prod, k_prod_bias, f, kernel_matrix_bias, kernel_matrix_bias

def reduced_density(x, qubits, seed):
    dev = qml.device('default.qubit', wires=qubits)
    reduced_state_fct = lambda x: circuit_function(x, product_encoding, qubits, return_reduced_state=True, seed=seed)
    reduced_state = qml.QNode(reduced_state_fct, dev)
    return reduced_state(x)


def tests():
    """
    Some test to check whether everything behaves the way it should. Just ignore.
    :return:
    """
    qubits = 4
    dev = qml.device('default.qubit', wires=qubits)
    # define the full kernel
    k_prod_fct = lambda x, y: full_kernel_fct(x, y, data_encoding=product_encoding)
    k_prod = qml.QNode(k_prod_fct, dev)

    reduced_state_fct = lambda x: circuit_function(x, product_encoding, qubits, return_reduced_state=True)
    reduced_state = qml.QNode(reduced_state_fct, dev)
    k_prod_bias = lambda x, y: biased_kernel_fct(x, y, reduced_state)

    print(k_prod([1]*qubits, [1]*qubits))
    print(k_prod_bias([1]*qubits, [1]*qubits))


def rkhs_dimension():
    """
    More tests
    :return:
    """
    qubit_count = [i for i in range(1,6)]
    ranks = [0 for qubit in qubit_count]
    for qubits in qubit_count:
        i = qubits-1
        k_full, k_bias, f, _ = get_functions(qubits, seed=0)

        samples = 150
        X = np.random.uniform([0]*qubits, [2 * np.pi]*qubits, size=(samples, qubits))
        K = np.array([[k_full(x, y) for x in X] for y in X])  # TODO: optimize by only computing upper triang
        # K_bias = np.array([[k_bias(x, y) for x in X] for y in X])

        ranks[i] = np.linalg.matrix_rank(K)
        print("i= ", i, "rank= ", ranks[i])

    plt.scatter(qubit_count, ranks)
    plt.show()



if __name__ == "__main__":
    pass