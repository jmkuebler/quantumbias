import pennylane as qml
from pennylane.templates.layers import RandomLayers
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.kernel_ridge import KernelRidge


@qml.template
def product_encoding(x, wires=[0]):
    assert len(x) == len(wires), 'number of parameters does not match number of qubits'
    for i in wires:
        # data encoding
        qml.RX(x[i], wires=i)


def circuit_function(x, data_encoding, qubits, return_reduced_state=False, seed=0):
    np.random.seed(seed)
    wires = [i for i in range(qubits)]
    # First encode the data
    data_encoding(x, wires=wires)


    layers = qubits ** 2  # we use qubits^2 rotations
    # create random angles
    weights = np.random.uniform(0, 2*np.pi, size=(layers, qubits))
    ratio_imprim = 0.5 # use as many CNOTs as rotation gates
    RandomLayers(weights=weights, ratio_imprim=ratio_imprim, wires=wires, seed=seed)

    if return_reduced_state:
        return qml.density_matrix(0)
    else:
        # the observable that defines the target function (f^*) is arbitrarily chosen as the pauli-Z on the first qubit.
        return qml.expval(qml.PauliZ(0))


def full_kernel_fct(x, y, data_encoding):
    # initial state is all zero
    qubits = len(x)
    wires = [i for i in range(qubits)]  # qubits to work on
    data_encoding(x, wires=wires)
    qml.inv(data_encoding(y, wires=wires))

    # project back onto the all 0 state
    projector = np.zeros((2 ** qubits, 2 ** qubits))
    projector[0, 0] = 1
    return qml.expval(qml.Hermitian(projector, wires=range(qubits)))

def full_kernel_classic(x,y):
    k = 1
    for i in range(len(x)):
        k = k* np.cos(1/2*(x[i]-y[i]))**2
    return k


def biased_kernel_fct(x, y, reduced_state):
    # works with the reduced first qubit density operators
    rho_x = reduced_state(x)
    rho_y = reduced_state(y)
    p0 = np.real(np.trace(rho_x @ rho_y))
    return 2 * (p0 - 1 / 2)

def biased_kernel_matrix(X, reduced_state):
    # works with the reduced first qubit density operators
    print(len(X))
    rho_X = torch.tensor([np.array(reduced_state(x)) for x in X])
    p0 = torch.einsum('aik,bki -> ab', rho_X, rho_X) # trace[rho(x)rho(y)
    return 2 * (p0 - 1 / 2)


def get_functions(qubits, seed=0):
    dev = qml.device('default.qubit', wires=qubits)
    # define the full kernel
    k_prod_fct = lambda x, y: full_kernel_fct(x, y, data_encoding=product_encoding)
    k_prod = qml.QNode(k_prod_fct, dev)

    reduced_state_fct = lambda x: circuit_function(x, product_encoding, qubits, return_reduced_state=True, seed=seed)
    reduced_state = qml.QNode(reduced_state_fct, dev)
    k_prod_bias = lambda x, y: biased_kernel_fct(x, y, reduced_state)
    compute_K_biased = lambda x: biased_kernel_matrix(x, reduced_state)

    f_fct = lambda x: circuit_function(x, product_encoding, qubits, return_reduced_state=False, seed=seed)
    f = qml.QNode(f_fct, dev)

    return k_prod, k_prod_bias, f, compute_K_biased


def tests():
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

def classic_vs_quantum():
    qubits = 3
    k_full, _, _ = get_functions(qubits, seed=0)
    print(k_full([1]*qubits, [2]*qubits), full_kernel_classic([1]*qubits, [2]*qubits))


def test_kernel_matrix():
    qubits = 3
    k_full, k_bias, f, compute_K_bias = get_functions(qubits, seed=0)
    samples = 5
    X = np.random.uniform([0] * qubits, [2 * np.pi] * qubits, size=(samples, qubits))
    K_bias = np.array([[k_bias(x, y) for x in X] for y in X])
    print(K_bias)

    K_direct = compute_K_bias(X)
    print(K_direct)

if __name__ == "__main__":
    # tests()
    # rkhs_dimension()
    # classic_vs_quantum()
    test_kernel_matrix()