import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge

import time


Pi = np.pi
QUBITS = 5 # global definition of the number of qubits. Needed here already as qml.devices are specified with number



@qml.template
def general_encoding_unitary(x, layers, parameters, wires):
    """
    A layered ansatz for data encoding. Each layer has a general parametrized rotation followed by an RZ rotation with
    x and a ring of CNOT gates.
    :param x: datum to be encoded
    :param layers: number of layers
    :param parameters: parametrization of the data-independent gates in each layer
    :param wires: list of qubits to act on
    :return:
    """
    assert isinstance(wires, list), 'List of qubits expected'
    assert len(parameters) == 3* layers*qubits, 'number of parameters does not match'
    par_index = 0 # tracks the free parameters of the encoding unitary
    for i in range(layers):
        # apply random unitary to each qubit and encode data in z rotation
        for j in range(len(wires)):
            angles = parameters[par_index:par_index+3]
            par_index += 3
            # random rotation
            qml.Rot(*angles, wires=wires[j])
            # data encoding
            qml.RZ(x, wires=wires[j])
        # entangle qubits
        if len(wires) > 1:
            for j in range(len(wires)):
                qml.CNOT(wires=[wires[j], wires[(j+1)%len(wires)]])
        else:
            # no entangling possible
            pass



qubits = QUBITS # number of qubits we define our circuit on

# node for all operations on a single qubit
# dev = qml.device('default.qubit', wires=qubits)
# @qml.qnode(dev)
def circuit_function(x, qubits=QUBITS, return_reduced_state=False):
    """
    Circuit to generate the Y data
    :param x: input
    :param qubits: number of qubits in the circuit
    :param return_reduced_state: if True: not the expectation of the observable is computet but the reduced first qubit
    :return:
    """
    wires = [i for i in range(qubits)]
    data_encoding(x, wires=wires)
    if return_reduced_state:
        return qml.density_matrix(0)
    else:
        # the observable that defines the target function (f^*) is arbitrarily chosen as the pauli-Z on the first qubit.
        return qml.expval(qml.PauliZ(0))





def full_kernel_fct(x, y, qubits=QUBITS):
    """
    defines the full kernel
    :param x: input 1
    :param y: input 2
    :param qubits: how many qubits it acts on
    :return:
    """
    # initial state is all zero
    data_encoding(x, wires=[i for i in range(qubits)])
    qml.inv(data_encoding(y, wires=[i for i in range(qubits)]))

    # project back onto the all 0 state
    projector = np.zeros((2 ** qubits, 2 ** qubits))
    projector[0, 0] = 1
    return qml.expval(qml.Hermitian(projector, wires=range(qubits)))

def biased_kernel(x,y):
    # works with the reduced first qubit density operators
    rho_x = circuit(x, return_reduced_state=True)
    rho_y = circuit(y, return_reduced_state=True)
    return np.real(np.trace(rho_x @ rho_y))

# node for computing the biased quantum kernel (we need a device with 2*QUBITS + 1, with the ancilla qubit for the
# SWAP test
dev_double = qml.device('default.qubit', wires=2 * qubits + 1)
@qml.qnode(dev_double)
def biased_kernel_raw(x, y, qubits=QUBITS):
    data_encoding(x, wires=[i for i in range(qubits)])  # encode x on first register (collection of QUBITS qubits)
    data_encoding(y, wires=[qubits + i for i in range(qubits)])     # encode y on second register

    # Swap test of first qubit each
    qml.Hadamard(wires=2*qubits)    # create superposition of ancilla
    qml.CSWAP(wires=[2* qubits, 0, qubits])     # the 2n+1 qubit (first in list) is the control qubit.
    qml.Hadamard(wires=2*qubits)

    projector = np.zeros((2, 2))
    projector[0, 0] = 1
    # this is not yet the kernel (see wrapping function biased_kernel). The Pennylane architecture did not allow
    # to directly compute the kernle in the return statement.
    return qml.expval(qml.Hermitian(projector, wires=[2*qubits]))


def biased_kernel_physical(x, y):
    p_0 = biased_kernel_raw(x, y)
    # k(x,x') = 2 * (p(0) - 1/2)
    return 2 * (p_0 - 1/2)





# ---------------------   now some experiments -------------------


# define the data encoding strategy
layers = 5
np.random.seed(0)
parameters = np.random.uniform(0, 2*Pi, 3 * layers * QUBITS)

@qml.template
def data_encoding(x, wires):
    general_encoding_unitary(x, layers, parameters, wires)

dev = qml.device('default.qubit', wires=qubits)
circuit = qml.QNode(circuit_function, dev)
full_kernel = qml.QNode(full_kernel_fct, dev)


# ## ------   Some sanity checks ------
def compare_runtime():
    # adjust the global number of QUBITS.
    start = time.time()
    res = biased_kernel(1,1)
    end = time.time()
    print("time with partial density: ", end-start)

    start = time.time()
    res = biased_kernel_physical(1,1)
    end = time.time()
    print("time with Swap test: ", end-start)
compare_runtime()
# print("biased: ", biased_kernel(1,1), biased_kernel_physical(1,1))
# result = circuit(0.9)
# print(result)
# print(result)
# X = np.arange(0,2*Pi, 0.01)
# Y = [circuit(x) for x in X]
#
# plt.plot(X,Y)
# plt.show()

# ### ---------------  a simple learning problem --------------
np.random.seed(0)
samples = 10
X = np.random.uniform(0, 2*Pi, samples)
noise = np.random.normal(0,0.01, samples)
# with noies
Y = [circuit(X[i]) + noise[i] for i in range(samples)]
# without noise
# Y = [circuit(X[i]) for i in range(samples)]

# plt.scatter(X,Y)
# biased_qreg = KernelRidge(alpha=0.00000001, kernel=biased_kernel)
# biased_qreg.fit(X.reshape(-1, 1), Y)
# print("Done Learning")
# X_plot = np.arange(0,2*Pi, 0.1)
# Y_q_pred = biased_qreg.predict(X_plot.reshape((-1,1)))
#
# full_qreg = KernelRidge(alpha=0.00000001, kernel=full_kernel)
# full_qreg.fit(X.reshape(-1, 1), Y)
# print("Done Learning")
# Y_q_full = full_qreg.predict(X_plot.reshape((-1,1)))
#
# Y_ground_truth = [circuit(x) for x in X_plot]
# plt.plot(X_plot, Y_q_pred, label= "biased", ls='dashdot')
# plt.plot(X_plot, Y_q_full, label="full", ls='dashed')
# plt.plot(X_plot, Y_ground_truth, label=r"$f^*$", ls='dotted')
# plt.legend()
# plt.show()

