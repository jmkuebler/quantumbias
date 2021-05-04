import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt


Pi = np.pi
QUBITS = 1



@qml.template
def encoding_unitary(x, seed=0, layers=3, wires=[0, 1, 2]):
    """
    Defines the data encoding unitary
    :param x:
    :param seed: Randomly sets the parametrized gates in the Ansatz that do not depend on x. (has to be the same always)
    :param layers: Number of layers
    :param wires: list of qubits to act on
    :return:
    """
    assert isinstance(wires, list), 'List of qubits expected'
    np.random.seed(seed)
    for i in range(layers):
        # apply random unitary to each qubit and encode data in z rotation
        for j in range(len(wires)):
            angles = np.random.uniform(0, 2*Pi, 3)
            # random rotation
            qml.Rot(*angles, wires=wires[j])
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
dev = qml.device('default.qubit', wires=qubits)
@qml.qnode(dev)
def circuit(x, seed=0, layers=5, qubits=QUBITS):
    wires=[i for i in range(qubits)]
    encoding_unitary(x, seed=seed, layers=layers, wires=wires)
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev)
def full_kernel(x, y, qubits=QUBITS):
    encoding_unitary(x, wires=[i for i in range(qubits)])
    qml.inv(encoding_unitary(y, wires=[i for i in range(qubits)]))

    projector = np.zeros((2 ** qubits, 2 ** qubits))
    projector[0, 0] = 1
    return qml.expval(qml.Hermitian(projector, wires=range(qubits)))


# node for computing the biased quantum kernel
dev_double = qml.device('default.qubit', wires=2 * qubits + 1)
@qml.qnode(dev_double)
def biased_kernel_raw(x, y, qubits=QUBITS):
    encoding_unitary(x, wires=[i for i in range(qubits)])
    encoding_unitary(y, wires=[qubits + i for i in range(qubits)])

    # Swap test of first qubit each
    qml.Hadamard(wires=2*qubits)
    qml.CSWAP(wires=[2* qubits, 0, qubits]) # the 2n+1 qubit (first in list) is the control qubit.
    qml.Hadamard(wires=2*qubits)

    projector = np.zeros((2, 2))
    projector[0, 0] = 1
    return qml.expval(qml.Hermitian(projector, wires=[2*qubits]))


def biased_kernel(x,y, qubits=QUBITS):
    p_0 = biased_kernel_raw(x, y)
    # k(x,x') = 2 * (p(0) - 1/2)
    return 2 * (p_0 - 1/2)







# print(full_kernel(1,1))
# print("biased: ", biased_kernel(1,1))
# result = circuit(0.9)
# print(result)
# print(result)
# X = np.arange(0,2*Pi, 0.01)
# Y = [circuit(x) for x in X]
#
# plt.plot(X,Y)
# plt.show()



from sklearn.kernel_ridge import KernelRidge
np.random.seed(0)
samples = 10
X = np.random.uniform(0, 2*Pi, samples)
noise = np.random.normal(0,0.01, samples)
Y = [circuit(X[i]) + noise[i] for i in range(samples)]
plt.scatter(X,Y)
biased_qreg = KernelRidge(alpha=0.00000001, kernel=biased_kernel)
biased_qreg.fit(X.reshape(-1, 1), Y)
print("Done Learning")
X_plot = np.arange(0,2*Pi, 0.1)
Y_q_pred = biased_qreg.predict(X_plot.reshape((-1,1)))

full_qreg = KernelRidge(alpha=0.00000001, kernel=full_kernel)
full_qreg.fit(X.reshape(-1, 1), Y)
print("Done Learning")
Y_q_full = full_qreg.predict(X_plot.reshape((-1,1)))

Y_ground_truth = [circuit(x) for x in X_plot]
plt.plot(X_plot, Y_q_pred, label= "biased")
plt.plot(X_plot, Y_q_full, label="full")
plt.plot(X_plot, Y_ground_truth, label=r"$f^*$")
plt.legend()
plt.show()
# KernelRidge(alpha=1.0)

