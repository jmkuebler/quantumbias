import pennylane as qml
import numpy as np
from pennylane.templates.layers import RandomLayers

dev = qml.device("default.qubit", wires=2)
weights = [[0.1, -2.1, 1.4]]

@qml.qnode(dev)
def circuit1(weights):
    RandomLayers(weights=weights, wires=range(2))
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev)
def circuit2(weights):
    RandomLayers(weights=weights, wires=range(2))
    return qml.expval(qml.PauliZ(0))
