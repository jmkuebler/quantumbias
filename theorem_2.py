from product_methods import get_functions
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
samplesize = 100
max_qubits = 6
runs = 10

eigenvalues = np.zeros((max_qubits, runs, 4))

for run in tqdm(range(runs)):
    seed = run
    np.random.seed(seed)
    for i in range(max_qubits):
        qubits = i+2
        _, _, _, kernel_matrix_bias, _ = get_functions(qubits=qubits, seed=seed, second_qubit=True)

        # generate data
        X = np.random.uniform([0]*qubits, [2 * np.pi]*qubits, size=(samplesize, qubits))
        K_XX = kernel_matrix_bias(X, X)
        eigenvalues[i, run] = 1 /samplesize * np.linalg.eigvalsh(K_XX)[-4:]
np.save("data/thm2_data", eigenvalues)


