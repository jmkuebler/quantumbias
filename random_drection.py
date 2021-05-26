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
qubits = 7


_, _, f, kernel_matrix_bias = get_functions(qubits=qubits, seed=1)

x = np.arange(0, np.pi*qubits, 0.1)
v = np.array([1,2,3,4,5,6,7])
# v = np.array([1]*qubits)
# v = np.array([1,0,0,0,0,0,0])
v = v / np.linalg.norm(v)
y = [f(x*v) for x in x]
plt.plot(x,y)
plt.savefig("random_projection.png")
