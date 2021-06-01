from matplotlib.font_manager import FontProperties
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

font = {
    # 'family' : 'normal',
    # 'weight' : 'bold',
    'size': 11
}

plt.rc('font', **font)
plt.rc('lines', linewidth=2)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

fontP = FontProperties()
fontP.set_size('xx-small')

import numpy as np
import matplotlib.pyplot as plt

eigenvalues = np.load("data/thm2_data.npy")
print(eigenvalues[5, :, 3])
max_qubits = 6
mean = np.mean(eigenvalues, axis=1)
err = np.sqrt(np.var(eigenvalues, axis=1))
qubits = [j + 2 for j in range(max_qubits)]
print(qubits)
lines= ['-', '--', '-.', ':']
label = [r'$\gamma_3$', r'$\gamma_2$', r'$\gamma_1$', r'$\gamma_0$']
for i in range(3, -1, -1):
    plt.errorbar(qubits, mean[:,i], yerr=err[:,i], color="#009E73", ls=lines[i], label=label[i], capsize=3)
plt.yscale("log", base=2)
plt.xlabel(r"Number of Qubits $d$")
plt.ylabel(" ")
plt.legend(fontsize='large')
plt.savefig("evaluations/thm2.pdf", bbox_inches="tight")
