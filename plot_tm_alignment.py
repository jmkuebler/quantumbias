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

qubits = 7
samplesize = 100
runs = 1
x = [i+1 for i in range(2* samplesize)]
for qubits in range(7, 8):
    bias, full, gauss, bias_wrong = np.load("data/tm_alignment_zero_mean_{0}_{1}_{2}.npy".format(qubits, samplesize, runs))
    plt.plot(x, full, label=r"$q_1^V$", color="#E69F00")
    plt.plot(x, bias, label=r"$k$", color="#009E73")
    plt.plot(x, gauss, label=r'$k_{rbf}$', color="#56B4E9" )
    plt.plot(x, bias_wrong, label=r'$q_2^V$', color='#CC79A7')
    # plt.title("Task-Model Alignment " + str(qubits) + " qubits")
    plt.ylabel(r"$C(i)$")
    plt.xlabel(r"$i$")
    plt.legend()
    plt.xscale("log")
    # plt.savefig("evaluations/tm_alignment_finite_mean_{0}_{1}_{2}.pdf".format(qubits, samplesize, runs))
    plt.savefig("evaluations/tm_alignment_zero_mean_{0}_{1}_{2}.pdf".format(qubits, samplesize, runs), bbox_inches="tight")

    plt.close()
