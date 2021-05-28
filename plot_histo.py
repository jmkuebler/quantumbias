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
runs = 50

bias, full, gauss, bias_wrong = np.load("data/hist_{0}_{1}_{2}.npy".format(qubits, samplesize, runs))
plt.hist((bias, full, gauss, bias_wrong), label=(r"$q$", r"$k$", r"$k_{rbf}$",r"$q_w$",), alpha=1,
         color=("#009E73", '#E69F00', '#56B4E9', '#CC79A7'))
plt.xlabel(r"Kernel Alignment $A$")
plt.legend(fontsize='large')
plt.xlim(0, 1)
plt.savefig("evaluations/hist_gauss_{0}_{1}_{2}.pdf".format(qubits, samplesize, runs), bbox_inches="tight")
plt.close()
