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


(qubits, full_train, full_test, bias_train, bias_test, gauss_train, gauss_test, bias_2_train, bias_2_test) = np.load("data/loss_optimized_15_steps.npy")
# (qubits, full_train, full_test, bias_train, bias_test, gauss_train, gauss_test, bias_2_train, bias_2_test) = np.load("data/loss_10_runs_alpha_e-3.npy")
# (qubits, full_train, full_test, bias_train, bias_test, gauss_train, gauss_test, bias_2_train, bias_2_test) = np.load("data/loss_fm_10_runs_alpha_e-3.npy")


errors = [bias_train, bias_test,full_train, full_test,  gauss_train, gauss_test, bias_2_train, bias_2_test]
labels = [r"$q$ train", r"$q$ test", r"$k$ train", r"$k$ test",
          r"$k_{rbf}$ train", r"$k_{rbf}$ test", r"$q_w$ train", r"$q_w$ test",]
styles = ["dashed", "solid", "dashed", "-", "dashed", "solid", "dashed", "solid"]
colors = ["#009E73", "#009E73", '#E69F00', "#E69F00",  "#56B4E9", "#56B4E9", '#CC79A7', '#CC79A7']
markers = ['d', 'd','x', 'x',  '*', '*', '.', '.']
alphas = [.5, 1, .5, 1, .5, 1, .5, 1] # plot training loss less strongly
for i in range(8):
    plt.plot(qubits, errors[i], label=labels[i], ls=styles[i], color=colors[i],alpha=alphas[i], marker=markers[i],
             markersize=10)
plt.xlabel(r"Number of Qubits $d$")
plt.ylabel("Mean Squared Error")
plt.yscale("log")
plt.legend(loc='right', fontsize='large',bbox_to_anchor=(1.32, .5))
# plt.savefig("evaluations/loss_10_alphae-3.pdf", bbox_inches="tight")
plt.savefig("evaluations/loss_optimizedHP.pdf", bbox_inches="tight")
# plt.savefig("evaluations/loss_fm_10_alphae-3.pdf", bbox_inches="tight")
