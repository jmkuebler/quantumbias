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


# (qubits, full_train, full_test, bias_train, bias_test, gauss_train, gauss_test) = np.load("data/loss_fm_optimized.npy")
# (qubits, full_train, full_test, bias_train, bias_test, gauss_train, gauss_test) = np.load("data/loss_optimized_15_steps.npy")
(qubits, full_train, full_test, bias_train, bias_test, gauss_train, gauss_test, bias_2_train, bias_2_test) = np.load("data/loss_10_runs_alpha_e-3.npy")

# (qubits, full_train, full_test, bias_train, bias_test, gauss_train, gauss_test) = np.load("data/loss_10_runs.npy")
# (qubits, full_train, full_test, bias_train, bias_test, gauss_train, gauss_test) = np.load("data/loss_fm_10_runs.npy")
# (qubits, full_train, full_test, bias_train, bias_test, gauss_train, gauss_test) = np.load("data/loss_fm_1_runs_regularized.npy")


errors = [full_train, full_test, bias_train, bias_test, gauss_train, gauss_test, bias_2_train, bias_2_test]
labels = [r"$k$ train", r"$k$ test", r"$q^V_1$ train", r"$q^V_1$ test",
          r"$k_{rbf}$ train", r"$k_{rbf}$ test", r"$q^V_2$ train", r"$q^V_2$ test",]
styles = ["dashed", "solid", "dashed", "-", "dashed", "solid", "dashed", "solid"]
colors = ['#E69F00', "#E69F00", "#009E73", "#009E73", "#56B4E9", "#56B4E9", '#CC79A7', '#CC79A7']
markers = ['x', 'x', 'd', 'd', '*', '*', '.', '.']
alphas = [.5, 1, .5, 1, .5, 1, .5, 1] # plot training loss less strongly
for i in range(8):
    plt.plot(qubits, errors[i], label=labels[i], ls=styles[i], color=colors[i],alpha=alphas[i], marker=markers[i],
             markersize=10)
plt.xlabel("Qubits")
plt.ylabel("MSE")
plt.yscale("log")
plt.legend()
plt.savefig("evaluations/loss_10_alphae-3.pdf", bbox_inches="tight")
# plt.savefig("evaluations/loss_zero_mean_10runs.pdf")
