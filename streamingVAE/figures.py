import os
import numpy as np
import matplotlib.pyplot as plt

outdir = "./output/modelselection"


def simple_plots():
    N = 100000
    batch_size = 10
    model_H = (2, 4, 6)
    model_elbo = []
    model_elbo_BIC = []
    model_entropies = []
    model_entropies_BIC = []
    model_gradmagnitude = []

    for H in model_H:
        model_elbo.append(np.loadtxt(os.path.join(outdir, "elbo-{}.txt".format(H))))
        model_elbo_BIC.append(np.loadtxt(os.path.join(outdir, "elbo-BIC-{}.txt".format(H))))
        model_entropies.append(np.loadtxt(os.path.join(outdir, "entropies-{}.txt".format(H))))
        model_entropies_BIC.append(np.loadtxt(os.path.join(outdir, "entropies-BIC-{}.txt".format(H))))
        model_gradmagnitude.append(np.loadtxt(os.path.join(outdir, "gradient-{}.txt".format(H))))

    colors = ("tab:blue", "tab:orange", "tab:green")

    plt.figure(figsize=[5, 4])
    for H, BIC, color in zip(model_H, model_elbo_BIC, colors):
        plt.plot(np.clip(BIC, None, 1000), 
                label="$H=" + ("{}$. ELBO".format(H)), 
                color=color, alpha=0.3)
    for H, BIC, color in zip(model_H, model_entropies_BIC, colors):
        plt.plot(BIC, label="$H=" + ("{}$. 3-entropies".format(H)), ls="-", color=color)
    plt.gca().grid(True, which='major',linewidth=0.5)
    plt.gca().tick_params(labelsize=9, pad=1)
    plt.legend(loc="lower left", fontsize=7)
    plt.xlabel("Batch", fontsize=12)
    plt.ylabel("BIC score", fontsize=12)
    #plt.ylabel("BIC from ELBO", fontsize=12)
    ymin, ymax = plt.gca().get_ylim()
    plt.ylim([-130, 590])
    plt.vlines(x=[N / batch_size, 2*N / batch_size], ymin=ymin, ymax=ymax, colors='blue', ls='-.', lw=1.0, zorder=1.9)
    for i, H_data in enumerate((6, 2, 4)):
        plt.text(3000+10000*i, 550, "$H_{data}=" + ("{}$".format(H_data)), fontsize=10, color='black', verticalalignment='center')
    plt.title("(a) BIC for streaming linear VAE", fontsize=16)
    #plt.title("(a) Model selection with ELBO", fontsize=16)
    plt.savefig(os.path.join(outdir, "modelselection-BIC.pdf"))
    plt.close()


if __name__ == "__main__":
    simple_plots()
