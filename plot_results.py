import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#########
# Paths #
results_dir = "resultados/%s"

#########
# Plots #
def plot_simulation(
    matches: list,
    mismatches: list,
    u: np.array,
    y_real: np.array,
    y_pred: np.array,
    filename: str,
    save_plot=False,
):
    """Plota gráfico da simulação"""
    n_cvs = y_real.shape[1]
    n_mvs = u.shape[1]
    fig, axs = plt.subplots(n_cvs, 1)
    fig.set_size_inches((10, 10))
    for i in range(n_cvs):
        axs[i].set_title(f"Previsão da CV {i+1}")
        axs[i].set_xlabel("k")
        axs[i].plot(y_real[:, i], label="Observação")
        axs[i].plot(y_pred[:, i], label="Previsão", linestyle="--")
        for change in matches:
            axs[i].axvline(change, color="g", linestyle="--")
        for change in mismatches:
            axs[i].axvline(change, color="r", linestyle="--")
        axs[i].legend()

    plt.tight_layout()

    if save_plot:
        plt.savefig(filename)

    plt.show()

simulation_data = pd.read_csv(results_dir % 'simulation_data.csv')
