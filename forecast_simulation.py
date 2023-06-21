import pandas as pd
import numpy as np
from scipy.stats import normaltest, norm
from packages.simulate_model import Simulator
import seaborn as sns
import matplotlib.pyplot as plt
#########
# Paths #
simualtion_dir = "resultados/%s"

########
# Plot #
def get_changes(mismatch_log: np.array):
    """Detecta mudanças na estimação da planta"""
    mismatch_log = mismatch_log.ravel()
    changes = mismatch_log[1:] - mismatch_log[:-1]
    mismatches = np.where(changes == 1)[0] + 1
    matches = np.where(changes == -1)[0] + 1
    return matches, mismatches

def forecast(model, y_real, u, horizon):
    """Prevê as saídas em um horizonte específico"""
    A, B = model
    y_pred = np.zeros(y_real.shape)
    y_pred[0, :] = y_real[0, :]
    N = y_real.shape[0]
    for i in range(1, N):
        y_row = y_pred[i - 1, :].reshape((n_cvs, 1))
        u_row = u[i - 1, :].reshape((n_mvs, 1))
        y_pred[i, :] = np.dot(A, y_row).T + np.dot(B, u_row).T 

    return y_pred         

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


def compute_error(y_real, y_pred, mismatch_log, tol=1e-10, save=True):
    """Gera as métricas da previsão"""
    error = y_pred - y_real
    error_abs = np.abs(error)
    error_percent = np.abs(error) / np.where(
            y_real == 0,
            tol,
            np.where(
                np.abs(y_real) < tol,
                np.sign(y_real) * tol,
                y_real,
            ),
        )

    error_sqr = (error) ** 2
    std = (error).std()
    # _, p_norm = normaltest(error)
    # p_unbias = norm.cdf(error_abs, loc=0, scale=std)

    changes_real = y_real[1:] - y_real[:-1]
    changes_pred = y_pred[1:] - y_pred[:-1]
    directions = np.sign(changes_real) == np.sign(changes_pred).astype(int)
    directions = np.append(np.zeros(1), directions)

    errors_df = pd.DataFrame()
    for i in range(error.shape[1]):
        errors_df[f'error_CV_{i+1}'] = error[:,i]
        errors_df[f'error_abs_CV_{i+1}'] = error_abs[:,i]
        errors_df[f'error_sqr_CV_{i+1}'] = error_sqr[:,i]
        errors_df[f'error_percent_CV_{i+1}'] = error_percent[:,i]

    errors_df[f'mismatch_log'] = mismatch_log
    return errors_df

def compute_metric(errors_df, history):
    errors_match = errors_df[errors_df["mismatch_log"]==0.0]
    errors_mismatch = errors_df[errors_df["mismatch_log"]==1.0]

    for i in range(n_cvs):
        pass
####################
# Inicia simulação #
n_cvs = 5
n_mvs = 5
h = 0.01  # Passo
sys_span = 10  # Módulo máximo dos valores randômicos das matrizes dinâmicas
noise_span = sys_span / 500 # Amplitude máxima do ruído
sim = Simulator(n_cvs, n_mvs, h, sys_span, noise_span, simualtion_dir)

N = 5000  # Horizonte de simulação
y0 = [0, 0, 0, 0, 0]  # Estado inicial
mismatch_ratio = 0.5
history = 30  # Horizonte passado analizado (varia)
n_coefs = 240
max_horizon = 480
noise = True

# Simula
u, y_real, mismatch_log, models_pred, models_real= sim.simulate(
    N, y0, mismatch_ratio, history, noise=noise
)
matches, mismatches = get_changes(mismatch_log)
matches_limits = np.zeros(matches.shape[0]+2).astype(int)
matches_limits[0] = 0
matches_limits[1:-1] = matches
matches_limits[-1] = N
y_pred = np.zeros(y_real.shape)

for m in range(len(models_pred)):
    model= models_pred[m]
    # print(f"Model {m+1}")
    y_window = y_real[matches_limits[m]:matches_limits[m+1],:]
    u_window = u[matches_limits[m]:matches_limits[m+1],:]
    # Gera o banco de dados de operação
    result = forecast(model,
                      y_window, 
                      u, 
                      -1)
    y_pred[matches_limits[m]:matches_limits[m+1],:] = result
    # Plota simulação

filename_plot = simualtion_dir % "simulation_forecast.png"
# plot_simulation(matches, mismatches, u, y_real, y_pred, filename=None)
errors_df = compute_error(y_real, y_pred, mismatch_log)
compute_metric(errors_df, history)