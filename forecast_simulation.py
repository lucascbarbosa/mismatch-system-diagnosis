import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import schur
from scipy.stats import normaltest, norm
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
from packages.simulate_model import Simulator
from packages.mdl_model import MDLModel

#########
# Paths #
simulation_dir = "03__modelagem/2023-04-05__revisao algorítimo/modelos/resultados/REFAP/simulacao/%s"


########
# Plot #
def get_changes(mismatch_log: np.array):
    """Detecta mudanças na estimação da planta"""
    mismatch_log = mismatch_log.ravel()
    changes = mismatch_log[1:] - mismatch_log[:-1]
    mismatches = np.where(changes == 1)[0] + 1
    matches = np.where(changes == -1)[0] + 1
    return matches, mismatches


def backtesting(args):
    """Previsão do backtest"""
    model, temp_model_database, ref_time, max_horizon = args
    result = model.predict(
        database=temp_model_database, max_horizon=max_horizon
    )
    prediction = result[["time", "forecast_h", "prediction"]].dropna()
    prediction = prediction[ref_time < prediction["time"]].copy()
    prediction["ref_time"] = ref_time
    prediction["output"] = output_var
    prediction = prediction[
        ["time", "output", "ref_time", "forecast_h", "prediction"]
    ]
    return prediction


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


def compute_metrics(
    y_real, y_pred, u, mismatch_log, history, tol=1e-10, save=True
):
    """Gera as métricas da previsão"""
    y_pred_match = y_pred[np.where(mismatch_log == 0)[0]]
    y_real_match = y_real[np.where(mismatch_log == 0)[0]]
    u_match = u[np.where(mismatch_log == 0)[0]]

    error_match = y_pred_match - y_real_match
    mae_match = np.mean(np.abs(error_match))
    mape_match = np.mean(
        np.abs(error_match)
        / np.where(
            y_real_match == 0,
            tol,
            np.where(
                np.abs(y_real_match) < tol,
                np.sign(y_real_match) * tol,
                y_real_match,
            ),
        )
    )

    mse_match = (error_match) ** 2
    std_match = (error_match).std()
    k2, p_norm = normaltest(error_match)
    p_unbias = norm.cdf(mae_match, loc=0, scale=std_match)
    y_pred_mismatch = y_pred[np.where(mismatch_log == 1)[0]]
    y_real_mismatch = y_real[np.where(mismatch_log == 1)[0]]
    u_mismatch = u[np.where(mismatch_log == 1)[0]]


####################
# Inicia simulação #
n_cvs = 5
n_mvs = 5
h = 0.01  # Passo
sys_span = 10  # Módulo máximo dos valores randômicos das matrizes dinâmicas
noise_span = sys_span / 10  # Amplitude máxima do ruído
sim = Simulator(n_cvs, n_mvs, h, sys_span, noise_span, simulation_dir)

N = 1000  # Horizonte de simulação
y0 = [0, 0, 0, 0, 0]  # Estado inicial
mismatch_ratio = 0.5
history = 30  # Horizonte passado analizado (varia)
n_coefs = 240
max_horizon = 480
noise = True

# Simula
u, y_real, mismatch_log, models = sim.simulate(
    N, y0, mismatch_ratio, history, noise=noise
)
matches, mismatches = get_changes(mismatch_log)
# scaler = MinMaxScaler()
# u = scaler.fit_transform(u)

for m in range(len(models)):
    model = models[m]
    print(f"Model {m+1}")
    # Extrai os coeficientes e gera o mdl
    coefs_model = sim.get_fir_coefs(model, n_coefs)
    outputs_mdl = sim.create_mdl(coefs_model)

    # Gera o banco de dados de operação
    for i in range(n_cvs):
        output_var = f"CV_{i+1}"
        print("Predicting:", output_var)
        model_database = sim.create_data(i, y_real, u)
        output_mdl = outputs_mdl[outputs_mdl["output"] == output_var].copy()
        mdl = MDLModel(mdl_parameters=output_mdl, output=output_var)
        list_args = []
        for j in range(len(model_database)):
            temp_model_database = model_database.copy()
            ref_time = model_database.iloc[j, 0]
            is_after_ref_time = ref_time < model_database["time"]
            temp_model_database.loc[is_after_ref_time, output_var] = np.nan
            list_args.append([mdl, temp_model_database, ref_time, max_horizon])

        with Pool(18) as p:
            results = p.map(backtesting, list_args)

# Plota simulação1
filename_plot = simulation_dir % "simulation_forecast.png"
# plot_simulation(matches, mismatches, u, y_real, y_pred, filename_plot)
# compute_metrics(y_real, y_pred, u, mismatch_log, history)
