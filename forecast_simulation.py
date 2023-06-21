import pandas as pd
import numpy as np
from packages.simulate_model import Simulator
#########
# Paths #
results_dir = "resultados/%s"

###########
# Funções #
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

####################
# Inicia simulação #
n_cvs = 5
n_mvs = 5
h = 0.01  # Passo
sys_span = 10  # Módulo máximo dos valores randômicos das matrizes dinâmicas
noise_span = sys_span / 500 # Amplitude máxima do ruído
sim = Simulator(n_cvs, n_mvs, h, sys_span, noise_span, results_dir)

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

simulation_data = pd.DataFrame()
for i in range(n_cvs):
    simulation_data[f'CV_{i+1}_real'] = y_real[:, i]
    simulation_data[f'CV_{i+1}_pred'] = y_pred[:, i]
for i in range(n_mvs):
    simulation_data[f'MV_{i+1}'] = y_pred[:, i]

filename = 'simulation_data.csv'
simulation_data.to_csv(results_dir%filename)