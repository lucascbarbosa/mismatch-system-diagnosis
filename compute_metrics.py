import pandas as pd
import numpy as np
from scipy.stats import norm
import scipy as sp

#########
# Paths #
results_dir = "resultados/"

###########
# Funções #
def compute_lag_error(y_real, y_pred, lags):
    errors = []
    for lag in lags:
        if lag < 0:
            error = np.mean(np.abs(y_pred[:lag] - y_real[-lag:]))
        elif lag == 0:
            error = np.mean(np.abs(y_pred - y_real))
        else:
            error = np.mean(np.abs(y_pred[lag:] - y_real[:-lag]))
        errors.append(error)
    lag = errors.index(min(errors))
    errors = np.array(errors).ravel()
    return lag, errors


def compute_gain_error(y_real, y_pred):
    return (y_pred / y_real).ravel()


def compute_metrics(output_var, ganho_mod, tempo_morto_mod, y_pred, y_real):
    #     output_name, fh_group, fh_index = args
    #     """Calcula métricas de previsão"""
    error = y_pred - y_real
    error_abs = np.abs(error)
    error_sqr = np.power(error, 2)
    error_percent = error_abs / np.clip(
        np.abs(y_real), a_min=1e-10, a_max=np.inf
    )

    # Normality and zero diff
    mean_log_error = np.abs(np.mean(error))
    std_log_error = np.std(error)

    k2, p_norm = sp.stats.normaltest(error)
    p_value_diff_zero = norm.cdf(mean_log_error, loc=0, scale=std_log_error)

    delta_yt = y_real - np.concatenate((np.zeros(1), y_real[:-1]))
    sq_total = np.sum((delta_yt - np.mean(delta_yt)) ** 2)
    sq_res = np.sum(error_sqr)
    r2 = 1 - sq_res / sq_total

    return {
        "output": output_var,
        "ganho_mod": ganho_mod,
        "tempo_morto_mod": tempo_morto_mod,
        "error": np.mean(error),
        "error_mae": np.mean(error_abs),
        "error_mape": np.mean(error_percent),
        "error_mse": np.mean(error_sqr),
        "r2": r2,
        "p_nao_normalidade": p_norm,
        "p_value_diff_zero": 1 - p_value_diff_zero,
    }


#     error_tu = np.mean(error_sqr / delta_yt**2)


#     changes_real = fh_group["real"][1:] - fh_group["real"][:-1].to_numpy()
#     changes_pred = (
#         fh_group["prediction"][1:] - fh_group["prediction"][:-1].to_numpy()
#     )
#     directions = np.sign(changes_real) == np.sign(changes_pred)
#     directions = directions.astype(int)
#     error_pocid = np.round(directions.mean() * 100, 2)

#     lag, errors_lag = compute_lag(
#         fh_index, fh_group["real"], fh_group["prediction"]
#     )

#     return {
#         "output": output_index,
#         "forecast_h": fh_index,
#         "error": np.mean(error),
#         "error_mae": np.mean(error_abs),
#         "error_mse": np.mean(error_sqr),
#         "p_nao_normalidade": p_norm,
#         "p_value_diff_zero": 1 - p_value_diff_zero,
#         "error_tu": error_tu,
#         "coef_det": coef_det,
#         "error_pocid": error_pocid,
#         "tempo_morto": lag,
#     }, errors_lag


simulation_data = pd.read_excel(results_dir + "simulacao.xlsx")

gain_errors_df = []
lag_errors_df = []

cols = simulation_data.filter(like="__prediction").columns

metrics_df = []
for col in cols:
    output_var, ganho, tempo_morto, _ = col.split("__")
    ganho_mod = float(ganho.split("_")[1])
    tempo_morto_mod = float(tempo_morto.split("_")[1])
    # print(f"G: {ganho_mod} TM: {tempo_morto_mod}")
    cv_real = simulation_data[output_var]
    cv_pred = simulation_data[col].to_numpy()
    metrics = compute_metrics(
        output_var, ganho_mod, tempo_morto_mod, cv_pred, cv_real
    )
    metrics_df.append(metrics)

metrics_df = pd.DataFrame.from_dict(metrics_df)
metrics_df.to_excel(results_dir + "metricas.xlsx", index=False)
