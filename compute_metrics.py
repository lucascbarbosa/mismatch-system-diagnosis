import pandas as pd
import numpy as np
from scipy.stats import normaltest, norm

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