import os
import pandas as pd
import numpy as np
from packages.mdl_model import MDLModel
from multiprocessing import Pool
import matplotlib.pyplot as plt

#########
# Paths #
dados_brutos = "01__carregamento_do_banco/dados_brutos"
dados_tratados = "01__carregamento_do_banco/dados_tratados/"
local_path = "03__modelagem/2023-04-05__revisao algorítimo/modelos/"
dir_forecast = "03__modelagem/2023-04-05__revisao algorítimo/modelos/resultados/REFAP/previsao/"

#########################
# Carregamento de dados #
refap_data = pd.read_excel(
    os.path.join(
        dados_brutos, "dados_producao/Dados CAV propeno REFAP sem PI.xlsx"
    ),
    sheet_name="DADOS",
)
refap_data.rename(columns={"TIME": "time"}, inplace=True)
refap_mdl = pd.read_excel(os.path.join(dados_tratados, "refap/refap_mdl.xlsx"))

for c in refap_data.columns.drop("time"):
    refap_data[c] = pd.to_numeric(refap_data[c], errors="coerce")


def create_data(
    cv_name: str,
    refap_mdl: pd.DataFrame,
    refap_data: pd.DataFrame,
    backtest: bool = False,
):
    """Cria dataframe de dados e dos coeficientes MDL"""
    output_mdl = refap_mdl[refap_mdl["output"] == cv_name].copy()
    model_database = refap_data[
        ["time", cv_name] + list(output_mdl["input"].unique())
    ].copy()
    model_database.drop(columns="time")
    model_database["time"] = pd.to_datetime(model_database["time"])
    model_database.sort_values("time", inplace=True)
    model_database = filter_outliers(model_database)

    # backtest
    if backtest:
        tmp_model_database = model_database.copy()
        ref_time = model_database.iloc[1000, 0]
        is_after_ref_time = ref_time < model_database["time"]
        tmp_model_database.loc[is_after_ref_time, cv_name] = np.nan
    else:
        tmp_model_database = model_database

    return output_mdl, model_database, tmp_model_database


def filter_outliers(database: pd.DataFrame):
    """Clipa outliers amostras de 3 deltas"""
    for col in database.columns:
        mean = database[col].mean()
        std = database[col].std()
        database[col] = np.clip(database[col], mean - 3 * std, mean + 3 * std)
    return database


def run_model(
    cv_name: str,
    output_mdl: pd.DataFrame,
    model_database: pd.DataFrame,
    tmp_model_database: pd.DataFrame,
):
    model = MDLModel(mdl_parameters=output_mdl, output=cv_name)
    result_pred = model.predict(database=tmp_model_database)
    print(result_pred.columns)
    cv_real = model_database[cv_name]
    cv_pred = result_pred["prediction"].to_numpy()
    return cv_real, cv_pred


def plot_forecast(
    cv_names: list,
    cv_reals: list,
    cv_preds: list,
    dir_forecast: str,
    save: bool = True,
):
    """Plota gráficos de previsão pra cada output (CV)"""
    fig, ax = plt.subplots(len(cv_names), 1)
    fig.set_size_inches((10, 10))
    for cv_name, cv_real, cv_pred in zip(cv_names, cv_reals, cv_preds):
        idx = cv_names.index(cv_name)
        ax[idx].set_title(f"Previsão da CV {cv_name}")
        ax[idx].set_xlabel("k")
        ax[idx].plot(cv_real, label="Observação")
        ax[idx].plot(cv_pred, label="Previsão")
        ax[idx].legend()

    plt.tight_layout()
    if save:
        plt.savefig(dir_forecast + "previsao.png")

    plt.show()


cv_names = [
    "AIC300305_04",
    "AIC300305_05",
    "FFI300385",
    "PDI300418",
    "QICT300009",
]

cv_reals = []
cv_preds = []

steps = 1

for cv_name in cv_names:
    print("Predicting:", cv_name)
    # Cria base de dados de operação e coeficientes mdl
    output_mdl, model_database, tmp_model_database = create_data(
        cv_name, refap_mdl, refap_data
    )

    # Roda o modelo
    cv_real, cv_pred = run_model(
        cv_name,
        output_mdl,
        model_database,
        tmp_model_database,
    )
    cv_preds.append(cv_pred)
    cv_reals.append(cv_real)

plot_forecast(cv_names, cv_reals, cv_preds, dir_forecast)
