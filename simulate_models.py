import os
import sys
import pandas as pd
import numpy as np
from packages.mdl_model import MDLModel
from multiprocessing import Pool

#########
# Paths #
models_dir = "modelos/excel/"
results_dir = "resultados/"

#############
# Functions #
def create_database(model_original: str, N: int):
    "Cria base de dados reais (simulação do modelo original)"
    model_mdl = pd.read_excel(os.path.join(models_dir, model_original))
    outputs_name = list(model_mdl["output"].unique())
    inputs_name = list(model_mdl["input"].unique())
    data = pd.DataFrame()
    time = pd.date_range(start=0, periods=N, freq="20T").to_numpy()
    data["time"] = time
    data[outputs_name] = np.nan
    data.loc[0, outputs_name] = np.random.uniform(0, 1, size=len(outputs_name))
    inputs = np.random.uniform(0, 10, size=(N, len(inputs_name)))
    data[inputs_name] = inputs
    for output_var in outputs_name:
        output_mdl = model_mdl[model_mdl["output"] == output_var].copy()
        model = MDLModel(mdl_parameters=output_mdl, output=output_var)
        database = data[
            ["time", output_var] + list(output_mdl["input"].unique())
        ].copy()
        ref_time = database.iloc[0, 0]
        database.drop(columns="time")
        database["time"] = pd.to_datetime(database["time"])
        database.sort_values("time", inplace=True)
        args = [model, database, output_var, ref_time]
        prediction_df = forecast(args)
        data.loc[1:, output_var] = prediction_df["prediction"].to_numpy()
    return data, outputs_name, inputs_name


def forecast(args):
    """Realiza a previsão a 1 passo a frente"""
    model, model_database, output_var, ref_time = args
    result = model.predict(
        database=model_database, max_horizon=len(model_database)
    )
    prediction = result.filter(
        regex="(time|forecast_h|prediction|effect)$"
    ).dropna()
    prediction["ref_time"] = ref_time
    prediction["output"] = output_var

    index_ = ["time", "output", "ref_time"]
    prediction_df = prediction.set_index(index_).reset_index()
    return prediction_df


def get_ganho_tm_mods(index, ganho_mods, tempo_morto_mods):
    ganho_mod = ganho_mods[index // len(tempo_morto_mods)]
    tempo_morto_mod = tempo_morto_mods[index % len(tempo_morto_mods)]
    return ganho_mod, tempo_morto_mod


# Original model filename
model_original = "ORIGINAL_1MV.xlsx"

# Create database
N = 1000
real_data, list_output_var, list_input_var = create_database(model_original, N)
for c in real_data.columns.drop("time"):
    real_data[c] = pd.to_numeric(real_data[c], errors="coerce")

simulation_data = pd.DataFrame()
simulation_data[list_output_var] = real_data[list_output_var]
simulation_data[list_input_var] = real_data[list_input_var]

# Load mismatched models
mismatch_models_filename = model_original[:-5] + "__mismatch_models.xlsx"
mismatch_models_df = pd.read_excel(
    os.path.join(models_dir, mismatch_models_filename)
)
list_args = []
ganho_mods = mismatch_models_df["ganho_mod"].unique()
tempo_morto_mods = mismatch_models_df["tempo_morto_mod"].unique()
for (ganho_mod, tempo_morto_mod), model_mdl in mismatch_models_df.groupby(
    ["ganho_mod", "tempo_morto_mod"]
):
    print(f"G: {ganho_mod} TM: {tempo_morto_mod}")
    for output_var in list_output_var:
        print("Forecasting: %s\n" % output_var)
        output_mdl = model_mdl[model_mdl["output"] == output_var].copy()
        model = MDLModel(mdl_parameters=output_mdl, output=output_var)
        model_database = real_data[
            ["time", output_var] + list(output_mdl["input"].unique())
        ].copy()
        ref_time = model_database.iloc[0, 0]
        model_database.drop(columns="time")
        model_database["time"] = pd.to_datetime(model_database["time"])
        model_database.sort_values("time", inplace=True)
        args = [model, model_database, output_var, ref_time]
        list_args.append(args)

with Pool(18) as p:
    results = p.map(forecast, list_args)


idx = 0
for prediction_df in results:
    ganho_mod, tempo_morto_mod = get_ganho_tm_mods(
        idx, ganho_mods, tempo_morto_mods
    )
    print(idx, ganho_mod, tempo_morto_mod)
    prediction_col = (
        output_var + "__" + f"G_{ganho_mod}__TM_{tempo_morto_mod}__prediction"
    )
    simulation_data.loc[0, prediction_col] = model_database.loc[0, output_var]
    simulation_data.loc[1:, prediction_col] = prediction_df[
        "prediction"
    ].to_numpy()
    idx += 1
simulation_data.to_excel(results_dir + "simulacao.xlsx", index=False)
