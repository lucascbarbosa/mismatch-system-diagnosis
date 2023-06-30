import pandas as pd
import numpy as np
from packages.parse_model import MDLParser
import os


#########
# Paths #
mdl_dir = "modelos/mdl/"
excel_dir = "modelos/excel/%s"
model_filename_mdl = "ORIGINAL_1MV.dmi"
model_filename_excel = "ORIGINAL_1MV.xlsx"
filenames = os.listdir(excel_dir[:-2])
# Remove previously saved models
if len(filenames) > 1:
    for filename in filenames:
        if filename != model_filename_excel:
            os.remove(excel_dir % filename)

###############
# Parse  models #
parser = MDLParser()
filename_in = os.path.join(mdl_dir, model_filename_mdl)
filename_out = excel_dir % model_filename_excel
parser.extract_model(filename_in, filename_out)
# filename_plot = None
# parser.plot_model(filename_plot, save=False)

###############
# Parse  models #
original_filename = excel_dir % model_filename_excel
original_mdl = pd.read_excel(excel_dir % model_filename_excel)

# Mismatch parameters
tempo_morto_mods = np.arange(-9,  10, 1)
ganho_mods = np.concatenate(
    (np.array([0.1, 0.25, 0.5, 0.75]), np.arange(1, 10, 1))
)

list_mod_mdl = []
for ganho_mod in ganho_mods:
    for tempo_morto_mod in tempo_morto_mods:
        mod_mdl = original_mdl.copy()
        mod_mdl["ganho_mod"] = ganho_mod
        mod_mdl["tempo_morto_mod"] = tempo_morto_mod
        for model_index, model_group in original_mdl.groupby(
            ["output", "input"]
        ):
            coefs = model_group["coef"].to_numpy()
            if tempo_morto_mod < 0:
                coefs_mod = np.concatenate(
                    (
                        coefs[-tempo_morto_mod:],
                        np.full(-tempo_morto_mod, coefs[-1]),
                    )
                )
            elif tempo_morto_mod > 0:
                coefs_mod = np.concatenate(
                    (np.zeros(tempo_morto_mod), coefs[:-tempo_morto_mod])
                )
            else:
                coefs_mod = coefs
            coefs_mod = coefs_mod * ganho_mod

            mod_mdl.loc[
                (mod_mdl["output"] == model_index[0])
                & (mod_mdl["input"] == model_index[1]),
                "coef",
            ] = coefs_mod

            mod_mdl = mod_mdl[
                [
                    "ganho_mod",
                    "tempo_morto_mod",
                    "output",
                    "input",
                    "steps",
                    "coef",
                ]
            ]
        list_mod_mdl.append(mod_mdl)

mod_mdl = pd.concat(list_mod_mdl)
mod_filename = f"{model_filename_excel[:-5]}__mismatch_models.xlsx"
mod_mdl.to_excel(excel_dir % mod_filename, index=False)
