import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#########
# Paths #
models_dir = "03__modelagem/2023-06-21__simulacao/modelos/Livia/excel/"
results_dir = "03__modelagem/2023-06-21__simulacao/modelos/resultados/Livia/"


#############
# Functions #
def plot_simulation(
    u: np.array,
    y_real: np.array,
    y_preds: list,
    cases: list,
    filename: str,
    save_plot=False,
):
    """Plota gráfico da simulação de todos os cenarios"""
    fig_outputs = plt.figure(figsize=(10, 10))
    ax_outputs = fig_outputs.add_subplot(111)
    ax_outputs.set_title("Previsão C5_GLP x TEMPERATURA")
    ax_outputs.plot(y_real, label="Observação")
    palette = sns.color_palette("magma", n_colors=len(y_preds) + 1)
    for i in range(len(y_preds)):
        y_pred = y_preds[i]
        case = cases[i]
        ax_outputs.plot(
            y_pred, linestyle="--", label=f"Previsão {case}", color=palette[i]
        )
        ax_outputs.set_xlabel("t")
    ax_outputs.legend()

    # fig_inputs = plt.figure(figsize=(10, 10))
    # ax_inputs = fig_inputs.add_subplot(111)
    # ax_inputs.set_title("MV TEMPERATURA para CV C5_GLP")
    # ax_inputs.plot(u)
    # ax_inputs.set_xlabel("t")

    # n_cvs = y_real.shape[1]
    # n_mvs = u.shape[1]
    # fig_outputs, axs = plt.subplots(n_cvs, 1)
    # fig_outputs.set_size_inches((10, 10))
    # for i in range(n_cvs):
    #     output_var = cv_real.columns[i]
    #     axs[i].set_title(f"Previsão da CV {output_var}")
    #     axs[i].set_xlabel("k")
    #     axs[i].plot(y_real.iloc[:, i], label="Observação")
    #     for j in range(len(y_preds)):
    #         y_pred = y_preds[j]
    #         case = cases[j]
    #         axs[i].plot(
    #             y_pred.iloc[:, i], label=f"Previsão {case}", linestyle="--"
    #         )
    #     axs[i].legend()

    fig_outputs.tight_layout()
    # fig_inputs.tight_layout()

    if save_plot:
        fig_outputs.savefig(filename + "__outputs.png")
        # fig_inputs.savefig(filename + "__inputs.png")

    plt.show()


# List inputs and outputs
list_output_var = ["CV_1"]
list_input_var = ["MV_1"]

# Carrega dados de simulacao
simulation_data = pd.read_excel(results_dir + "simulacao_livia.xlsx")
cv_real = simulation_data[list_output_var]
mv_real = simulation_data[list_input_var]
cases = list(
    set(simulation_data.columns).difference(list_output_var + list_input_var)
)
ganho_mods = sorted(
    list(set([float(case.split("__")[1].split("_")[1]) for case in cases]))
)

tempo_morto_mods = sorted(
    list(set([float(case.split("__")[2].split("_")[1]) for case in cases]))
)

cv_preds = []
for ganho_mod in ganho_mods:
    cases_cols = [
        case
        for case in cases
        if float(case.split("__")[1].split("_")[1]) == ganho_mod
    ]
    cv_preds_case = simulation_data[cases_cols].to_numpy()
    cv_preds.append(cv_preds_case.mean(axis=1))

cases_ganho = [f"G={ganho_mod}" for ganho_mod in ganho_mods]
filename_ganho = results_dir + "simulacao_livia__ganho"
plot_simulation(
    mv_real,
    cv_real,
    cv_preds,
    cases_ganho,
    filename_ganho,
    save_plot=True,
)
plt.close()
cv_preds = []
for tempo_morto_mod in tempo_morto_mods:
    cases_cols = [
        case
        for case in cases
        if float(case.split("__")[2].split("_")[1]) == tempo_morto_mod
    ]
    cv_preds_case = simulation_data[cases_cols].to_numpy()
    cv_preds.append(cv_preds_case.mean(axis=1))

cases_tempo_morto = [
    f"TM={tempo_morto_mod}" for tempo_morto_mod in tempo_morto_mods
]
filename_tempo_morto = results_dir + "simulacao_livia__tempo_morto"
plot_simulation(
    mv_real,
    cv_real,
    cv_preds,
    cases_tempo_morto,
    filename_tempo_morto,
    save_plot=True,
)
