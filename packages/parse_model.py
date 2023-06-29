import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt


class MDLParser(object):
    def __init__(self):
        """Lê e salva as informações do arquivo MDL."""

    def extract_model(self, filename_mdl, filename_csv):
        cvs_name = []
        mvs_name = []
        outputs = []
        inputs = []
        steps = []
        """Realiza o parsing do arquivo MDL para extrair as informações."""
        with open(filename_mdl, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            model_name = lines[0]
            n_mvs = int(lines[2])  # numero de MVs
            n_cvs = int(lines[3])  # numero de CVs
            n_coefs = int(
                lines[4]
            )  # numero de coeficientes até o estado estacionário
            h = int(
                float(lines[6][:-2])
            )  # tempo necessário para chegar no estado estacionário
            line_idx = 8 + n_mvs + n_cvs  #
            data_coefs = pd.DataFrame()
            coefs_list = []
            for cv in range(n_cvs):
                cv_name = lines[line_idx].split(" ")[0]  # nome da CV
                cvs_name.append(cv_name)
                line_idx += 11
                for mv in range(n_mvs):
                    mv_name = lines[line_idx].split(" ")[0]  # nome da MV
                    if len(mvs_name) < n_mvs:
                        mvs_name.append(mv_name)
                    line_idx += 1
                    data = []
                    for line in lines[line_idx : line_idx + n_coefs // 5]:
                        line = line.split(" ")
                        line = [x for x in line if x.strip()]
                        line = np.array(line).astype(np.float64)
                        data.append(line)
                    data = np.array(data).reshape(
                        n_coefs,
                    )
                    coefs_list.append(data)
                    line_idx += n_coefs // 5
                    outputs += [cv_name for i in range(n_coefs)]
                    inputs += [mv_name for i in range(n_coefs)]
                    steps += range(n_coefs)

            data = np.array(coefs_list).reshape(n_coefs * n_cvs * n_mvs)
            data_coefs["output"] = outputs
            data_coefs["input"] = inputs
            data_coefs["steps"] = steps
            data_coefs["coef"] = data
            data_coefs.to_excel(filename_csv, index=False)

    def plot_model(self, df, save=True):
        """Plota os coeficientes do arquivo MDL"""
        cols = df.columns

        fig, ax = plt.subplots(self.n_cvs, self.n_mvs)
        fig.set_size_inches((22, 10))

        for i in range(self.n_cvs):
            for j in range(self.n_mvs):
                idx = i * self.n_cvs + j
                cv_name, mv_name = cols[idx]
                ax[i][j].plot(range(self.n_coefs), df[(cv_name, mv_name)])
                ax[i][j].tick_params(axis="both", which="major", labelsize=6)
                if i == 0:
                    ax[i][j].set_title(mv_name)
                if j == 0:
                    ax[i][j].set_ylabel(cv_name)
        if save:
            plt.savefig(self.filename_plot)
            plt.show()


# filename_mdl = (
#     "03__modelagem/2023-04-05__kickoff/modelos/REFAP/cmvu300_depr.mdl"
# )
# filename_csv = (
#     "03__modelagem/2023-04-05__kickoff/modelos/REFAP/coefs_model.csv"
# )
# filename_plot = "03__modelagem/2023-04-05__kickoff/modelos/resultados/REFAP/coefs_model_plot.png"

# model = Model(filename_mdl, filename_csv, filename_plot)
# model.extract_model()
# model.load_model()
# model.plot_model(save=True)
