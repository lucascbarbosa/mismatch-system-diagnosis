import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#########
# Paths #
results_dir = "03__modelagem/2023-06-21__simulacao/modelos/resultados/Livia/"


#############
# Plots #
def plot_metric(
    metric_df: pd.Series,
    metric_index: str,
    output_var: str,
    dir_plot: str,
    save_plot: bool = False,
):
    "Plota métrica específica para uma determinada saída"
    metric_df = metric_df[metric_df["output"] == output_var]
    heatmap_data = metric_df.pivot_table(
        index="tempo_morto_mod", columns="ganho_mod", values=metric_index
    )
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(heatmap_data, cmap="magma", annot=True)
    plt.title(f"{metric_index} para previsão de {output_var}")
    plt.tight_layout()

    if save_plot:
        plt.savefig(
            dir_plot + f"metricas__livia__{output_var}__{metric_index}.png"
        )
    plt.show()


metrics_df = pd.read_excel(results_dir + "livia_metricas.xlsx")

# Carrega dados de simulacao
plot_metric(metrics_df, "r2", "CV_1", results_dir, True)
