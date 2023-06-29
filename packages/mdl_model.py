import os
import pandas as pd
import numpy as np
from scipy.linalg import block_diag


class MDLModel:
    """Faz a previsão utilizando os parametros do arquivo MDL."""

    def __init__(self, mdl_parameters: pd.DataFrame, output: str):
        """
        __init__.

        Args:
            mdl_parameters [pd.DataFrame]: Coeficientes do modelos MDL.
            output [str]: Coluna que vai ser considerada a saída do modelo.
            steps [int]: Horizonte de predição da saída do modelo.
        kwards:
            Sem kwards.
        Exceptions:
            Sem exceções.
        """
        mdl_parameters["steps"] = mdl_parameters["steps"].astype(int)

        # Criando a matrix com os coefs para os atrasos
        coef_matrix = pd.pivot_table(
            mdl_parameters, columns="input", index="steps", values="coef"
        )
        diff_values = coef_matrix.diff().fillna(value=0)
        diff_values = diff_values.unstack().reset_index()

        new_row = {"input": output, "steps": 1, 0: 1}
        diff_values = pd.concat(
            [diff_values, pd.DataFrame([new_row])], ignore_index=True
        )

        diff_values.rename(columns={0: "value"}, inplace=True)

        diff_values["coef_name"] = diff_values.apply(
            lambda x: MDLModel.create_shift_names(**x), axis=1
        )

        # Setting parameters
        self.mdl_parameters = mdl_parameters
        self.mdl_parameters_diff = diff_values
        self.output = output

    @staticmethod
    def create_shift_names(input, steps, **kwards):
        return "{input}__shift_{steps}".format(
            input=input, steps=str(int(steps)).zfill(5)
        )

    def build_input_matrix(self, database: pd.DataFrame):
        """Cria a matrix com os atrasos para aplicar os coefs do MDL."""
        melted_database = database.melt(id_vars=["time"])

        all_shifts_list = []
        for index, row in self.mdl_parameters_diff.iterrows():
            index_variable = melted_database["variable"] == row["input"]
            temp_var = melted_database[index_variable].copy()
            temp_var["value"] = temp_var["value"].shift(int(row["steps"]))
            temp_var["coef"] = self.create_shift_names(
                input=row["input"], steps=row["steps"]
            )
            temp_var.set_index(["time", "coef"], inplace=True)
            all_shifts_list.append(temp_var)

        all_shifts_pd = pd.concat(all_shifts_list)
        input_dataframe = all_shifts_pd["value"].unstack()
        input_dataframe.sort_index(inplace=True)
        input_dataframe.fillna(method="backfill", inplace=True)

        outputs_lags = self.mdl_parameters_diff[
            self.mdl_parameters_diff["input"] == self.output
        ]
        is_in_output = input_dataframe.columns.isin(outputs_lags["coef_name"])
        input_dataframe.loc[:, ~is_in_output] = input_dataframe.loc[
            :, ~is_in_output
        ].diff()

        return input_dataframe[self.mdl_parameters_diff["coef_name"]]

    def _predict_core(
        self,
        input_data: pd.DataFrame,
        is_forecast_series: np.array,
        max_horizon: int,
    ):
        # Check if there is variables of recorrence
        outputs_lags = self.mdl_parameters_diff[
            self.mdl_parameters_diff["input"] == self.output
        ]
        coef_value = self.mdl_parameters_diff.set_index("coef_name")["value"]

        lags_per_var = (
            self.mdl_parameters_diff.groupby("input").size().sort_index()
        )

        vars_ = list(lags_per_var.index)
        cols_ = ["forecast_h", "prediction"] + vars_
        prediction_df = pd.DataFrame(
            np.full((len(input_data), len(cols_)), np.nan),
            columns=cols_,
            index=input_data.index,
        )

        # Construindo matriz de blocos para calculo de efeitos
        p = len(vars_)
        r = lags_per_var.values
        ones = [np.ones([r[i], 1]) for i in range(p)]
        A = block_diag(*ones)

        i = 0
        temp_forecast_h = 1
        n_rows = len(input_data)

        while i < n_rows:
            row_data = input_data.iloc[i, :]
            prediction = np.inner(row_data, coef_value)
            effects_prediction = (row_data * coef_value).sort_index() @ A

            assert np.isnan(prediction) or np.isclose(
                prediction, np.sum(effects_prediction)
            ), "Erro na decomposição dos efeitos"

            # Preenchendo as variáveis de recorrencia
            for index, row in outputs_lags.iterrows():
                temp_i = i + row["steps"]
                if temp_i < n_rows:
                    lag_output_value = input_data.iloc[temp_i, index]
                    if pd.isnull(lag_output_value):
                        input_data.iloc[temp_i, index] = prediction

            # Colocando os valores de predição
            prediction_df.iloc[i, 0] = temp_forecast_h
            prediction_df.iloc[i, 1] = prediction
            prediction_df.iloc[i].loc[vars_] = effects_prediction

            # Atualizando o stepste de previsão para a próxima interação
            is_forecast = is_forecast_series.iloc[i]
            if is_forecast:
                temp_forecast_h += 1
            else:
                temp_forecast_h = 1
            i += 1

            # Pensar em uma solução melhor
            if temp_forecast_h > max_horizon:
                break

        prediction_df.reset_index(inplace=True)
        return prediction_df

    def predict(self, database: pd.DataFrame, max_horizon: int):
        input_data = self.build_input_matrix(database=database)
        # Check if is a forecast or in-sample prediction
        is_forecast_series = database[self.output].isna()
        is_forecast_series.index = database["time"].values
        is_forecast_series.sort_index(inplace=True)
        prediction_df = self._predict_core(
            input_data=input_data,
            is_forecast_series=is_forecast_series,
            max_horizon=max_horizon,
        )
        final_dataframe = database.merge(
            prediction_df,
            how="left",
            validate="1:1",
            on="time",
            suffixes=("", "_effect"),
        )

        return final_dataframe
