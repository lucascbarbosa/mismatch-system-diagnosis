import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


######### 
# Paths #
models_dir = "modelos/excel/"
results_dir = "resultados/"
output_var = "CV_1"
#############
# Functions #

def split_data(df, test_size=0.2):
    # Assuming 'df' is your DataFrame with inputs and outputs
    X = df.drop(['ganho_mod', 'tempo_morto_mod'], axis=1).to_numpy() # Input features
    y_ganho = df['ganho_mod'].to_numpy()  # Output targets
    y_tempo_morto = df['tempo_morto_mod'].to_numpy()  # Output targets
    # Split the data into training and testing sets
    X_train, X_test, y_ganho_train, y_ganho_test = train_test_split(
        X, 
        y_ganho, 
        test_size=test_size, 
        random_state=42
    )
    X_train, X_test, y_tempo_morto_train, y_tempo_morto_test = train_test_split(
        X, 
        y_tempo_morto, 
        test_size=test_size, 
        random_state=42
    )
    return X_train, X_test, y_ganho_train, y_ganho_test, y_tempo_morto_train, y_tempo_morto_test 

def run_experiment(model, X_train, X_test, y_train, y_test):
    metrics = {}
    # Train the model for ganho
    model.fit(X_train, y_train)

    # Make predictions on ganho test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store the metrics in the dictionary
    metric = {'Mean Squared Error': mse, 'Mean Absolute Error': mae, 'R-squared': r2}

    return model, metric

def select_model(models, metrics, weights):
    def calculate_score(metrics, weights):
        score = 0
        for metric, weight in weights.items():
            score += metrics[metric] * weight
        return score   

    best_model = max(metrics, key=lambda x: calculate_score(metrics[x], weights))
    return models[best_model], calculate_score(metrics[best_model], weights)

# Load data
metrics_df = pd.read_excel(results_dir + "metricas.xlsx")
metrics_df = metrics_df[metrics_df['output']==output_var]
metrics_df.drop('output', axis=1, inplace=True)

# Prepare data
metrics_df = metrics_df[list(metrics_df.columns[2:]) + list(metrics_df.columns[:2])]
test_size = 0.3
X_train, X_test, y_ganho_train, y_ganho_test, y_tempo_morto_train, y_tempo_morto_test = split_data(metrics_df, test_size=test_size)


# Create a dictionary to store the models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regression': RandomForestRegressor(),
    'Gradient Boosting Regression': GradientBoostingRegressor(),
    'Support Vector Regression': SVR(),
    'Neural Network': MLPRegressor()
}

models_ganho = {}
models_tempo_morto = {}

metrics_ganho = {}
metrics_tempo_morto = {}

for model_name, model in models.items():
    model_ganho, metric_ganho = run_experiment(model, X_train, X_test, y_ganho_train, y_ganho_test)
    models_ganho[model_name] = model_ganho
    metrics_ganho[model_name] = metric_ganho
   
    model = models[model_name]

    model_tempo_morto, metric_tempo_morto = run_experiment(model, X_train, X_test, y_tempo_morto_train, y_tempo_morto_test)
    models_tempo_morto[model_name] = model_tempo_morto
    metrics_tempo_morto[model_name] = metric_tempo_morto


# Select best
weights = [0.2, 0.2, 0.6]
weights = dict(zip(metrics_ganho[model_name].keys(), weights))
best_model_ganho, score_ganho = select_model(models_ganho, metrics_ganho, weights)
best_model_tempo_morto, score_tempo_morto = select_model(models_tempo_morto, metrics_tempo_morto, weights)