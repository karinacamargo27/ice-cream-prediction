import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Criar parser para argumentos de linha de comando
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Caminho do dataset")
args = parser.parse_args()

mlflow.set_experiment("IceCreamSales")

with mlflow.start_run():
    # Carregar dados do caminho recebido como argumento
    df = pd.read_csv(args.data_path)
    df['Temperatura'] = df['Temperatura'].astype(float)

    # Definir variáveis de entrada e saída
    X = df[['Temperatura']]
    y = df['Vendas']

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Criar e treinar o modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Fazer previsões
    y_pred = modelo.predict(X_test)

    # Avaliação do modelo
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Registrar métricas no MLflow
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)

    # Criar diretório de saída e salvar o modelo
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(modelo, "outputs/model.pkl")

    # Registrar modelo no MLflow
    input_example = X_test.iloc[:1]  
    mlflow.sklearn.log_model(modelo, "ice_cream_model", input_example=input_example)

    print(f"Modelo registrado no MLflow! MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}")
