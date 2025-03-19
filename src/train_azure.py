import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from azureml.core import Run
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Captura o contexto de execução do Azure ML
run = Run.get_context()

# Configurar o MLflow para usar o tracking URI correto no Azure ML
mlflow.set_tracking_uri(run.experiment.workspace.get_mlflow_tracking_uri())
mlflow.set_experiment(run.experiment.name)

# Argumentos para entrada de dados no Azure ML
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

# Corrigir o caminho do arquivo CSV
data_file = os.path.join(args.data_path, "ice_cream_sales.csv")  # Nome exato do arquivo
if not os.path.isfile(data_file):
    raise FileNotFoundError(f"Arquivo não encontrado: {data_file}")

# Carregar os dados
df = pd.read_csv(data_file)
df['Temperatura'] = df['Temperatura'].astype(float)

# Variáveis
X = df[['Temperatura']]
y = df['Vendas']

# Dados (treino e teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Previsões
y_pred = modelo.predict(X_test)

# Avaliação do modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Log das métricas no Azure ML
run.log("MAE", mae)
run.log("MSE", mse)
run.log("R2", r2)

# Salvar o modelo no MLflow dentro do Azure ML
with mlflow.start_run():
    mlflow.sklearn.log_model(modelo, "model")
    model_uri = mlflow.get_artifact_uri("model")

# Registrar o modelo no workspace do Azure ML
mlflow.register_model(model_uri, "ice-cream-sales-model")

print(f"Modelo treinado no Azure ML! MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}")
