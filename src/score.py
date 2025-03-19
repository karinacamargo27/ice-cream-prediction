import json
import mlflow
import pandas as pd

def init():
    global model
    model_path = mlflow.pyfunc.get_model_uri("models:/ice-cream-sales-model/1")  # Use a versão correta!
    model = mlflow.pyfunc.load_model(model_path)

def run(data):
    try:
        input_data = pd.DataFrame(json.loads(data))
        predictions = model.predict(input_data)
        return predictions.tolist()
    except Exception as e:
        return str(e)
