import joblib
import pandas as pd

# Modelo treinado
model = joblib.load("src/model/ice_cream_model.pkl")

# Previsão
def prever_vendas(temperatura):
    dados = pd.DataFrame({"Temperatura": [temperatura]})
    previsao = model.predict(dados)
    return previsao[0]

# Teste
if __name__ == "__main__":
    temp = float(input("Digite a temperatura: "))
    print(f"Previsão de vendas: {prever_vendas(temp):.2f} sorvetes")
