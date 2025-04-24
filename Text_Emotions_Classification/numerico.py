import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Função principal para calcular a expectativa de vida
def calcular_expectativa_vida(file_path):
    # Carrega os dados do arquivo Excel
    data = pd.read_excel('C:/Users/pedro/OneDrive/Software Development/Portfólio/'
                         'Python Projects/Text_Emotions_Classification/expec_vida.xlsx')

    # Substitui vírgulas por pontos e converte colunas numéricas para float
    for col in ['Renda per Cápita (US$)', 'PIB (US$ bilhões)', 'População (milhões)']:
        data[col] = data[col].astype(str).str.replace(',', '').astype(float)

    # Separa as variáveis independentes (X) e a variável dependente (y)
    X = data[['Renda per Cápita (US$)', 'PIB (US$ bilhões)', 'População (milhões)']]
    y = data['Expectativa de Vida (anos)']

    # Divide os dados em conjuntos de treino e teste (45 para treino e 5 para teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=5/50, random_state=42)

    # Cria o modelo de regressão linear múltipla
    model = LinearRegression()

    # Treina o modelo com os dados de treino
    model.fit(X_train, y_train)

    # Aplica o modelo aos dados de teste para fazer previsões
    y_pred = model.predict(X_test)

    # Calcula o erro quadrático médio das previsões
    mse = mean_squared_error(y_test, y_pred)

    # Exibe os coeficientes da regressão e o erro
    print("Coeficientes da regressão:", model.coef_)
    print("Intercepto:", model.intercept_)
    print("Erro Quadrático Médio (MSE) no conjunto de teste:", mse)

    # Exibe as previsões e os valores reais
    resultados = pd.DataFrame({'País': data.iloc[X_test.index]['País'], 'Expectativa Real': y_test, 'Expectativa Prevista': y_pred})
    print(resultados)

# Faz upload do arquivo no ambiente do Colab
#from google.colab import files
#uploaded = files.upload()

file_path = ('C:/Users/pedro/OneDrive/Software Development/Portfólio/'
                         'Python Projects/Text_Emotions_Classification/expec_vida.xlsx')
# Calcular a expectativa de vida usando o arquivo fornecido
calcular_expectativa_vida(file_path)