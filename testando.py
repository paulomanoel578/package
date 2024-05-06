## Teste de função inicial 
from sklearn.datasets import load_iris
import pandas as pd
from regressao import reg_linear
from regressao import Resumo
import numpy as np
from graficos import *
from residuos_teste import *

   # Dados de exemplo
#x = np.array([[1, 2, 20, 56], [2, 3, 8, 49], [3, 20, 4, 50], [10, 5, 15, 78], [5, 120, 13, 40], [1, 34, 56, 99]])
#y = np.array([2, 3, 4, 5, 6, 10])

    # Realizando a regressão linear
#teste = reg_linear(x, y)

    # Resultados

#print(teste[0])

#plot_predicao_vs_obs(teste)

#plot_residuos(teste)


# Carregar o conjunto de dados Boston House Prices
iris = load_iris()

# Criar um DataFrame do Pandas
dados = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])

# Adicionar a variável dependente ao DataFrame
iris_numeric = dados[['sepal length (cm)', 'petal width (cm)', 'petal length (cm)']]

# Dividindo os dados em variáveis independentes (X) e dependente (y)
x = iris_numeric[['sepal length (cm)', 'petal width (cm)']]
y = iris_numeric['petal length (cm)']

teste2 = reg_linear(x, y)

print(teste2[0])

#print(teste2[1])

#teste[1]

plot_residuos(teste2)

plot_predicao_vs_obs(teste2)

#teste_homocedasticidade(teste)
#teste_multicolinearidade(teste)
#teste_normalidade_residuos(teste)
#teste_autocorrelacao_residuos(teste)


teste_homocedasticidade(teste2)
teste_multicolinearidade(teste2)
teste_normalidade_residuos(teste2)
teste_autocorrelacao_residuos(teste2)

plot_qqplot(teste2)
