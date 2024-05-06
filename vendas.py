from regressao import reg_linear
from graficos import *
from residuos_teste import *
import pandas as pd

url = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Advertising.csv"
vendas = pd.read_csv(url)

print(vendas.head())

independentes = vendas[["TV", "Radio", "Newspaper"]]
dependentes = vendas["Sales"]

modelo2 = reg_linear(x=independentes, y=dependentes)

print(modelo2[0])

teste_autocorrelacao_residuos(modelo2)
teste_homocedasticidade(modelo2)
teste_multicolinearidade(modelo2)
teste_normalidade_residuos(modelo2)


plot_predicao_vs_obs(modelo2)
plot_residuos(modelo2)
plot_qqplot(modelo2)