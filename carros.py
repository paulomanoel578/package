import seaborn as sns
from regressao import *
from graficos import *
from residuos_teste import *


data = sns.load_dataset("mpg")
data = data.dropna()
print(data)

independentes = data[["cylinders", "horsepower", "weight", "acceleration", "model_year"]]
dependentes = data["mpg"]

modelo = reg_linear(x=independentes, y=dependentes)

print(modelo[0])

teste_autocorrelacao_residuos(modelo)
teste_homocedasticidade(modelo)
teste_multicolinearidade(modelo)
teste_normalidade_residuos(modelo)


plot_predicao_vs_obs(modelo)
plot_residuos(modelo)
plot_qqplot(modelo)