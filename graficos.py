import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from regressao import reg_linear
import statsmodels.api as sm

def plot_predicao_vs_obs(objeto):
    
    resultado, residuos, y_pred, y_obs, _ = objeto
    
    y_pred = y_pred
    y_obs = y_obs
    
    valores_x = np.linspace(min(y_obs), max(y_obs), 1000)
    valores_y = valores_x
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=y_obs, color='blue', label='Valores Preditos vs Observados')
    sns.lineplot( x = valores_x, y = valores_y, color='red', label='y = x')
    plt.title('Valores Preditos vs Valores Observados')
    plt.xlabel('Valores Preditos')
    plt.ylabel('Valores Observados')
    plt.legend(loc = "upper left")
    plt.grid(True)
    plt.show()


def plot_residuos(objeto):
    residuos = objeto[1]
    indices_x = range(len(residuos.residuos))
    valores_residuos = residuos.residuos
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=indices_x, y=valores_residuos, color='blue', label='Resíduos')
    plt.axhline(y=3, color='red')
    plt.axhline(y=-3, color='red')
    plt.title('Gráfico de Resíduos')
    plt.xlabel('Observações')
    plt.ylabel('Resíduos')
    plt.legend(loc = "upper left")
    plt.grid(True)
    plt.show()

def plot_qqplot(objeto):
    residuos = objeto[1]
    residuos = residuos.residuos
    quantis = sm.ProbPlot(residuos)
    sm.qqplot(residuos, line ='q', alpha=0.5, markersize=5)
    #plt.plot([np.min(residuos), np.max(residuos)], [np.min(residuos), np.max(residuos)], color = "red")
    plt.title('Q-Q Plot')
    plt.xlabel('')
    plt.ylabel('')
    plt.grid(True)
    plt.show()