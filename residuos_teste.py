import numpy as np
import pandas as pd
from scipy.stats import kstest
import statsmodels.api as sm
import statsmodels.stats.api as sms
from regressao import reg_linear
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import reset_ramsey


# Teste de Normalidade - Via Kolmogorov Smirnov
def teste_normalidade_residuos(objeto):
    resultados, residuos, y_pred, y_obs, x  = objeto
    _, p_valor = kstest(residuos.residuos, "norm")
    print(f"\n\nTeste de Normalidade dos Resíduos - Kolmogorov Smirnov")
    print(f"A hipótese nula é de que os resíduos estão distribuídos normalmente")
    if p_valor > 0.05:
        print(f"\033[0;32mOs resíduos não violam a suposição de Normalidade.\033[m\n Com resultado de p-valor de {p_valor.round(8)}")
    else:
        print(f"\033[0;31mExiste evidências significativas que os resíduos violam a suposição de normalidade. \033[m\nCom resultado de p-valor: {p_valor.round(8)}")

# Teste de Homocedasticidade - via Breusch-Pagan

def teste_homocedasticidade(objeto):
    resultados, residuos, y_pred, y_obs, x = objeto
    bp_teste = sms.het_breuschpagan(residuos.residuos, np.column_stack((np.ones(len(y_pred)), y_pred)))
    print(f"\n\nTeste de Homocedasticidade dos Resíduos - Breusch Pagan")
    print(f"A hipótese nula é de que a variância dos resíduos é constante")
    if bp_teste[1] > 0.05:
        print(f"\033[0;32mOs resíduos não violam a suposição de Homocedasticidade.\033[m\nCom resultado de p-valor de {bp_teste[1].round(8)}")
    else:
        print(f"\033[0;31mExiste evidências de que os resíduos violam a suposição de homocedasticidade. \033[m\nCom resultado de p-valor: {bp_teste[1].round(8)}")

#teste de Multicolinearidade - via  número de condições

def teste_multicolinearidade(objeto):
    resultados, residuos, y_pred, y_obs, x = objeto

    condicao = np.linalg.cond(x)
    print(f"\n\nTeste de Multicolinearidade dos Resíduos - Número de Condições")
    if condicao < 100:
        print(f"\033[0;32mOs resíduos não violam a suposição de Multicolinearidade.\033[m\nCom resultado de: {condicao.round(2)}, sendo menor do que 100. O limite estabelecido")
    elif condicao >= 100 and condicao <= 1000:
        print(f"\033[0;31mExiste evidências significativas que os resíduos violam a suposição de Multicolinearidade, tendo multicolinearidade moderada. \033[m\nCom resultado de: {condicao.round(2)}, sendo maior do que 30. O limite estabelecido\n")
    else:
        print(f"\033[0;31mExiste evidências significativas que os resíduos violam a suposição de Multicolinearidade, tendo multicolinearidade forte. \033[m\nCom resultado de: {condicao.round(2)}, sendo maior do que 30. O limite estabelecido\n")

# Teste de AutoCorrelação - via Durbin Watson

def teste_autocorrelacao_residuos(objeto, lag = None):
    resultados, residuos, y_pred, y_obs, x = objeto
    residuos = residuos.residuos.flatten()
    estatistica_dw = durbin_watson(residuos)
    print(f"\n\nTeste de Autocorrelação dos Resíduos - Durbin Watson")
    if estatistica_dw > 1.5 and estatistica_dw < 2.5:
       print(f"\033[0;32mOs resíduos não violam a suposição de Autocorrelação.\033[m\nCom valor de estatística DW de: {estatistica_dw.round(8)}")
    else:
       print(f"\033[0;31mExiste evidências significativas que os resíduos violam a suposição de Autocorrelação. \033[m\nCom valor de estatística DW de: {estatistica_dw.round(8)}")
