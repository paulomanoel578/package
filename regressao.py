import numpy as np  
import pandas as pd
      
class Resumo:
    def __init__(self, beta, r_squared, adj_rsquared, equacao):
        self.coeficiente = beta
        self.r_quadrado = r_squared
        self.r_quadrado_ajustado = adj_rsquared
        self.equacao = equacao
        
    def __str__ (self):
        return f"O valor dos Coeficientes da Regressão é: {self.coeficiente}, \nO valor do R-quadrado é: {self.r_quadrado}, \nO valor do R-Quadrado Ajustado é de: {self.r_quadrado_ajustado}, \nA equação da reta é dada por: {self.equacao}\n"
    
        

class Residuos:
    def __init__ (self, residuos):
        self.residuos = residuos
    
    def __str__ (self):
        return f"Os resíduos da regressão linear é: {self.residuos}\n"
    
    def get_residuos (self):
        return self.residuos_lista

def reg_linear(x, y, alpha = 0):
    
    if isinstance(x, pd.DataFrame):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values
    
    
    
    n, p = x.shape
    # Adicionando uma coluna de 1s para o termo constante
    x = np.column_stack((np.ones(x.shape[0]), x))    
    # Calculando os coeficientes
    beta = np.linalg.inv(x.T.dot(x) + alpha *np.identity(p+1)).dot(x.T).dot(y)
    # Calculando a previsão
    y_pred = x.dot(beta)

    # Calculando o R-quadrado
    y_mean = np.mean(y)
    total_sum_squares = np.sum((y - y_mean) ** 2)
    residual_sum_squares = np.sum((y - y_pred) ** 2)
    r_squared = 1 - (residual_sum_squares / total_sum_squares)
    
    residuos = y - y_pred
    
    equacao = "y = "
    for i in range(len(beta)):
        if i == 0:
            equacao += f"{beta[i]:.4f}"
        else:
            equacao += f" + {beta[i]:.4f} * x{i}"
    
    adj_r_squared = 1 - ((n - 1) / (n - p - 1))* (1 - r_squared)
    
    return  Resumo(beta=beta, r_squared=r_squared.round(4), adj_rsquared=adj_r_squared.round(4), equacao=equacao), Residuos(residuos=residuos), y_pred.astype(float), y.astype(float), x.astype(float)
