# abrir o arquivo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

df = pd.read_csv("C:/Users/Castelar/Desktop/Faculdade/6° Período/Ciência de dados/aula8/regressao.csv", sep=";", decimal=".")
df = df['horas, notas'].str.split(',', expand=True)
df.columns = ['horas', 'notas']

print(df)

# Criar o conjunto de dados de treinamento

treinamento = df.drop(['notas'], axis=1)
print("\nTreino:\n", treinamento)

#Criar o vetor com as classes

y_Train = pd.array(df.notas[0:len(df.notas)])
print("\nArray com notas:\n", y_Train)
res = y_Train.astype(np.float) 

#carregar o pacote

from sklearn.linear_model import LinearRegression

#Converter os dados categoricos em valores inteiros

X_train = pd.get_dummies(treinamento,drop_first=True)
print(X_train)

#Separando dados de treino e de teste
#utilizamos 70% dos dados para treino e o restante (30%) para teste.
x_train, x_test, y_train, y_test = train_test_split( treinamento, y_Train, test_size=0.15)

#Precisamos redimensionar os dados para fazer a regressão linear
x_train=x_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)
y_test=y_test.reshape(-1,1)

#chamada a funcao LinearRegression

clf = LinearRegression() #criar o classificador
treino = clf.fit(x_train, y_train)#treinar o classificador

pred = clf.predict(y_test)


#Visualizar os coeficientes

print("\nIntercept: ", clf.intercept_)

coef = clf.coef_
print("\nCoeficiente: ", coef)

plt.scatter(treinamento, res, color="blue")
plt.plot(x_test, pred, color="red")
plt.title("Regressão linear")
plt.xlabel("Horas")
plt.ylabel("Notas")
plt.show()