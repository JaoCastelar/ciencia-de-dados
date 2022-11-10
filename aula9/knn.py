import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

##################################### Organizando o dataframe #####################################
pesquisa = pd.read_csv("C:/Users/Castelar/Desktop/Faculdade/6Período/Ciência de dados/aula9/bank.csv", sep=";")

df = pesquisa.drop(['age', 'balance', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous'], axis=1)

##################################### Separando treinamento e teste #####################################
treinamento = df[0:3021]

teste = df[3021:]

rotulo = treinamento.get('y')

treinamento = treinamento.drop(['y'], axis=1)

##################################### Convertendo string em int #####################################
X_test = pd.get_dummies(treinamento,drop_first=True)

##################################### Treinando o modelo #####################################
classificador = KNeighborsClassifier(n_neighbors = 8)
classificador.fit(X_test,rotulo)

##################################### Fazendo teste #####################################
teste = teste.drop(['y'], axis=1)
X_train = pd.get_dummies(teste,drop_first=True)

y_true = df.y[3021:]
y_pred = classificador.predict(X_train)

##################################### Matriz de confusão #####################################
mcNB = confusion_matrix(y_true, y_pred)

print(mcNB)

acur = (mcNB[0,0] + mcNB[1,1]) / len(y_true)

print("Acurácia knn: ",acur*100, "%")