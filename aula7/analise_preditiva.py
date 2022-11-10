import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

pesquisa = pd.read_csv("C:/Users/Castelar/Desktop/Faculdade/6° Período/Ciência de dados/aula7/parkinsons.data", sep=",", decimal=".")

df = pesquisa.drop(['name'], axis=1)

treinamento = df[0:135]

teste = df[135:195]

rotulo = treinamento.get('status')

treinamento = treinamento.drop(['status'], axis=1)

#Treinando o modelo

classificador = KNeighborsClassifier(n_neighbors = 33)
classificador.fit(treinamento,rotulo)

#Fazendo teste

teste = teste.drop(['status'], axis=1)

#Criando matriz de confusão

y_true = df.status[135:195]
y_pred = classificador.predict(teste)

from sklearn.metrics import confusion_matrix

mcNB = confusion_matrix(y_true, y_pred)

acur = (mcNB[0,0] + mcNB[1,1]) / len(y_true)

print("Acurácia knn: ",acur*100, "%")