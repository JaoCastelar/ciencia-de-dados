# abrir o arquivo

import pandas as pd

df = pd.read_csv("C:/Users/Castelar/Desktop/Faculdade/6° Período/Ciência de dados/aula7/parkinsons.data", sep=";", decimal=",")

df = df['name,MDVP:Fo(Hz),MDVP:Fhi(Hz),MDVP:Flo(Hz),MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP,MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA,NHR,HNR,status,RPDE,DFA,spread1,spread2,D2,PPE'].str.split(',', expand=True)

df.columns = ['name', 'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'status', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']

#Criar o vetor com as classes

classes = pd.array(df.status[0:len(df.status)])

#carregar o pacote
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

treino = df[0:135]
treino = treino.drop(['name', 'status'], axis=1)

rotulo = pd.array(df.status[0:135])

teste = df[135:195]
teste = teste.drop(['name', 'status'], axis=1)

clf = GaussianNB() #criar o classificador
clf.fit(treino, rotulo)#treinar o classificador

y_true = df.status[135:195]
y_pred = clf.predict(teste)

#Criando matriz de confusão

from sklearn.metrics import confusion_matrix

mcNB = confusion_matrix(y_true, y_pred)

acur = (mcNB[0,0] + mcNB[1,1]) / len(y_true)

print("Acurácia nb: ",acur*100, "%")