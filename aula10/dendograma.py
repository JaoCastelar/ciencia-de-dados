import pandas as pd

# from sklearn.datasets import load_iris

# iris = load_iris()
# X,y = load_iris(return_X_y = True)#X irá conter as features e y irá conter os rótulos

df = pd.read_csv("processed.cleveland.data", sep=",")
print(df)

from scipy.spatial.distance import pdist

d = pdist(df, metric='euclidean')


from scipy.cluster.hierarchy import linkage

# my_cluster = linkage(d)


from matplotlib import pyplot as plt

from scipy.cluster.hierarchy import dendrogram

# dn = dendrogram(my_cluster)

# plt.show()

my_cluster = linkage(d, 'ward')

dn = dendrogram(my_cluster)

plt.show()