import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from matplotlib import pyplot as plt

from scipy.cluster.hierarchy import dendrogram

df = pd.read_csv("processed.cleveland.data", sep=",", decimal=".")
df = df.drop(["thal", "ca"], axis=1)

d = pdist(df, metric='euclidean')

my_cluster = linkage(d, 'ward')

dn = dendrogram(my_cluster)

plt.show()