import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv('../data/scotch_csv.csv', sep=";")
df_name = df['NAME']
df_region = df['REGION']
df.drop(['NAME', 'name1', 'REGION', 'DISTRICT', 'AGE', 'SCORE', '%'], axis=1, inplace=True)

pd.set_option('display.max_columns', None)
print(df)

'''
# elbow method to choose number of clusters 
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
clusters = range(1, 10)
for c in clusters:
    # Building and fitting the model
    kmeans = KMeans(n_clusters=c).fit(df)
    kmeans.fit(df)

    distortions.append(sum(np.min(cdist(df, kmeans.cluster_centers_,
                                        'euclidean'), axis=1)) / df.shape[0])
    inertias.append(kmeans.inertia_)

    mapping1[c] = sum(np.min(cdist(df, kmeans.cluster_centers_,
                                   'euclidean'), axis=1)) / df.shape[0]
    mapping2[c] = kmeans.inertia_

plt.plot(clusters, distortions, 'bx-')
plt.xlabel('v')
plt.ylabel('Distortion')
plt.title('Distortion')
plt.show()


plt.plot(clusters, inertias, 'bx-')
plt.xlabel('clusters')
plt.ylabel('Inertia')
plt.title('Inertia')
plt.show()
'''


kmeans = KMeans(
    init="random",
    n_clusters=5,
    n_init=10,
    max_iter=500,
    random_state=42 )

kmeans.fit(df)

prediction = kmeans.labels_
prediction_df = pd.DataFrame(data=prediction, columns=['kmeans_cluster'])
df_full = pd.concat([df_name, df, df_region, prediction_df], axis=1)
print(df_full)

print(df_full['kmeans_cluster'].value_counts())
print(df_full['DIST'].value_counts())

# clusters are almost equal
df_full.kmeans_cluster.value_counts().plot(kind='barh')
plt.show()

#save predictions
#df_full.to_csv('predictions.csv')

print(df_full.loc[(df_full['kmeans_cluster'] ==0)]['DIST'].value_counts())
#print(df_full.loc[(df_full['kmeans_cluster'] ==1)]['DIST'].value_counts())

print(df_full.loc[(df_full['kmeans_cluster'] ==0)]['REGION'].value_counts())
#print(df_full.loc[(df_full['kmeans_cluster'] ==1)]['REGION'].value_counts())


# plot pie chart to visualize frequency
for c in range(5):
    df_full.loc[(df_full['kmeans_cluster'] ==c)]['DIST'].value_counts().plot.pie(autopct='%.0f%%')
    plt.legend()
    plt.title('cluster: ' + str(c))
    plt.show()

for c in range(5):
    df_full.loc[(df_full['kmeans_cluster'] ==c)]['REGION'].value_counts().plot.pie(autopct='%.0f%%')
    plt.legend()
    plt.title('cluster: ' + str(c))
    plt.show()




