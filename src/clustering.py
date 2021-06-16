
# Modelling
# The kmeans algorithm was used to cluster the input data. The number of clusters was selected based on
# the analysis of the results of two methods: elbow metod oraz silhouette score. Results are saved in folder ...
# Selection of 3 clusters in terms of the number of clusters. The set of classes are the most balanced.
#
# cluster_no    count
# 0             31
# 1             27
# 2             51


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from yellowbrick.cluster import SilhouetteVisualizer


df = pd.read_csv('../data/scotch_csv.csv', sep=";")
df_name = df['NAME']
df_region = df['REGION']
df.drop(['NAME', 'name1', 'REGION', 'DISTRICT', 'AGE'], axis=1, inplace=True)

df_coordinates = pd.read_csv('../data/geo.csv', sep=";")
df_coordinates.columns = ['NAME', 'longitude', 'latitude']

df = pd.concat([df, df_name], axis=1)

threshold_n = 0.95
sel = VarianceThreshold(threshold=(threshold_n * (1 - threshold_n)))
sel_var = sel.fit_transform(df)
df = df[df.columns[sel.get_support(indices=True)]]


def get_elbow_score(df):
    sum_squared_distances = []
    n_clusters = range(1, 15)
    for k in n_clusters:
        km = KMeans(n_clusters=k)
        km = km.fit(df)
        sum_squared_distances.append(km.inertia_)


    fig = plt.figure(figsize=(10, 6))
    plt.plot(n_clusters, sum_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    #fig.savefig('../images/elbow_method.png', dpi=fig.dpi)


def get_sillhoette_score(df):
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    for i in [2, 3, 4, 5]:
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
        q, mod = divmod(i, 2)
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q - 1][mod])
        visualizer.fit(df)

        cluster_labels = km.fit_predict(df)
        silhouette_avg = silhouette_score(df, cluster_labels)
        print("For n_clusters =", i,
              "The average silhouette_score is :", silhouette_avg)
        df['predicted_label'] = km.labels_
        print(df['predicted_label'].value_counts())

    plt.show()
    #plt.savefig('../images/silhoette_score.png')

# get_sillhoette_score(df)
# get_elbow_score(df)


# pca = PCA().fit(df)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.show()
#
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(df)
#
# pca = PCA(n_components=0.95)
# components = pca.fit_transform(scaled_data)
#
# total_var = pca.explained_variance_ratio_.sum() * 100
# print(total_var)
#
# print(pca.explained_variance_ratio_* 100)
# print(len(pca.explained_variance_ratio_))




def kmeans_result(df, n_clusters):
    df_name = df['NAME']
    df = df.iloc[:, 1:-1]

    model = KMeans(n_clusters, init='random', n_init=10, max_iter=100, random_state=42).fit_predict(df)

    df['predicted_label'] = model.labels_
    df_name = pd.DataFrame(df_name)
    df_2 = pd.concat([df_name, df, df_region], axis=1)

    #df_2.to_csv('../images/kmeans_results.csv', index=False)
    return df_2


df_predicted = kmeans_result(df, 3)

# show cluster size
df_predicted.predicted_label.value_counts().plot(kind='barh')
plt.show()


df = pd.merge(df_predicted, df_coordinates, on=["NAME"])


def get_recommendation(df):
    df_name = df['NAME'].str.lower()
    print('podaj nazwe whisky')
    input1 = input()
    input1 = input1.lower()

    # top five recommendation based on score
    if df_name.isin([input1.lower()]).any():
        label = df.predicted_label[df['NAME'].str.lower() == input1]
        label = int(label)

        df_label = df[df['predicted_label'] == label]

        #print(df_label.nlargest(5, 'DIST'))
        print(df_label.nlargest(5, 'SCORE'))

    else:
        print("Selected whisky name does not exists")


# get_recommendation(df)

