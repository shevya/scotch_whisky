# Analysis:
# 1. class distribution per region.
# The results show how the different groups are characterized:
#       class 0 - most records from regions islay and low
#       class 1 - there is no record form region islay
#       class 2 - the largest class, most records from region high
# cluster 0 contains records with the highest score
# cluster 1 contains records with the lowest score
#
# 2. map: map showing the location of each distillery, the colors correspond to the values of the kmeans cluster results
#   file is stored in the directory.. and named 'map_predicted.html'
#
# 3. feature importance (using feature importance from random forest and method 'select kbest')
#       The results for both methods are similar, the most importance features are: dist, score,
#       %, NOSE_FRESH, PAL_dry, NOSE_AROMA, BODY_firm, NOSE_RICH
#
# 4. Percentage share per cluster.
# The results show how the different groups are characterized:
# cluster 0   : NOSE_SWEET , NOSE_SEA, NOSE_RICH , NOSE_SHERRY, PAL_salt , FIN_long
#         1   : NOSE_LIGHT, NOSE_DRY, NOSE_FRUIT, BODY_soft, BODY_light, PAL_dry
#         2   : NOSE_AROMA, BODY_smooth, BODY_firm, PAL_smooth, PAL_fruit ,
#
# Next steps:
# -> refactoring
# -> create classification models that will allow adding new records to the input set e.g. KNN, SVM multiclass,
#   decision trees, gradient boosting trees
# -> create an application that will enable better and easier recommendations, e.g. on the basis
#   of selected tastes, regions etc.


import clustering
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
import folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


df = pd.read_csv('../data/scotch_csv.csv', sep=";")
df_name = df['NAME']
df_region = df['REGION']
df.drop(['NAME', 'name1', 'REGION', 'DISTRICT', 'AGE'], axis=1, inplace=True)

threshold_n = 0.95
sel = VarianceThreshold(threshold=(threshold_n * (1 - threshold_n)))
sel_var = sel.fit_transform(df)
df = df[df.columns[sel.get_support(indices=True)]]

df_coordinates = pd.read_csv('../data/geo.csv', sep=";")
df_coordinates.columns = ['NAME', 'longitude', 'latitude']

df = pd.concat([df, df_name], axis=1)
df_predicted = clustering.kmeans_result(df, 3)


# show cluster size
df_predicted.predicted_label.value_counts().plot(kind='barh')
plt.show()


df_predicted = pd.merge(df_predicted, df_coordinates, on=["NAME"])


# class distribution per region
for c in range(3):
    print(df_predicted.loc[(df_predicted['predicted_label'] ==c)]['REGION'].value_counts())
    df_predicted.loc[(df_predicted['predicted_label'] == c)]['REGION'].value_counts().plot(kind='barh')
    plt.title('cluster: ' + str(c))
    plt.show()



# map
map_predicted = folium.Map(location=[57.00, -2.80], zoom_start=7)

labels_region = df_predicted['REGION']
labels_dist = df_predicted['DIST']
labels_score = df_predicted['SCORE']
labels_predicted = df_predicted['predicted_label']
col =['orange','green', 'blue']

for l_region, l_dist, l_score, l_predicted, lon, lat, c in \
        zip(labels_region, labels_dist, labels_score, labels_predicted,
            df_predicted['longitude'], df_predicted['latitude'], df_predicted['predicted_label']):
    lon = float(lon)*(-1)
    folium.Marker([lat, lon], popup=('predicted class: ' + str(l_predicted).capitalize() + '<br>'
                'REGION: ' + str(l_region).capitalize() + '<br>'
                'DIST: ' + str(l_dist) + '<br>' 'SCORE: ' + str(l_score) + '<br>'),
                icon=folium.Icon(color=col[c])).add_to(map_predicted)

#map_predicted.save("../images/map_predicted.html")


X = df_predicted.loc[:,'color_p.gold':'lowland']
y = df_predicted['predicted_label']

# standarization
scaler = StandardScaler()
scaled = scaler.fit_transform(X)
df2 = scaled

# check feature importance
model = RandomForestClassifier()
model.fit(X, y)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.show()



best_features = SelectKBest(score_func=f_classif, k=20)
fit = best_features.fit(X, y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)
feature_scores = pd.concat([df_columns, df_scores],axis=1)
feature_scores.columns = ['Feature_Name', 'Score']  # name output columns
print(feature_scores.nlargest(15, 'Score'))  # print 20 best features


X = df_predicted.loc[:, 'color_p.gold':'predicted_label']
df_per_cluster = X.groupby(['predicted_label']).sum()


df_res = []
df_res = pd.DataFrame(df_res)
for x in df_per_cluster.columns:
    df_res_pre = (df_per_cluster[x] / X['predicted_label'].value_counts()) * 100
    df_res_pre = pd.DataFrame(df_res_pre)
    df_res = pd.concat([df_res, df_res_pre], axis=1)

pd.set_option('display.max_rows', None)

df_res.columns = df_per_cluster.columns
print(df_res.T)


df_res = []
df_res = pd.DataFrame(df_res)
for x in df_per_cluster.columns:
    df_res_pre = (df_per_cluster[x] / X[x].sum()) * 100
    df_res_pre = pd.DataFrame(df_res_pre)
    df_res = pd.concat([df_res, df_res_pre], axis=1)

df_res.columns = df_per_cluster.columns
print(df_res.T)



## models
# X = df_predicted.loc[:,'color_p.gold':'islands']
# y = df_predicted['predicted_label']
#
#
# #split dataset into train and test data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
#
# print('-------- KNN --------')
#
# knn = KNeighborsClassifier(n_neighbors = 5)
# knn.fit(X_train,y_train)
# print(knn.score(X_test, y_test))
# y_pred = knn.predict(X_test)
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
#
#
#
# knn_cv = KNeighborsClassifier(n_neighbors=5)
# cv_scores = cross_val_score(knn_cv, X, y, cv=10)
# #print(cv_scores)
# print('cv_scores mean:{}'.format(np.mean(cv_scores)))
# y_pred = cross_val_predict(knn_cv, X, y, cv=10)
# conf_mat = confusion_matrix(y, y_pred)
# print(conf_mat)
# print(classification_report(y, y_pred))


# print('-------- SVM --------')
#
# poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X, y)
# cv_scores = cross_val_score(poly, X, y, cv=10)
# print('cv_scores mean:{}'.format(np.mean(cv_scores)))
# y_pred = cross_val_predict(poly, X, y, cv=10)
# print(confusion_matrix(y, y_pred))
# print(classification_report(y, y_pred))



