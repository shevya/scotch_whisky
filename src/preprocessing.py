# Dataset:
# Dataset contains data from 109 distinct scottish distilleries. Dataset contains 68 columns describing
#   characteristic of the whiskies e.g. taste, smell, color etc.
#   (14 columns describing the color, 12 columns describing the smell, 8 body, 15 palate , 19 finish).
#   Besides those features, there are columns of distillery name,region, information about quality
#   (score (scotch score) and dist (distillery score)), latitude and longitude of the distilleries.
# The original dataset (scotch.csv) is in folder data in this project. Geolocation data are in file geo.csv in folder
#   named data. Datasets don't contain null values.

# The dataset used for the analysis is scotch_csv.csv which is a modification of the original
#   one: the first two rows have been merged to create unique column names.
#
# Data analysis steps:
# 1. Percentage share per region ->
# region high   : color_f.gold , NOSE_SWEET, BODY_smooth, PAL_fruit , PAL_sweet
#       islay   : NOSE_PEAT, NOSE_SEA, BODY_firm , BODY_oily, FIN_full
#       low     : color_gold, NOSE_GRASS, BODY_soft , BODY_light, PAL_light , PAL_grass , FIN_big
# 2. Remove features with variation below 95
# 3. Histograms of selected features (['REGION', 'DIST', 'SCORE', '%' ]). Conclusions:
#       large disparity in the number of collections per region
#       half of the records have a distillery score of 3
#       more than 1/3 of the records have a scotch score of about 75
#       3/4 of the records are drinks with 40% alcohol content
# 4. box plots of selected features -> two outliers for the score feature, most records in the 70-80 range
# 5. map showing the location of each distillery. The colors correspond to the values of the DIST column
#   File is stored in the directory images and named 'map_preprocessing.html'
# 6. correlation matrix
# the correlation coefficient is the highest between pairs: DIST - SCORE and latitude - spey that is obvious
# there is an interesting correlation between pairs: PAL_salt - FIN_salt, BODY_full - BODY_light, NOSE_SEA - islay
# the correlation coefficient is the worst between pairs: BODY_light - PAL_sweet and FIN_sweet - BODY_light



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import numpy as np
from sklearn.feature_selection import VarianceThreshold


# read csv
df_src = pd.read_csv('../data/scotch_csv.csv', sep=";")

# drop unused variables
df_src.drop(['AGE', 'name1'], axis=1, inplace=True)

# read file with geolocation info
df_coordinates = pd.read_csv('../data/geo.csv', sep=";")
df_coordinates.columns = ['NAME', 'longitude', 'latitude']

# merge into one dataframe
df_merged = pd.merge(df_src, df_coordinates, on=["NAME"])



# there are some variables that frequency of occurrence of the value one in the dataset is much higher
# e.g. NOSE_PEAT (50%) and color_yellow (0.2%)

#print(df_src.shape)
# shape: (109, 83)

#print(df_src.info())

# check if any null value
#print(df_src.isnull().values.any())
#print(df_coordinates.isnull().values.any())


## liczba 1 / udział / per region
df_2 = df_src.iloc[:,1:-10]

df_2.drop(['DIST', 'SCORE', '%'], axis=1, inplace=True)

df_per_region = df_2.groupby(['REGION']).sum()

df_res = []
df_res = pd.DataFrame(df_res)
for x in df_per_region.columns:
    df_res_pre = (df_per_region[x] / df_src['REGION'].value_counts()) * 100
    df_res_pre = pd.DataFrame(df_res_pre)
    df_res = pd.concat([df_res, df_res_pre], axis=1)


pd.set_option('display.max_rows', None)

df_res.columns = df_per_region.columns
print(df_res.T)




# remove features with variation below 95
df = df_src.copy()
# remove categorical features
df.drop(['REGION', 'NAME', 'DISTRICT'], axis=1, inplace=True)
threshold_n = 0.95
sel = VarianceThreshold(threshold=(threshold_n * (1 - threshold_n)))
sel_var = sel.fit_transform(df)
print(df[df.columns[sel.get_support(indices=True)]])



columns = ['REGION', 'DIST', 'SCORE', '%' ]

# histogram
fig, axes = plt.subplots(ncols=2, nrows=2)

for i, ax in zip(columns, axes.flat):
    sns.histplot(data = df_src[i].value_counts() , x = df_src[i], ax=ax)

#fig.savefig('../images/histogram.png')
plt.show()


fig, axes = plt.subplots(ncols=2, nrows=1)
columns = [ 'DIST', 'SCORE']
for i, ax in zip(columns, axes.flat):
    sns.boxplot(x=df_src[i] , ax=ax)
plt.show()
#fig.savefig('../images/boxplot.png')



df = df_merged.copy()

map = folium.Map(location=[57.00, -2.80], zoom_start=7)

labels_region = df['REGION']
labels_dist = df['DIST']
labels_score = df['SCORE']
col =['orange','green', 'blue', 'lightgray', 'yellow', 'black']

for l_region, l_dist, l_score, lon, lat, c in zip(labels_region, labels_dist, labels_score, df['longitude'], df['latitude'], df['DIST']):
    lon = float(lon)*(-1)
    folium.Marker([lat, lon], popup=('REGION: ' + str(l_region).capitalize() + '<br>'
                 'DIST: ' + str(l_dist) + '<br>' 'SCORE: ' + str(l_score) + '<br>'),
                  icon=folium.Icon(color=col[c])).add_to(map)

#map.save("../images/map_preprocessing.html")




corr = df.corr()
c1 = corr.abs().unstack()
print(c1.sort_values(ascending = False))

matrix = np.triu(df.corr())
sns.heatmap(df.corr(), annot=True, mask=matrix)
plt.show()





