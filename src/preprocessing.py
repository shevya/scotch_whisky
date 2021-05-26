import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans

# read csv
df_src = pd.read_csv('../data/scotch_csv.csv', sep=";")

#drop unused variables
df_src.drop(['DISTRICT', 'AGE'], axis=1, inplace=True)

# read coordinates
df_coordinates = pd.read_csv('../data/geo.csv', sep=";")

# there are some variables that frequency of occurrence of the value one in the dataset is much higher
# e.g. NOSE_PEAT (50%) and color_yellow (0.2%)
print(df_src.describe())

# check if any null value
print(df_src.isnull().values.any())

df_coordinates.columns = ['NAME', 'longitude', 'latitude']
print(df_coordinates)

# merge into one dataframe
df = pd.merge(df_src, df_coordinates, on=["NAME"])


print(df['REGION'].value_counts())
print(df['DIST'].value_counts())

# plot histogram
# there are many more records with the result 2-4 dist score
# there are many more records from the region called 'high' than other two
# dataset is imbalanced
fig, ax = plt.subplots(1, 2)
ax[0].hist(df['DIST'])
ax[1].hist(df['REGION'])
plt.show()

sns.scatterplot('latitude', 'longitude', data=df, hue='REGION')
plt.show()


