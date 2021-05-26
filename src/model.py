import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import os
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans

# read excel file
df = pd.read_excel('../data/scotch.xlsx', header=[0, 1], engine='openpyxl')
print(df.describe())

# remove last 2 rows
df.drop(df.tail(2).index, inplace=True)
# check if any null value
print(df.isnull().values.any())

df_coordinates = pd.read_csv('../data/DISTCOOR.txt', sep="\t", header = None,  skiprows = 6, error_bad_lines=False)

#df_coordinates.columns = ['NAME','longitude','latitude']
print(df_coordinates)
