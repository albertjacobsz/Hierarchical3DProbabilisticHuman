import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error

df = pd.read_csv('./csv_files/all_data.csv')

corr = df.corr()
cor_matrix = df.corr().abs()
print(cor_matrix)
upper_tri = cor_matrix.where(
    np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
print(upper_tri)
to_drop = [column for column in upper_tri.columns if any(
    upper_tri[column] < 0.2 and upper_tri[column] > -0.2)]
print()
print(to_drop)
df1 = df.drop(df.columns[to_drop], axis=1)
print()
print(df1.head())
fig = plt.figure(figsize=(100, 100))
ax = fig.add_subplot(111)

cax = ax.matshow(df1, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, len(df.columns), 1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df.columns, fontsize=78)
ax.set_yticklabels(df.columns, fontsize=78)
plt.savefig("representation.png")
