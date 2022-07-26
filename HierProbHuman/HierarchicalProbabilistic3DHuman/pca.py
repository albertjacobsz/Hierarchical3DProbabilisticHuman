import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
df = pd.read_csv('./csv_files/all_data.csv')

corr = df.corr()
corr_mat = df.corr(method='pearson')

# Retain upper triangular values of correlation matrix and
# make Lower triangular values Null
upper_corr_mat = corr_mat.where(
    np.triu(np.ones(corr_mat.shape), k=1).astype(bool))

# Convert to 1-D series and drop Null values
unique_corr_pairs = upper_corr_mat.unstack().dropna()

# Sort correlation pairs
sorted_mat = unique_corr_pairs.sort_values()
sorted_mat = sorted_mat[sorted_mat.iloc[:, 2] > 0.5]
print(sorted_mat.head())


# sort_T.to_csv('correlations.csv')
