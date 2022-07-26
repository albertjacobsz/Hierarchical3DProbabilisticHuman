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
    np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool))

# Convert to 1-D series and drop Null values
unique_corr_pairs = upper_corr_mat.unstack().dropna()

# Sort correlation pairs
sorted_mat = unique_corr_pairs.sort_values()
header = ['x', 'y', 'z']
print(sorted_mat)
with open('correlations.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)
    for x in sorted_mat:
        writer.writerow(x)
