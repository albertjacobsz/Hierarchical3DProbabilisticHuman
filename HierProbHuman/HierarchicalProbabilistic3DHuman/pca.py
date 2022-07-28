import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def correlation_using_pearson():
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
    sorted_mat.to_csv('correlations.csv')


def pcr():
    df = pd.read_csv('./csv_files/all_data.csv')
    df = df[:, 0:]
    print(df)


def main():
    pcr()
