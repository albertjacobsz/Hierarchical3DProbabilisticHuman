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
    y = df["Average Uncertainty"]
    X = df[[ 'Camera Scale', 'Camera X translation', 'Camera Y translation', 'Global X rotation', 'Global Y rotation', 'Global Z rotation', 'R0X', 'R0Y', 'R0Z', 'R1X', 'R1Y', 'R1Z', 'R2X', 'R2Y', 'R2Z', 'R3X', 'R3Y', 'R3Z', 'R4X', 'R4Y', 'R4Z', 'R5X', 'R5Y', 'R5Z', 'R6X', 'R6Y', 'R6Z', 'R7X', 'R7Y', 'R7Z', 'R8X', 'R8Y', 'R8Z', 'R9X', 'R9Y', 'R9Z', 'R10X', 'R10Y', 'R10Z', 'R11X', 'R11Y', 'R11Z', 'R12X', 'R12Y', 'R12Z',
                            'R13X', 'R13Y', 'R13Z', 'R14X', 'R14Y', 'R14Z', 'R15X', 'R15Y', 'R15Z', 'R16X', 'R16Y', 'R16Z', 'R17X', 'R17Y', 'R17Z', 'R18X', 'R18Y', 'R18Z', 'R19X', 'R19Y', 'R19Z', 'R20X', 'R20Y', 'R20Z', 'R21X', 'R21Y', 'R21Z', 'R22X', 'R22Y', 'R22Z', 'J0', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16' ]]
    print(X)
    pca = PCA()
    X_reduced = pca.fit_transform(scale(X))

#define cross validation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    regr = LinearRegression()
    mse = []
    # Calculate MSE with only the intercept
    score = -1*model_selection.cross_val_score(regr,
           np.ones((len(X_reduced),1)), y, cv=cv,
           scoring='neg_mean_squared_error').mean()    
    mse.append(score)
    for i in np.arange(1, 92):
        score = -1*model_selection.cross_val_score(regr,
               X_reduced[:,:i], y, cv=cv, scoring='neg_mean_squared_error').mean()
        mse.append(score)
    
# Plot cross-validation results    
    plt.plot(mse)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('MSE')
    plt.title('Average Uncertainty')
    plt.savefig("mse.png")
    pd.DataFrame(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)).to_csv("percentage_variance.csv")
    

#split the dataset into training (70%) and testing (30%) sets
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0) 

#scale the training and testing data
    X_reduced_train = pca.fit_transform(scale(X_train))
    X_reduced_test = pca.transform(scale(X_test))[:,:1]

#train PCR model on training data 
    regr = LinearRegression()
    regr.fit(X_reduced_train[:,:1], y_train)

#calculate RMSE
    pred = regr.predict(X_reduced_test)
    print(np.sqrt(mean_squared_error(y_test, pred)))


def main():
    pcr()
    correlation_using_pearson()


main()
