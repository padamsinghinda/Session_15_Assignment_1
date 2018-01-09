# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 10:47:20 2018

@author: Padam Singh

"""

def Linear_regression_model() :
    """Problem : Build a linear regression model to predict housing price using the boston housing dataset. """
    #Import third party libraries
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    
    #import sklearn modules used for linear regresion
    from sklearn.datasets import load_boston
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    
    #Load boston housing pricing dataset
    boston = load_boston()
    
    #View boston dataset features
    print(f"boston dataset feature_names are :\n{boston.feature_names}")
    
    #Create DataFrame df_X, df_Y using boston dataset which contains features and target values
    df_x = pd.DataFrame(boston.data, columns=boston.feature_names)
    df_y = boston.target
    
    #Apply sklearn linear regression model
    lr = LinearRegression()
    
    #Split boston dataset into trainning and test dataset
    x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=5)
    
    #Fit linear regression model on training dataset
    lr.fit(x_train,y_train)
    
    #Perform prediction on test dataset
    pred = lr.predict(x_test)
    
    #Plot a figure for linear regression model
    print("\n\nPlot between Actual and Predicted price :")
    print("-----------------------------------------")
    plt.xlabel("Actual Price ($1000) ")
    plt.ylabel("Predicted Price ($1000) ")
    plt.title("Figure : Actual vs Predicted Price")
    plt.scatter(y_test, pred, color='black')
    plt.show()
    
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, pred))
    
def main():
    Linear_regression_model()
    
if __name__ == '__main__':
    main()
