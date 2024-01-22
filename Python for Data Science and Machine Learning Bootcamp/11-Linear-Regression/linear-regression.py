import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\Asus\\github\\testspace\\Learning\\Python for Data Science and Machine Learning Bootcamp\\11-Linear-Regression\\USA_Housing.csv")
df
df.info()
df.describe()
df.columns

sns.pairplot(df)
sns.distplot(df['Price'])
sns.heatmap(df.corr(), annot=True)


from sklearn.model_selection import train_test_split

X = df[['Avg. Area Income',
        'Avg. Area House Age',
        'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms',
        'Area Population']]
        # left out price because that's what we're trying to predict
        # left out Adress because SciKit doesn't process text

y = df['Price'] # what we want to predict

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()  # instantiate a linear regression model object
lm.fit(X_train, y_train)  # train the model

lm.intercept_  # evaluate the model: where is the y-intercept?
lm.coef_       # evaluate the model: coefficients of each column of X
lm.score(X_train, y_train)

cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])
cdf  # Explanation of coefficients: "if you hold all other parameters
     # fixed, a $1 increase in Avg. Area Income corresponds to 21.52$
     # increase in Price"


predictions = lm.predict(X_test)
predictions  # the predicted prices for each house
y_test       # the actual prices for each house
sns.scatterplot(y_test, predictions)  # looks like good prediction

sns.distplot((y_test-predictions))
    # distribution of residuals (diff between actual vs predicted)
    # normally distributed residuals = correct model for data


# Loss functions as evaluation metrics
from sklearn import metrics

metrics.mean_absolute_error(y_test, predictions)            # MAE
metrics.mean_squared_error(y_test, predictions)             # MSE
np.sqrt(metrics.mean_squared_error(y_test, predictions))    # RMSE
metrics.explained_variance_score(y_test, predictions)       # R-squared
