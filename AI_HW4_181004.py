from sklearn import datasets
from sklearn import tree, metrics
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

forestfires = pd.read_csv("C:/Users/tzuling/Desktop/forestfires.csv")

month_mapping = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12
}
day_mapping = {
    "mon": 1,
    "tue": 2,
    "wed": 3,
    "thu": 4,
    "fri": 5,
    "sat": 6,
    "sun": 7
}
forestfires['month'] = forestfires['month'].map(month_mapping)
forestfires['day'] = forestfires['day'].map(day_mapping)

x = forestfires.iloc[:, :-1].values
y = forestfires.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.ensemble import RandomForestRegressor
def accuracy(a, ans):
    regressor = RandomForestRegressor(n_estimators=ans, random_state=0)
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    accuracy = regressor.score(x_train, y_train)
    print(a,accuracy)
    
accuracy("while estimator=20 : ", 20)
accuracy("while estimator=50 : ", 50)
accuracy("while estimator=100 : ", 100)
