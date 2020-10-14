# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 19:08:59 2018

@author: tzuling
"""

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from mlxtend.evaluate import mcnemar_table
from mlxtend.plotting import checkerboard_plot
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

forestfires = pd.read_csv("C:/Users/tzuling/Desktop/forestfires.csv")
df = pd.DataFrame(forestfires)

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

#forestfires['month'] = forestfires['month'].map(month_mapping)
forestfires['day'] = forestfires['day'].map(day_mapping)

scaler = StandardScaler()
scaler.fit(df.drop('month',axis=1))
scaled_features = scaler.transform(df.drop('month',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
x = df_feat
y = df['month']


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=14)

#===================================CART=======================================
cart = tree.DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=0)
cart.fit(x_train, y_train)
cart_pred = cart.predict(x_test)
accuracy = cart.score(x_train, y_train)
print("CART Accuracy：",accuracy)
print("====================================")

#====================================LR========================================
logistic_regr = LogisticRegression()
logistic_regr.fit(x_train,y_train)
accuracy = logistic_regr.score(x_train, y_train)
print("LR Accuracy：",accuracy)
print("====================================")

#===================================KNN========================================
accuracy_rate = []
for i in range(1,60):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    accuracy_rate.append(np.mean(pred_i == y_test))


#將k=1~60的錯誤率製圖畫出。k=23之後，錯誤率就在5%-6%之間震盪，
plt.figure(figsize=(10,6))
plt.plot(range(1,60),accuracy_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Accuracy Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy Rate')

print("KNN Accuracy：",accuracy_rate.index(max(accuracy_rate))+1)




