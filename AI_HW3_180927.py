from sklearn import datasets
from sklearn import tree, metrics
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz

forestfires = pd.read_csv("C:/Users/tzuling/Desktop/forestfires.csv")

#類別值轉換
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

x = pd.DataFrame(forestfires, columns=[
                 'day','FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area'])
y = pd.DataFrame(forestfires, columns=['month'])

"""
# 公頃->公畝
y = y * 100 
y = y.astype(int)
"""

cart = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
cart.fit(x, y)
dot_data = tree.export_graphviz(
    cart, out_file=None, 
    class_names=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], 
    filled=True, rounded=True,special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("forestfires")

#使用隨機森林演算法計算重要性
rf = RandomForestClassifier()
rf.fit(x, y)
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index=x.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)    
print(feature_importances)
