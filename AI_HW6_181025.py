import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR,SVC
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

forestfires = pd.read_csv("C:/Users/tzuling/Desktop/forestfires.csv")
df = pd.DataFrame(forestfires)
forestfires.columns = ['X', 'Y', 'month', 'day', 'FFMC',
                       'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    random_state=0)

# Standardizing the data.
sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.transform(x_test)

# use sklearn
def index_value(e,d):
    clf = SVR(C=e, gamma='auto', kernel='rbf')

    clf.fit(x_train, y_train)
    test_times = 3
    x_pred = clf.predict(x_test)
    x_pred_reshape = x_pred.reshape(-1, 1)
    clf_test_accuracy = cross_val_score(
        clf, x_pred_reshape, y_test, cv=test_times)
    clf_test_accuracy2 = clf.score(x_test, y_test)

    print('懲罰參數 =  %s' % e)
    print('kernel =  %s' % d)
    print('交叉驗證正確率\n%s' % clf_test_accuracy.max())
    print('一般驗證正確率\n%s' % clf_test_accuracy2)
    print("-------------------------------------------------------------------------------------")

[index_value(i, j) for i in [0.2, 1.0, 5.0]for j in ['rbf', 'sigmoid']]
