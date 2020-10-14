import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#學習資料載入

training = np.random.randint(1, 100, size=(20, 2))

df = pd.DataFrame(training)
print(df)
train_x = training[:,0]
train_y = training[:,1]

# plt.subplot(3, 1, 1)
plt.title("random data")
# plt.subplots_adjust(bottom=0.1, top=0.9)
plt.scatter(train_x,train_y)
plt.show()

#參數初始化
theta0 = np.random.rand()
theta1 = np.random.rand()
#線性回歸函數後的預測函數
def f(x):
    return theta0 + theta1 * x

#cost function
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)

#學習率
ETA = 1e-3

#誤差的差分
diff = 1

#更新回數
count = 0

#標準化
mu = train_x.mean()
sigma = train_x.std()

def standardize(x):
    return (x-mu) / sigma

train_z = standardize(train_x)
# plt.subplot(3, 1, 2)
plt.title("standardize")
# plt.subplots_adjust(bottom=0.1, top=0.9)
plt.scatter(train_z,train_y)
plt.show()

#畫出回歸線
slr = LinearRegression()
train_z = train_z.reshape(-1, 1)
slr.fit(train_z, train_y)
predicted_y1 = slr.predict(train_z)
# plt.subplot(3, 1, 3)
plt.title("GD")
# plt.subplots_adjust(bottom=0.1, top=0.9)
plt.scatter(train_z, train_y)
plt.plot(train_z, predicted_y1, color='r')
# plt.subplots_adjust(hspace=1)
plt.show()

#參數調校更新直到誤差差分小於0.01
error = E(train_z,train_y)
while diff > 1e-2  :
    #更新結果暫時儲存
    tmp_theta0 = theta0 - ETA * np.sum((f(train_z)-train_y))
    tmp_theta1 = theta1 - ETA * np.sum((f(train_z)-train_y) * train_z)
    #更新參數
    theta0 = tmp_theta0
    theta1 = tmp_theta1
    #計算與前一項誤差的差分
    current_error = E(train_z,train_y)
    diff = error - current_error
    error = current_error
    #運算過程
    count += 1
    log = '{}次: theta0 = {:.3f}, theta1 = {:.3f}, 差分 = {:.4f}'
    print(log.format(count,theta0,theta1,diff))
