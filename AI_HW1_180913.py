import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#100個點的sin函數
x1=np.linspace(0,2*np.pi,100)
y1=np.sin(x1)+np.random.randn(len(x1))/5.0
x1 = x1.reshape(-1, 1)

#畫出M=1迴歸線
poly_features_1 = PolynomialFeatures(degree=1, include_bias=False)
x_poly_1 = poly_features_1.fit_transform(x1)
lin_reg_1 = LinearRegression()
lin_reg_1.fit(x_poly_1, y1)
print("M=1迴歸係數:", lin_reg_1.coef_)
print("M=1截距:", lin_reg_1.intercept_)

x_plot = np.linspace(0, 6, 1000).reshape(-1, 1)
x_plot_poly = poly_features_1.fit_transform(x_plot)
y_plot = np.dot(x_plot_poly, lin_reg_1.coef_.T)+lin_reg_1.intercept_
plt.subplot(4, 1, 1)
plt.title("M = 1")
plt.subplots_adjust(bottom=0.1, top=0.9)
plt.plot(x_plot, y_plot, 'r-')
plt.plot(x1, y1, 'b.')

#畫出M=3迴歸線
poly_features_3 = PolynomialFeatures(degree=3, include_bias=False)
x_poly_3 = poly_features_3.fit_transform(x1)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(x_poly_3, y1)
print("M=3迴歸係數:", lin_reg_3.coef_)
print("M=3截距:", lin_reg_3.intercept_)

x_plot=np.linspace(0,6,1000).reshape(-1,1)
x_plot_poly=poly_features_3.fit_transform(x_plot)
y_plot=np.dot(x_plot_poly,lin_reg_3.coef_.T)+lin_reg_3.intercept_
plt.subplot(4,1,2)
plt.title("M = 3")
plt.subplots_adjust(bottom=0.1, top=0.9)
plt.plot(x_plot,y_plot,'r-')
plt.plot(x1,y1,'b.')

#畫出M=6迴歸線
poly_features_6 = PolynomialFeatures(degree=6, include_bias=False)
x_poly_6 = poly_features_6.fit_transform(x1)
lin_reg_6 = LinearRegression()
lin_reg_6.fit(x_poly_6, y1)
print("M=6迴歸係數:", lin_reg_6.coef_)
print("M=6截距:", lin_reg_6.intercept_)

x_plot = np.linspace(0, 6, 1000).reshape(-1, 1)
x_plot_poly = poly_features_6.fit_transform(x_plot)
y_plot = np.dot(x_plot_poly, lin_reg_6.coef_.T)+lin_reg_6.intercept_
plt.subplot(4, 1, 3)
plt.subplots_adjust(bottom=0.1, top=0.9)
plt.title("M = 6")
plt.plot(x_plot, y_plot, 'r-')
plt.plot(x1, y1, 'b.')

#畫出M=9迴歸線
poly_features_9 = PolynomialFeatures(degree=9, include_bias=False)
x_poly_9 = poly_features_9.fit_transform(x1)
lin_reg_9 = LinearRegression()
lin_reg_9.fit(x_poly_9, y1)
print("M=9迴歸係數:", lin_reg_9.coef_)
print("M=9截距:", lin_reg_9.intercept_)
x_plot = np.linspace(0, 6, 1000).reshape(-1, 1)
x_plot_poly = poly_features_9.fit_transform(x_plot)
y_plot = np.dot(x_plot_poly, lin_reg_9.coef_.T)+lin_reg_9.intercept_
plt.subplot(4, 1, 4)
plt.subplots_adjust(bottom=0.1, top=0.9)
plt.title("M = 9")
plt.plot(x_plot, y_plot, 'r-')
plt.plot(x1, y1, 'b.')

plt.subplots_adjust(hspace=1)
plt.show()
