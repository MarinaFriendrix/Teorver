# Задача 1 Даны значения величины заработной платы заемщиков банка (zp) и значения их
# поведенческого кредитного скоринга (ks): zp = [35, 45, 190, 200, 40, 70, 54, 150, 120, 110],
# ks = [401, 574, 874, 919, 459, 739, 653, 902, 746, 832]. Используя математические
# операции, посчитать коэффициенты линейной регрессии, приняв за X заработную плату
# (то есть, zp - признак), а за y - значения скорингового балла (то есть, ks - целевая
# переменная). Произвести расчет как с использованием intercept, так и без.

import math
import scipy.stats as stats
import numpy as np
from sklearn.linear_model import LinearRegression

# y = b0 + b1*x
zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])
n = 10

# b1 = ((n*np.sum (zp*ks)) - (np.sum(zp)*np.sum(ks)))/((n* np.sum(zp**2)) - np.sum (zp)**2)
# print (b1)

# b0 = np.mean(ks) - (b1*np.mean(zp))
# print (b0)

# y_pred = b0 + (b1*zp)

# print (y_pred)

model = LinearRegression()
s = zp.reshape(-1,1)
# print (s)

# regres = model.fit(s,ks)
# print (regres.intercept_)

# print (regres.coef_)

# y_pred = model.predict(s)
# print (y_pred)

# (y_1  y_2  y_3  )= (1 x_1  1 x_2  1 x_3  )  (β_0  β_1  )

# x = zp.reshape((10,1))
# # print (x)
# y = ks.reshape((10,1))
# # print(y)

# X = np.hstack([np.ones((10,1)),x])
# # print (X)

# B = np.dot(np.linalg.inv(np.dot(X.T,X)), X.T @ y)
# print (B)

# Задача 2 Посчитать коэффициент линейной регрессии при заработной плате (zp), используя
# градиентный спуск (без intercept)

def mse_(B1, y = ks, x= zp, n= 10):
    return np.sum(((B1*x) - y)**2)/n

B1 = 0.1
alfa = 0.000001

for i in range (10):
    B1 = B1 - alfa*(2/n)*np.sum(((B1*zp) - ks)*zp)
    regres = model.fit(s,ks)

    print ('B1 = {}'.format(B1)+' '+'intercept = {}'.format (regres.intercept_)) 
    

# Задача 3 (Дополнительно) Произвести вычисления как в пункте 2, но с вычислением intercept. Учесть, что
# изменение коэффициентов должно производиться
# на каждом шаге одновременно (то есть изменение одного коэффициента не должно
# влиять на изменение другого во время одной итерации).


