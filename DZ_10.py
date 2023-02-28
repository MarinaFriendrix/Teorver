# Провести дисперсионный анализ для определения того, есть ли различия среднего
# роста среди взрослых футболистов, хоккеистов и штангистов.
# Даны значения роста в трех группах случайно выбранных спортсменов:
# Футболисты: 173, 175, 180, 178, 177, 185, 183, 182.
# Хоккеисты: 177, 179, 180, 188, 177, 172, 171, 184, 180.
# Штангисты: 172, 173, 169, 177, 166, 180, 178, 177, 172, 166, 170.
# H0: m1=m2=m3
# H1.1: m1=m2
# H1.2: m1=m3
# H1.3: m2=m3
import numpy as np
import math
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd

alfa = 0.05

y1 = np.array([173, 175, 180, 178, 177, 185, 183, 182])
y2 = np.array([177, 179, 180, 188, 177, 172, 171, 184, 180])
y3 = np.array([172, 173, 169, 177, 166, 180, 178, 177, 172, 166, 170])

k=3
n=len(y1)+ len(y2)+len(y3)
# print(n)

y1_mean = np.mean(y1)
y2_mean = np.mean(y2)
y3_mean = np.mean(y3)

# print (y1_mean, y2_mean, y3_mean)

y_total = np.hstack([y1, y2, y3])
# print (y_total)
y_total_mean = np.mean(y_total)
# print (y_total_mean)

S_o = np.sum((y_total - y_total_mean)**2)
print (S_o)

S_f = (((y1_mean - y_total_mean)**2)*len(y1)) + (((y2_mean - y_total_mean)**2)*len(y2)) + (((y3_mean - y_total_mean)**2)*len(y3))
print (S_f)

S_ost = np.sum((y1 -y1_mean)**2) + np.sum((y2 -y2_mean)**2) + np.sum((y3 -y3_mean)**2)
print (S_ost)

D_f = S_f/(k-1)
D_ost = S_ost/(n-k)
print (D_f, D_ost)

F_n = D_f/D_ost
print (F_n)

F_n1 = stats.f_oneway(y1, y2, y3)
print (F_n1)
# p_v = 0.01048 < 0.05, принимаем H1

