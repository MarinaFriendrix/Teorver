import math
import scipy.stats as stats
import numpy as np

# Задача 1 Даны значения величины заработной платы заемщиков банка (zp) и значения их
# поведенческого кредитного скоринга (ks):
# zp = [35, 45, 190, 200, 40, 70, 54, 150, 120, 110],
# ks = [401, 574, 874, 919, 459, 739, 653, 902, 746, 832].
# Найдите ковариацию этих двух величин с помощью элементарных действий, а затем с
# помощью функции cov из numpy
# Полученные значения должны быть равны.
# Найдите коэффициент корреляции Пирсона с помощью ковариации и
# среднеквадратичных отклонений двух признаков,
# а затем с использованием функций из библиотек numpy и pandas.

# 〖cov〗_xy=M(XY)-M (X)*M(Y)

# r=〖cov〗_xy/(σ_x*σ_y )
# zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
# ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])

# cov = np.mean(zp*ks)- (np.mean(zp)*np.mean(ks))
# print (cov)

# cov_1 = np.cov(zp, ks)
# print (cov_1)

# std_zp = np.std(zp, ddof=0)
# std_ks = np.std(ks, ddof =0)

# r = cov/(std_zp*std_ks)
# print (r)

# r_1 = np.corrcoef(zp, ks)
# print (r_1)


# Задача 2 Измерены значения IQ выборки студентов,
# обучающихся в местных технических вузах:
# 131, 125, 115, 122, 131, 115, 107, 99, 125, 111.
# Известно, что в генеральной совокупности IQ распределен нормально.
# Найдите доверительный интервал для математического ожидания с надежностью 0.95.

# a = np.array([131, 125, 115, 122, 131, 115, 107, 99, 125, 111])
# n= len(a)
# alfa = 0.05
# t = stats.t.ppf((1-alfa/2), (n-1))
# print(t)

# x_mean = np.mean (a)
# D = np.var(a)
# print (x_mean, D)

# t1 = x_mean + t*math.sqrt((D/n))
# t2 = x_mean - t*math.sqrt((D/n))

# print (t1, t2)
# # доверительный интервал (125.26; 110.94)


# Задача 3 Известно, что рост футболистов в сборной распределен нормально
# с дисперсией генеральной совокупности, равной 25 кв.см. Объем выборки равен 27,
# среднее выборочное составляет 174.2. Найдите доверительный интервал для
# математического
# ожидания с надежностью 0.95.

D = 25
n= 27
M = 174.2
alfa = 0.05
std = 5

z = stats.norm.ppf((1-(alfa/2)))
z1 = M + (z*(std/math.sqrt(n)))
z2 = M - (z*(std/math.sqrt(n)))

print (z1, z2)
# доверительный интервал (176.09; 172.31)


