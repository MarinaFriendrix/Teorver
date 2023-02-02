import math
def combinations (n,k):
    comb = math.factorial(n) / (math.factorial(k) * math.factorial(n-k))
    return (comb)

def bernuli (n,p,k):
    bern = combinations(n,k)*pow(p,k)*pow((1-p),(n-k))
    return (bern)


# Задача 1. Вероятность того, что стрелок попадет в мишень, выстрелив один раз, равна 0.8. 
# Стрелок выстрелил 100 раз. Найдите вероятность того, что стрелок попадет в цель ровно 85 раз.

# p=bernuli(100,0.8,85 )
# print ("{:.2%}".format(p))

# Задача.2 Вероятность того, что лампочка перегорит в течение первого дня эксплуатации, равна 0.0004. 
# В жилом комплексе после ремонта в один день включили 5000 новых лампочек. 
# Какова вероятность, что ни одна из них не перегорит в первый день?

# n=5000
# m1=0
# m2=2
# p=0.004

# la=p*n
# print (la)

# p1= pow(la,m1)/math.factorial(m1)*math.exp(-la)

# print (p1)
 
# # Какова вероятность, что перегорят ровно две?

# p2= pow(la,m2)/math.factorial(m2)*math.exp(-la)
# print (p2)

# Задача 3. Монету подбросили 144 раза. 
# Какова вероятность, что орел выпадет ровно 70 раз?

# p=bernuli(100, 0.5, 70)
# print ("{:%}".format(p))


# задача 4. В первом ящике находится 10 мячей, из которых 7 - белые. Во втором ящике - 11 мячей, из которых 9 белых. 
# Из каждого ящика вытаскивают случайным образом по два мяча. 
# Какова вероятность того, что все мячи белые? 
# p1 - 2 белых из первого ящика
# p2 - 2 белых из 2 ящика
# p = p1*p2 -  из 1 и из 2 ящика по 2 белых мяча.

# общее число исходов 1 ящик:
c1 = combinations (10, 2)
# общее число исходов 2 ящик:
c2 = combinations (11,2)

# благоприятные исходы 1 ящик:
c11 = combinations (7, 2)
# благоприятные исходы 2 ящик:
c22 = combinations (9, 2)

p1 = c11/c1
p2 = c22/c2

# p = p1*p2
# print ("{:%}".format(p))

# Какова вероятность того, что ровно два мяча белые?
# вероятность выпадения 1 белого мяча из 1 ящ и из 2 ящ.

# вероятность вытщить белый шар из 1 ящика
p11 = 7/10

# вероятность вытащить после белого черный шар из 1 ящика
p12 = 3/9

# вероятность вытщить белый шар из 2 ящика
p21 = 9/11

# вероятность вытащить после белого черный шар из 2 ящика
p22 = 2/10

# вероятность вытащить белый потом черный из 1 ящика
p111 = p11*p12

# вероятность вытащить белый потом черный из 2 ящика
p222 = p21*p22

# общая веротяность по 1 белому мячу из каждого ящика
p3 = p111*p222

# общая вероятность вытащить 2 белых мяча
# p = p1*(1-p2) + p2*(1-p1) + p3
# print ("{:%}".format(p))

# Какова вероятность того, что хотя бы один мяч белый?

# вероятность вытащить 2 черных из 1 ящ.
c3 = combinations(3,2)
q1 = c3/c1

# вероятность вытащить 2 черных из 2 ящ.
c4 = combinations(2,2)
q2 = c4/c2

# вероятность вытащить все черные мячи
q = q1*q2
print("{:%}".format(q))

p = 1-q
print ("{:%}".format(p))
