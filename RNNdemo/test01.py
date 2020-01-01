import numpy as np


def one_hot(x):
    # 每个验证码有4个文字，每个文字由62个可能，因此标签的形状为（4,62）
    z = np.zeros(shape=[4, 62])
    li1 = [chr(i) for i in range(48, 58)]
    li2 = [chr(i) for i in range(65, 91)]
    li3 = [chr(i) for i in range(97, 123)]
    li = li1 + li2 + li3
    # print(li.index('z'))
    for i in range(4):
        index = int(li.index(x[i]))
        z[i][index] += 1
    return z

a = '1Dz0'
print(one_hot(a))


def one_hot(self, x):
    # 每个验证码有4个文字，每个文字由62个可能，因此标签的形状为（4,62）
    z = np.zeros(shape=[4, 62])
    for i in range(4):
        index = int(x[i])
        z[i][index] += 1
    return z