import matplotlib.pyplot as plt
import numpy as np

print('胡依')

x = [1, 2]
y = [2, 3]
y2 = [3, 4]

# fig = plt.figure()
# plt.plot(x, y, label='佳佳')
# plt.plot(x, y2, label='胡依')


def f(x, y):
    return x ** 2 + (y - np.cbrt(x ** 2)) ** 2


# 设置x,y的范围  在（-10，10）之间取100个点

x = np.linspace(-10, 10, 100)

y = np.linspace(-10, 10, 100)

# 将xy的值对应起来  类似100*100的二维矩阵 类似网格
x, y = np.meshgrid(x, y)

# z可以理解成是等高线的高度
z = f(x, y)

# 将等高线表示出来
plt.contour(x, y, z,label='胡依')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.legend()
plt.show()
