import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
#实现n次多项式拟合
def fit_func(w,x):
    '根据公式，定义次多项式'
    f=np.poly1d(w) #np.ploy1d()用来构造多项式，默认 ax3+bx2+cx+d
    return f(x)
def err_func(w,x,y):
    '残差函数'
    ret=fit_func(w,x)-y
    return ret
def n_poly(n,x,y):
    'n次多项式拟合'
    w_init=np.random.randn(n) #生成n个随机数最为参数初值
    #调用最小二乘法,x,y为列表型变量
    parameters=leastsq(err_func,w_init,args=(np.array(x),np.array(y)))
    return parameters[0]
#读入文本文件数据
def load_data(filename):
    xys=[]
    with open(filename,'r') as f:
        for line in f:
            xx=line.strip().split()
            xys.append(xx)

    xs=[]
    ys=[]
    size=len(xys)
    for i in range(size):
        xs.append(xys[i][0])
        ys.append(xys[i][1])
    xs=np.array(xs).astype(float)
    ys=np.array(ys).astype(float)
    return xs,ys

if __name__=='__main__':
    #读入数据
    filename_train='train.txt'
    filename_test='test.txt'
    x_train,y_train=load_data(filename_train)
    #print(x_train)
    #x_train=x_train.reshape(-1,1)
    #y_train=y_train.reshape(-1,1)
    k=n_poly(8,x_train,y_train)
    print("k=",k)
    #'''绘制出3,4,5,6,7,8次多项式的拟合回归'''
    #绘制拟合回归时需要的监测点
    x_temp=np.linspace(0,25,10000)
    #绘制子图
    fig,axes=plt.subplots(2,3,figsize=(15,10))
    axes[0,0].plot(x_temp,fit_func(n_poly(4,x_train,y_train),x_temp),'r')
    axes[0,0].scatter(x_train,y_train)
    axes[0,0].set_title('m=3')

    axes[0, 1].plot(x_temp, fit_func(n_poly(5, x_train, y_train), x_temp), 'r')
    axes[0, 1].scatter(x_train, y_train)
    axes[0, 1].set_title('m=4')

    axes[0, 2].plot(x_temp, fit_func(n_poly(6, x_train, y_train), x_temp), 'r')
    axes[0, 2].scatter(x_train, y_train)
    axes[0, 2].set_title('m=5')

    axes[1, 0].plot(x_temp, fit_func(n_poly(7, x_train, y_train), x_temp), 'r')
    axes[1, 0].scatter(x_train, y_train)
    axes[1, 0].set_title('m=6')

    axes[1, 1].plot(x_temp, fit_func(n_poly(8, x_train, y_train), x_temp), 'r')
    axes[1, 1].scatter(x_train, y_train)
    axes[1, 1].set_title('m=7')

    axes[1, 2].plot(x_temp, fit_func(n_poly(14, x_train, y_train), x_temp), 'r')
    axes[1, 2].scatter(x_train, y_train)
    axes[1, 2].set_title('m=13')
    plt.show()
