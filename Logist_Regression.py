import numpy as np
import pandas as pd
import time as time
import random as random

def loaddata (filename):
    # 读取文件
    data = pd.read_csv(filename, header=None)
    # 在对DataFrame类型的数据进行处理时，需要将其转换成array类型
    data = data.values
    # 切分x，y
    x_label = data[:, 1:]
    y_label = data[:, 0]
    # 二分类
    y_label[y_label > 0] = 1
    return x_label, y_label

def sigmoid(x):
    # 设定hx的最小值，防止出现很接近0的数字，导致log无限大，loss为nan
    minhx = np.exp(-3)
    hx = 1 / (1 + np.exp(-1 * x))
    hx[hx < minhx] = minhx
    return hx

def logitRegression(x_train, y_train, epochs):
    # 随机生成参数w=758*1，列向量； x_train=n*758 hx=n*1
    w = np.mat([random.uniform(0, 1)for _ in range(len(x_train[0]))]).reshape(-1, 1)
    # 样本转化为矩阵
    x_train = np.mat(x_train)
    y_train = np.mat(y_train)
    # 设定学习效率
    learning_rate = 0.001
    for i in range(epochs):
        print(f'in {i} epoch')
        hx = sigmoid(x_train * w)
        w -= learning_rate * x_train.T * (hx-y_train.T)
    return w

def predict (x):
    hx = sigmoid(x)
    if hx >= 0.5:
        return 1
    else:
        return 0

def test (x_test, y_test, w):
    acc = 0
    for i in range(len(y_test)):
        x = np.mat(x_test[i])
        # x=1*758; w=758*1
        hx = predict(x * w)
        if hx == y_test[i]:
            acc += 1
    print('accuracy =', acc/len(y_test))

if __name__=='__main__':
    start = time.time()
    x_train, y_train = loaddata('E:/OneDrive/Software/python_code/Logist_Regression/Mnist/mnist_train.csv')
    x_test, y_test = loaddata('E:/OneDrive/Software/python_code/Logist_Regression/Mnist/mnist_test.csv')
    w = logitRegression(x_train, y_train, 10)
    test(x_test, y_test, w)
    end = time.time()
    print('runtime =', end - start)





