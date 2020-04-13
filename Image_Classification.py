import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# 数据加载
fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(x_valid.shape, y_valid.shape)

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 数据归一化 x=（x-e）/std 标准正态分布
# 提高训练速度从而提高准确率
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
# fit可以提取期望和方差，训练集 测试集 验证集的期望方差应一致
x_train_scalar = scalar.fit_transform(
    x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_test_scalar = scalar.transform(
    x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_valid_scalar = scalar.transform(
    x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
print(np.max(x_train), np.min(x_train))
print(np.max(x_train_scalar), np.min(x_train_scalar))


# 打印图片
def show_single_image(img_arr):
    plt.imshow(img_arr, cmap='binary')
    plt.show()


show_single_image(x_train[0])


def show_imgs(n_rows, n_cols, x_data, y_data, class_names):
    assert len(x_data) == len(y_data)
    assert n_cols*n_rows < len(x_data)
    plt.figure(figsize=(n_cols * 1.4, n_rows * 1.6))
    for row in range(n_rows):
        for clo in range(n_cols):
            index = row * n_cols + clo
            plt.subplot(n_rows, n_cols, index+1)
            plt.imshow(x_data[index], cmap='binary', interpolation='nearest')
            plt.axis('off')
            a = int(y_data[index])
            plt.title(class_names[a])
    plt.show()


show_imgs(3, 5, x_train, y_train, class_names)

# 模型构建
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))
model.add(tf.keras.layers.Dense(300, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# metrics是列表，要加[]
# y是index，不是向量，所以loss要前缀sparse
# model.Sequential 可以直接用列表定义多层
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train_scalar, y_train, epochs=10, validation_data=(x_valid_scalar, y_valid))
model.evaluate(x_test_scalar, y_test)


def plot_learing_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


plot_learing_curves(history)

model.evaluate(x_test_scalar, y_test)