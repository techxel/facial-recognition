from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import keras
import numpy as np
import os

X = []
y = []

# function to get the images and label data
def trainImages(path, user_id):
    for img_path in os.listdir(path):
        img = plt.imread(img_path)
        # is image ?
        if not hasattr(img, 'shape'):
            print(img_path, 'is not a image file!')
            continue

        X.append(img)
        y.append(user_id)

trainImages('dataset/1', 1)
trainImages('dataset/2', 2)

X = np.array(X)
X = X.reshape(len(y), 64, 64, 1)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = keras.Sequential()
# 第一层卷积，卷积的数量为128，卷积的高和宽是3x3，激活函数使用relu
model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu', input_shape=(64, 64, 1)))
# 第二层卷积
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
#把多维数组压缩成一维，里面的操作可以简单理解为reshape，方便后面Dense使用
model.add(keras.layers.Flatten())
#对应cnn的全链接层，可以简单理解为把上面的小图汇集起来，进行分类
model.add(keras.layers.Dense(40, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(X_test, y_test)
print("n\Test accuracy: ", test_acc)

# model.save("face.h5")

# model = keras.models.load_model("face.h5")

y_predicts = model.predict(X_test)

for i in range(len(y_test)):
    plt.grid(False)
    plt.imshow(X_test[i], cmap="gray")
    plt.title('实际结果: ' + str(y_test[i]))
    y_predict = np.argmax(y_predicts[i])
    plt.xlabel('预测结果: ' + str(y_predict))
    plt.show()