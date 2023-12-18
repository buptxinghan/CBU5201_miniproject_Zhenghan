# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

# 读取图片标签文件
labels = pd.read_csv('D:/Documents/Junior1/Machine Learning/LAB/Mini Project/genki4k/labels.txt', sep=' ', header=None, names=['filename', 'smile', 'yaw', 'pitch', 'roll'])

# 定义一个函数，用来读取和预处理图片
def preprocess_image(filename):
  # 读取图片文件
  image = cv2.imread('D:/Documents/Junior1/Machine Learning/LAB/Mini Project/genki4k/files/' + str(filename))
  # 转换为灰度图
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # 使用OpenCV的人脸检测器，找到人脸的位置
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  faces = face_cascade.detectMultiScale(image, 1.3, 5)
  # 如果没有检测到人脸，返回None
  if len(faces) == 0:
    return None
  # 如果检测到多个人脸，只取第一个人脸
  x, y, w, h = faces[0]
  # 裁剪人脸区域
  image = image[y:y+h, x:x+w]
  # 调整图片大小为64x64
  image = cv2.resize(image, (64, 64))
  # 归一化图片像素值到[0, 1]区间
  image = image / 255.0
  # 返回图片数组
  return image

# 定义一个空的列表，用来存储图片数组
images = []
# 遍历图片标签数据框的每一行
for index, row in labels.iterrows():
  # 调用预处理函数，得到图片数组
  image = preprocess_image(row['filename'])
  # 如果图片数组不是None，把它添加到列表中
  if image is not None:
    images.append(image)
  # 打印进度
  print(f'Processed {index + 1} / {len(labels)} images')

# 把图片列表转换成numpy数组
images = np.array(images)
# 为图片数组增加一个维度，表示通道数为1
images = np.expand_dims(images, axis=-1)

# 把标签数据框转换成numpy数组
labels = labels.to_numpy()

# 使用sklearn的train_test_split函数，划分数据集
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# 定义模型的输入层，大小为64x64x1
inputs = Input(shape=(64, 64, 1))
# 定义卷积层和池化层，使用ReLU激活函数
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
# 定义全连接层，使用ReLU激活函数
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
# 定义两个输出层，一个用来做二分类，一个用来做回归
outputs1 = Dense(1, activation='sigmoid', name='smile')(x) # 二分类输出层，使用sigmoid激活函数，输出笑脸的概率
outputs2 = Dense(3, name='head_pose')(x) # 回归输出层，输出头部姿态的三个参数
# 定义模型，指定输入和输出
model = Model(inputs=inputs, outputs=[outputs1, outputs2])
# 打印模型的结构
model.summary()

# 定义优化器，使用Adam
optimizer = Adam(lr=0.001)
# 编译模型，指定损失函数，优化器，和评估指标
model.compile(loss={'smile': 'binary_crossentropy', 'head_pose': 'mse'}, # 使用交叉熵损失函数和均方误差损失函数
              optimizer=optimizer,
              metrics={'smile': 'accuracy', 'head_pose': 'mae'}, # 使用准确率和平均绝对误差作为评估指标
              loss_weights={'smile': 1, 'head_pose': 1}) # 给两个输出的损失赋予相同的权重

# 定义一个回调函数，用来在训练过程中减少学习率
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)

# 定义一个数据生成器，用来进行数据增强
datagen = ImageDataGenerator(rotation_range=10, # 随机旋转角度
                             width_shift_range=0.1, # 随机水平平移
                             height_shift_range=0.1, # 随机竖直平移
                             shear_range=0.1, # 随机错切变换
                             zoom_range=0.1, # 随机缩放
                             horizontal_flip=True, # 随机水平翻转
                             fill_mode='nearest') # 填充方式

# 训练模型，使用数据生成器，指定批量大小，迭代次数，验证数据，和回调函数
history = model.fit(datagen.flow(X_train, {'smile': y_train[:, 1], 'head_pose': y_train[:, 2:]}, batch_size=32),
                    steps_per_epoch=len(X_train) // 32,
                    epochs=100,
                    validation_data=(X_val, {'smile': y_val[:, 1], 'head_pose': y_val[:, 2:]}),
                    callbacks=[reduce_lr])
# 绘制训练和验证的损失和准确率曲线
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(history.history['smile_loss'], label='train')
plt.plot(history.history['val_smile_loss'], label='val')
plt.title('Smile Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(2, 2, 2)
plt.plot(history.history['smile_accuracy'], label='train')
plt.plot(history.history['val_smile_accuracy'], label='val')
plt.title('Smile Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(2, 2, 3)
plt.plot(history.history['head_pose_loss'], label='train')
plt.plot(history.history['val_head_pose_loss'], label='val')
plt.title('Head Pose Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(2, 2, 4)
plt.plot(history.history['head_pose_mae'], label='train')
plt.plot(history.history['val_head_pose_mae'], label='val')
plt.title('Head Pose MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.show()
# 在测试集上评估模型的性能
model.evaluate(X_test, {'smile': y_test[:, 1], 'head_pose': y_test[:, 2:]})
