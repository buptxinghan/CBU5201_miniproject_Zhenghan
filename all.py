# 预处理：
# 这一步的目的是将图片中的人脸区域裁剪出来，并调整为统一的大小和格式。
# 这样可以减少无关的干扰信息，提高模型的效率和准确性。可以使用OpenCV提供的人脸检测和裁剪的功能
import cv2
import os
import numpy as np

# 读取数据集中的图片和标签文件
images_path = "D:/Documents/Junior1/Machine Learning/LAB/Mini Project/genki4k/files"
labels_path = "D:/Documents/Junior1/Machine Learning/LAB/Mini Project/genki4k/labels.txt"
images = os.listdir(images_path)
labels = np.loadtxt(labels_path)

# 定义输出的图片大小和格式
output_size = (224, 224) # 你可以根据你的模型和硬件条件来调整这个参数
output_format = ".jpg"

# 加载OpenCV的人脸检测器
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 创建一个新的文件夹来存储裁剪后的图片
output_path = "D:\Documents\Junior1\Machine Learning\LAB\Mini Project\genki4k\cropped"
if not os.path.exists(output_path):
    os.mkdir(output_path)

# 遍历数据集中的每一张图片，检测人脸，裁剪，保存
for i, image in enumerate(images):
    # 读取图片
    image_path = os.path.join(images_path, image)
    image = cv2.imread(image_path)
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 检测人脸
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    # 如果没有检测到人脸，跳过这张图片
    if len(faces) == 0:
        continue
    # 如果检测到多个人脸，只取最大的那个
    max_face = max(faces, key=lambda x: x[2] * x[3])
    # 裁剪人脸区域
    x, y, w, h = max_face
    cropped = image[y:y+h, x:x+w]
    # 调整图片大小
    resized = cv2.resize(cropped, output_size)
    # 保存图片，文件名为原始图片的文件名加上标签
    output_name = image[:-4] + "_" + str(int(labels[i])) + output_format
    output_file = os.path.join(output_path, output_name)
    cv2.imwrite(output_file, resized)



# 划分数据集：
# 这一步的目的是将裁剪后的图片分为训练集、验证集和测试集，以便进行模型的训练、评估和测试。
# 可以使用scikit-learn提供的数据集划分的功能
from sklearn.model_selection import train_test_split
import os
import glob

# 读取裁剪后的图片文件夹
cropped_path = "D:\Documents\Junior1\Machine Learning\LAB\Mini Project\genki4k\cropped"
cropped_images = glob.glob(cropped_path + "/*.jpg")

# 提取图片的标签，标签是图片文件名的最后一个字符
cropped_labels = [int(image[-5]) for image in cropped_images]

# 划分数据集，你可以根据你的需要来调整划分的比例
train_images, test_images, train_labels, test_labels = train_test_split(cropped_images, cropped_labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25, random_state=42)

# 打印划分后的数据集的大小
print("Train set size:", len(train_images))
print("Validation set size:", len(val_images))
print("Test set size:", len(test_images))



# 提取特征：
# 这一步的目的是将图片转换为模型可以处理的数值特征。使用深度学习的方法，比如卷积神经网络（CNN），来自动地学习图片的特征。
# 使用一些现成的预训练的模型，比如VGG, ResNet等，来提取图片的特征。
# 使用TensorFlow提供的深度学习的功能
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
import numpy as np

# 定义图片的大小和格式，要和预处理时的一致
input_size = (224, 224)
input_format = ".jpg"

# 加载预训练的VGG16模型，不包括最后的分类层，只用来提取特征
vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(input_size[0], input_size[1], 3))

# 定义一个函数，用来将图片转换为特征向量
def image_to_feature(image_path):
    # 读取图片
    image = image.load_img(image_path, target_size=input_size)
    # 转换为数组
    image = image.img_to_array(image)
    # 扩展一个维度，表示批量大小为1
    image = np.expand_dims(image, axis=0)
    # 预处理图片，使其符合VGG16的输入要求
    image = tf.keras.applications.vgg16.preprocess_input(image)
    # 用VGG16模型提取特征
    feature = vgg_model.predict(image)
    # 将特征展平为一维向量
    feature = feature.flatten()
    return feature

# 定义一个函数，用来将一个数据集中的所有图片转换为特征矩阵和标签向量
def images_to_features(images, labels):
    # 创建一个空的特征矩阵，行数为图片的数量，列数为特征的维度
    features = np.zeros((len(images), 25088))
    # 遍历每一张图片，提取特征，存入特征矩阵
    for i, image in enumerate(images):
        feature = image_to_feature(image)
        features[i] = feature
    # 将标签向量转换为数组
    labels = np.array(labels)
    return features, labels

# 将验证集中的图片转换为特征矩阵和标签向量
val_features, val_labels = images_to_features(val_images, val_labels)
# 将测试集中的图片转换为特征矩阵和标签向量
test_features, test_labels = images_to_features(test_images, test_labels)
                                              

# 分类或回归：
# 这一步的目的是根据提取的特征，建立一个模型来预测人脸的表情和姿态。
# 可以使用不同的机器学习算法，比如逻辑回归，支持向量机，决策树，随机森林，神经网络等，来实现这个任务。
# 也可以使用scikit-learn或TensorFlow提供的一些现成的模型，或者自己定义一个模型。
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, mean_squared_error

# 定义一个函数，用来评估模型的性能，包括分类准确率和回归均方误差
def evaluate_model(model, features, labels, task):
    # 预测标签
    predictions = model.predict(features)
    # 如果是分类任务，计算准确率
    if task == "classification":
        accuracy = accuracy_score(labels, predictions)
        print("Accuracy:", accuracy)
    # 如果是回归任务，计算均方误差
    elif task == "regression":
        mse = mean_squared_error(labels, predictions)
        print("Mean squared error:", mse)
    # 返回预测结果
    return predictions

# 定义一个逻辑回归模型，用来预测人脸的表情，即笑还是不笑
expression_model = LogisticRegression()
# 使用训练集的特征和标签来训练模型
expression_model.fit(train_features, train_labels)
# 使用验证集的特征和标签来评估模型
print("Expression model performance on validation set:")
expression_predictions = evaluate_model(expression_model, val_features, val_labels, "classification")

# 定义三个支持向量回归模型，用来预测人脸的姿态，即偏航、俯仰和滚动角度
yaw_model = SVR()
pitch_model = SVR()
roll_model = SVR()
# 使用训练集的特征和标签来训练模型
yaw_model.fit(train_features, train_labels[:, 0])
pitch_model.fit(train_features, train_labels[:, 1])
roll_model.fit(train_features, train_labels[:, 2])
# 使用验证集的特征和标签来评估模型
print("Yaw model performance on validation set:")
yaw_predictions = evaluate_model(yaw_model, val_features, val_labels[:, 0], "regression")
print("Pitch model performance on validation set:")
pitch_predictions = evaluate_model(pitch_model, val_features, val_labels[:, 1], "regression")
print("Roll model performance on validation set:")
roll_predictions = evaluate_model(roll_model, val_features, val_labels[:, 2], "regression")


# 数据归一化和增强：这一步的目的是对数据进行一些处理，使其更适合模型的输入和输出。你可以使用一些常用的方法，比如最大最小归一化，标准化，数据增强等，来改善数据的分布和多样性。你可以使用scikit-learn或TensorFlow提供的一些现成的功能，或者自己定义一些函数。例如：
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 定义一个最大最小归一化器，用来将特征的值缩放到0到1之间
feature_scaler = MinMaxScaler()
# 使用训练集的特征来拟合归一化器
feature_scaler.fit(train_features)
# 将归一化器应用到训练集、验证集和测试集的特征上
train_features = feature_scaler.transform(train_features)
val_features = feature_scaler.transform(val_features)
test_features = feature_scaler.transform(test_features)

# 定义一个数据增强器，用来对图片进行一些随机的变换，比如旋转，平移，缩放，翻转等
image_generator = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True)
# 使用数据增强器来生成更多的训练图片
train_images = image_generator.flow_from_directory(cropped_path, target_size=input_size, batch_size=32, subset="training")

# 分析结果：这一步的目的是根据模型的预测结果，对模型的性能和效果进行一些分析和评价。你可以使用一些可视化的方法，比如绘制混淆矩阵，绘制学习曲线，绘制预测结果和真实结果的对比图等，来展示模型的优劣和改进的方向。你可以使用matplotlib或seaborn等库来绘制图形。例如：
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 定义一个函数，用来绘制混淆矩阵，展示分类模型的准确率和误差
def plot_confusion_matrix(labels, predictions, title):
    # 计算混淆矩阵
    cm = confusion_matrix(labels, predictions)
    # 绘制热力图
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    # 设置标题和坐标轴标签
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    # 显示图形
    plt.show()

# 使用测试集的特征和标签来评估表情模型，并绘制混淆矩阵
print("Expression model performance on test set:")
expression_predictions = evaluate_model(expression_model, test_features, test_labels, "classification")
plot_confusion_matrix(test_labels, expression_predictions, "Expression confusion matrix")

# 定义一个函数，用来绘制预测结果和真实结果的对比图，展示回归模型的拟合程度和误差
def plot_regression_results(labels, predictions, title):
    # 绘制散点图，真实结果为蓝色，预测结果为红色
    plt.scatter(labels, predictions, c=["blue", "red"])
    # 绘制对角线，表示完美的预测
    plt.plot([min(labels), max(labels)], [min(labels), max(labels)], c="black")
    # 设置标题和坐标轴标签
    plt.title(title)
    plt.xlabel("True value")
    plt.ylabel("Predicted value")
    # 显示图形
    plt.show()

# 使用测试集的特征和标签来评估姿态模型，并绘制预测结果和真实结果的对比图
print("Yaw model performance on test set:")
yaw_predictions = evaluate_model(yaw_model, test_features, test_labels[:, 0], "regression")
plot_regression_results(test_labels[:, 0], yaw_predictions, "Yaw regression results")
print("Pitch model performance on test set:")
pitch_predictions = evaluate_model(pitch_model, test_features, test_labels[:, 1], "regression")
plot_regression_results(test_labels[:, 1], pitch_predictions, "Pitch regression results")
print("Roll model performance on test set:")
roll_predictions = evaluate_model(roll_model, test_features, test_labels[:, 2], "regression")
plot_regression_results(test_labels[:, 2], roll_predictions, "Roll regression results")

