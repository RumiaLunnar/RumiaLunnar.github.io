---
layout:     post
title:      "Tensorflow notes 0x00"
subtitle:   " \"fashion mnist\""
date:       2019-07-27 8:00:00
author:     "Rumia"
catalog: true
category: blog
tags:
    - tensorflow
    - machine learning
---
> Tensorflow官方出了新的[文档](https://www.tensorflow.org/tutorials)，经典的MNIST手写数字识别被换成了衣物识别，还挺有意思的，大概整理记录一下

# 训练你的第一个神经网络

对于有一定机器学习知识基础的初学者，MNIST手写数字识别无疑是一个非常棒的选择，这里google似乎又把难度降低了一层，改成了所谓的FASHION MNIST服装图像进行分类。

## 1.导入相关库

```python
from __future__ import absolute_import, division, print_function, unicode_literals

# 导入TensorFlow和tf.keras
import tensorflow as tf
from tensorflow import keras

# 导入辅助库
import numpy as np
import matplotlib.pyplot as plt

```

[tips](https://www.liaoxuefeng.com/wiki/897692888725344/923030465280480): 

```python
from __future__ import ...
# 意思是把下一个版本的特性导入当前版本
```

## 2.导入 Fashion MNIST数据集

本指南使用[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) 数据集，其中包含了10个类别中共70,000张灰度图像。图像包含了低分辨率（28 x 28像素）的单个服装物品，如下所示:

![Fashion mnist](https://www.tensorflow.org/images/fashion-mnist-sprite.png)

Fashion MNIST 旨在替代传统的[MNIST](http://yann.lecun.com/exdb/mnist/)数据集 — 它经常被作为机器学习在计算机视觉方向的"Hello, World"。MNIST数据集包含手写数字（0,1,2等）的图像，其格式与我们在此处使用的服装相同。

本指南使用Fashion MNIST进行多样化，因为它比普通的MNIST更具挑战性。两个数据集都相对较小，用于验证算法是否按预期工作。它们是测试和调试代码的良好起点。

我们将使用60,000张图像来训练网络和10,000张图像来评估网络模型学习图像分类任务的准确程度。您可以直接从TensorFlow使用Fashion MNIST，只需导入并加载数据

```python
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  ## 此处可能需要一些小tricks才能正常下载
```



​	加载数据集并返回四个NumPy数组:

- `train_images`和`train_labels`数组是*训练集*—这是模型用来学习的数据。
- 模型通过*测试集*进行测试, 即`test_images`与 `test_labels`两个数组。

图像是28x28 NumPy数组，像素值介于0到255之间。*labels*是一个整数数组，数值介于0到9之间。这对应了图像所代表的服装的*类别*:

| 标签 | 类别        |
| ---- | ----------- |
| 0    | T-shirt/top |
| 1    | Trouser     |
| 2    | Pullover    |
| 3    | Dress       |
| 4    | Coat        |
| 5    | Sandal      |
| 6    | Shirt       |
| 7    | Sneaker     |
| 8    | Bag         |
| 9    | Ankle boot  |

每个图像都映射到一个标签。由于*类别名称*不包含在数据集中,因此把他们存储在这里以便在绘制图像时使用:

```python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

## 3.探索数据

让我们在训练模型之前探索数据集的格式。以下显示训练集中有60,000个图像，每个图像表示为28 x 28像素:

```python
train_images.shape
```

```
(60000, 28, 28)
```

同样，训练集中有60,000个标签:

```python
len(train_labels)
```

```
60000
```

每个标签都是0到9之间的整数:

```python
train_labels
```

```
array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
```

测试集中有10,000个图像。 同样，每个图像表示为28×28像素:

```python
test_images.shape
```

```
(10000, 28, 28)
```

测试集包含10,000个图像标签:

```python
len(test_labels)
```

```
10000
```

## 4.数据预处理

在训练网络之前必须对数据进行预处理。 如果您检查训练集中的第一个图像，您将看到像素值落在0到255的范围内:

```python
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
```

![](https://www.tensorflow.org/tutorials/keras/basic_classification_files/output_m4VEw8Ud9Quh_0.png)

在馈送到神经网络模型之前，我们将这些值缩放到0到1的范围。为此，我们将像素值值除以255。重要的是，对*训练集*和*测试集*要以相同的方式进行预处理:

```python
train_images = train_images / 255.0test_images = test_images / 255.0
```



显示*训练集*中的前25个图像，并在每个图像下方显示类名。验证数据格式是否正确，我们是否已准备好构建和训练网络。

```python
plt.figure(figsize=(10,10))
for i in range(25):    
    plt.subplot(5,5,i+1)    
    plt.xticks([])    
    plt.yticks([])    
    plt.grid(False)    
    plt.imshow(train_images[i], cmap=plt.cm.binary)   		     plt.xlabel(class_names[train_labels[i]])
plt.show()
```

![png](https://www.tensorflow.org/tutorials/keras/basic_classification_files/output_oZTImqg_CaW1_0.png)

## 5.构建模型

构建神经网络需要配置模型的层，然后编译模型。

设置网络层

一个神经网络最基本的组成部分便是*网络层*。网络层从提供给他们的数据中提取表示，并期望这些表示对当前的问题更加有意义

大多数深度学习是由串连在一起的网络层所组成。大多数网络层，例如`tf.keras.layers.Dense`，具有在训练期间学习的参数。

```python
model = keras.Sequential([    keras.layers.Flatten(input_shape=(28, 28)),    keras.layers.Dense(128, activation=tf.nn.relu),    keras.layers.Dense(10, activation=tf.nn.softmax)])
```



网络中的第一层, `tf.keras.layers.Flatten`, 将图像格式从一个二维数组(包含着28x28个像素)转换成为一个包含着28 * 28 = 784个像素的一维数组。可以将这个网络层视为它将图像中未堆叠的像素排列在一起。这个网络层没有需要学习的参数;它仅仅对数据进行格式化。

在像素被展平之后，网络由一个包含有两个`tf.keras.layers.Dense`网络层的序列组成。他们被称作稠密链接层或全连接层。 第一个`Dense`网络层包含有128个节点(或被称为神经元)。第二个(也是最后一个)网络层是一个包含10个节点的*softmax*层—它将返回包含10个概率分数的数组，总和为1。每个节点包含一个分数，表示当前图像属于10个类别之一的概率。

编译模型

在模型准备好进行训练之前，它还需要一些配置。这些是在模型的*编译(compile)*步骤中添加的:

- *损失函数* —这可以衡量模型在培训过程中的准确程度。 我们希望将此函数最小化以"驱使"模型朝正确的方向拟合。
- *优化器* —这就是模型根据它看到的数据及其损失函数进行更新的方式。
- *评价方式* —用于监控训练和测试步骤。以下示例使用*准确率(accuracy)*，即正确分类的图像的分数。

```python
model.compile(optimizer='adam',        loss='sparse_categorical_crossentropy',              metrics=['accuracy'])
```



训练模型

训练神经网络模型需要以下步骤:

1. 将训练数据提供给模型 - 在本案例中，他们是`train_images`和`train_labels`数组。
2. 模型学习如何将图像与其标签关联
3. 我们使用模型对测试集进行预测, 在本案例中为`test_images`数组。我们验证预测结果是否匹配`test_labels`数组中保存的标签。

通过调用`model.fit`方法来训练模型 — 模型对训练数据进行"拟合"。

```python
model.fit(train_images, train_labels, epochs=5)
```

随着模型训练，将显示损失和准确率等指标。该模型在训练数据上达到约0.88(或88％)的准确度。

进行预测

通过训练模型，我们可以使用它来预测某些图像。

```python
predictions = model.predict(test_images)
```



在此，模型已经预测了测试集中每个图像的标签。我们来看看第一个预测:

```python
predictions[0]
```



```
array([6.6858855e-05, 2.5964803e-07, 5.3627105e-06, 4.5019146e-06,
       2.7420206e-06, 4.7881842e-02, 2.3233067e-04, 5.4705784e-02,
       8.5581087e-05, 8.9701480e-01], dtype=float32)
```



预测是10个数字的数组。这些描述了模型的"信心"，即图像对应于10种不同服装中的每一种。我们可以看到哪个标签具有最高的置信度值：

```python
np.argmax(predictions[0])
```



```
9
```



因此，模型最有信心的是这个图像是ankle boot，或者 `class_names[9]`。 我们可以检查测试标签，看看这是否正确:

```python
test_labels[0]
```



```
9
```



我们可以用图表来查看全部10个类别

```python
def plot_image(i, predictions_array, true_label, img):  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]  plt.grid(False)  plt.xticks([])  plt.yticks([])    plt.imshow(img, cmap=plt.cm.binary)    predicted_label = np.argmax(predictions_array)  if predicted_label == true_label:    color = 'blue'  else:    color = 'red'    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],                                100*np.max(predictions_array),                                class_names[true_label]),                                color=color)def plot_value_array(i, predictions_array, true_label):  predictions_array, true_label = predictions_array[i], true_label[i]  plt.grid(False)  plt.xticks([])  plt.yticks([])  thisplot = plt.bar(range(10), predictions_array, color="#777777")  plt.ylim([0, 1])  predicted_label = np.argmax(predictions_array)    thisplot[predicted_label].set_color('red')  thisplot[true_label].set_color('blue')
```



让我们看看第0个图像，预测和预测数组。

```python
i = 0plt.figure(figsize=(6,3))plt.subplot(1,2,1)plot_image(i, predictions, test_labels, test_images)plt.subplot(1,2,2)plot_value_array(i, predictions,  test_labels)plt.show()
```



![png](https://www.tensorflow.org/tutorials/keras/basic_classification_files/output_HV5jw-5HwSmO_0.png)

```python
i = 12plt.figure(figsize=(6,3))plt.subplot(1,2,1)plot_image(i, predictions, test_labels, test_images)plt.subplot(1,2,2)plot_value_array(i, predictions,  test_labels)plt.show()
```



![png](https://www.tensorflow.org/tutorials/keras/basic_classification_files/output_Ko-uzOufSCSe_0.png)

让我们绘制几个图像及其预测结果。正确的预测标签是蓝色的，不正确的预测标签是红色的。该数字给出了预测标签的百分比(满分100)。请注意，即使非常自信，也可能出错。

```python
# 绘制前X个测试图像，预测标签和真实标签# 以蓝色显示正确的预测，红色显示不正确的预测num_rows = 5num_cols = 3num_images = num_rows*num_colsplt.figure(figsize=(2*2*num_cols, 2*num_rows))for i in range(num_images):  plt.subplot(num_rows, 2*num_cols, 2*i+1)  plot_image(i, predictions, test_labels, test_images)  plt.subplot(num_rows, 2*num_cols, 2*i+2)  plot_value_array(i, predictions, test_labels)plt.show()
```



![png](https://www.tensorflow.org/tutorials/keras/basic_classification_files/output_hQlnbqaw2Qu__0.png)

最后，使用训练的模型对单个图像进行预测。

```python
# 从测试数据集中获取图像img = test_images[0]print(img.shape)
```



```
(28, 28)
```



`tf.keras`模型经过优化，可以一次性对*批量*,或者一个集合的数据进行预测。因此，即使我们使用单个图像，我们也需要将其添加到列表中:

```python
# 将图像添加到批次中，即使它是唯一的成员。img = (np.expand_dims(img,0))print(img.shape)
```



```
(1, 28, 28)
```



现在来预测图像:

```python
predictions_single = model.predict(img)print(predictions_single)
```



```
[[6.6858927e-05 2.5964729e-07 5.3627055e-06 4.5019060e-06 2.7420206e-06
  4.7881793e-02 2.3233047e-04 5.4705758e-02 8.5581087e-05 8.9701480e-01]]
```



```python
plot_value_array(0, predictions_single, test_labels)plt.xticks(range(10), class_names, rotation=45)plt.show()
```



![png](https://www.tensorflow.org/tutorials/keras/basic_classification_files/output_6Ai-cpLjO-3A_0.png)

`model.predict`返回一个包含列表的列表，每个图像对应一个列表的数据。获取批次中我们(仅有的)图像的预测:

```python
prediction_result = np.argmax(predictions_single[0])print(prediction_result)
```



```
9
```



而且，和之前一样，模型预测标签为9。  