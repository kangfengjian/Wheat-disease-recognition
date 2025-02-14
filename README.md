# Wheat-disease-recognition


[小麦病害识别](https://www.ssfssp.com/competition/question/detail/task001)

# 赛题相关

及时准确识别小麦病虫害对于精准施药、减灾保产至关重要。常见的小麦病虫害有6种：小麦条锈病（Rust）、小麦白粉病（Powdery Mildew）、小麦赤霉病（Wheat scab）、根冠腐烂（Crown and Root Rot）、小麦黑穗病（Wheat Loose Smut）与小麦纹枯病（wheat sharp eyespot）。健康小麦及染病小麦的典型特征参见以下图例。本赛题旨在研发 “看图识麦” 的模型， 帮助农户从生长期小麦的图片即刻识别病虫害风险。

## 提交手册

各参赛选手需参照算法开发指导文件，在本地自行搭建算法开发、自测环境，并参照镜像封装指导文件，将算法封装成镜像上传至算法评分工具。本次参赛算法需适配NVIDIA T4算力卡。

先把算法模型开发出来，然后构建镜像，上传镜像，进行评分。

### 1 算法训练

### 2 对接

### 3 镜像封装

### 4 提交评测

# 算法训练

# 1. 文件组织

## 1.1. 数据

图像分类项目结构

```json
data文件夹，下设数据集文件夹
	数据集文件夹，以图像类别名为文件夹，文件夹内是该类别的图片
项目文件夹，以项目名称命名
	data文件夹，保存数据相关的文件
		classes.txt 保存类别信息
		train.txt 保存训练图片的信息，每行两个字段，一个是路径一个是类别名。
		test.txt
	data.py 文件，用于处理数据集，生成data文件夹里的各个文件，用于分析数据
	train.py 文件，用于训练模型




有train、test、infer文件夹，classes.txt
train文件夹里是类别名称命名的文件夹，每个文件夹里存放各自类别的图片
test文件夹里是无标签的数据，只有一个文件夹
classes.txt文件夹里是类别名，每个类别名占一行
```

#### 训练文件

训练文件的功能

- 加载数据集

- 加载模型

- 训练

- 处理命令行

###### 定义数据类

写好了训练文件，然后在GPU服务器上训练

#### 推理文件

加载数据

加载模型和参数

推理，结果转换

# 调参技巧

可以调整学习率

调整模型输入大小

数据增强

调整epochs

调整bitchsize

多种方式投票，

交叉验证投票

1. 当损失动荡时，减小学习率
2. 

# 用目标检测做图像分类

与业务场景密切相关
个人认为，分类任务中，前景（目标）与背景的有效信息比例，决定使用分类还是目标检测+。
在个人的实际业务经验中，如果前景尺寸相比背景较小，且背景特征多样性较大，用目标检测，然
后通过backbone提取特征进行分类，精度可以得到保证。
反之，如果前景尺寸占比较大，或者是背景特征较为单一（在一些业务场景，可以保证背景单
一），用单纯分类网络+即可实现足够的精度。
举例来说，做个啤酒品类识别，如果应用在用户自己拍照上传某瓶啤酒并进行识别，那先做个自标
检测，然后冉分类，精度肯定比直接分类来的高不少
而如果是应用在收银台上固定的摄像头拍照摆放的某瓶啤酒，识别品类，那直接进行分类精度也不
会差哪去，毕竟该场景背景单一，儿乎不存在额外干扰信息。

这里我描述的不准确，可能会带来误解。

目标尺寸较小，背景多样性较大的应用场景，要实现分类，建议集成目标检测的方式。以上流程采用one stage还是two stage，这个还是要根据应用场景来决定。以我做过的案例为例，

one stage方法下，目标检测网络输出的就是box+cls或者是box+feature

two stage方法下，目标检测网络输出的是box+cls(一级分类）,后续将box的图像剪裁输入到对应的分类网络提取特征或输出二级分类。

具体采用哪种方法，和场景需求、部署设备资源情况等均有关系

# **API请求与返回示例**

#### 版本信息查询请求

* 请求地址：`GET /api/algo/instance/version`
* 返回示例：

```json
{
    "code": "0",
    "msg": "success",
	"data": {
		"version":"V1.0.0-20250101"
	}
}
```

#### **健康检查请求**

* 请求地址：`GET /api/algo/instance/healthy`
* 返回示例：

```json
{
  "code": "0",
  "msg": "success",
  "data": {
    "status": "ready",
    "msg": "normal"
  }
}
```

#### **图片分类请求**

* 请求地址：`POST /api/algo/task/picture/analysis`
* 请求示例：

```json
{
  "channelCode": "10001",
  "pictureFormat": "JPEG",
  "pictureType": 1,
  "pictureData": "http://192.168.102.59:20000/2k/1003.jpeg"
}

```

* 返回示例：

```json
{
  "code": "0",
  "msg": "success",
  "data": {
		"class": 817,
		"prob": 0.5135124325752258
  }
}
```

# 打包

环境中已有docker，那么可以直接用

先拉取镜像，再运行容器

docker pull reg.sdses.com/nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

人家内部镜像我们不能用，然后可以自己搞一个，那么就选择pytorch的镜像

[pytorch/pytorch - Docker Image](https://hub.docker.com/r/pytorch/pytorch)

找镜像，tag形如：`1.10.0-cuda11.3-cudnn8-runtime`

1.10.0 是pytorch的版本；

cuda11.3 是cuda的版本，意味着镜像中已经安装了 CUDA 11.3 库

cuDNN 是 NVIDIA 提供的用于加速深度学习计算的高性能库，PyTorch 使用 cuDNN 进行高效的卷积操作、激活函数计算等任务。

runtime
含义：这是镜像的类型，表示这是一个 运行时镜像。
解释：runtime 表示该镜像包含了运行 PyTorch 所需的基本环境和依赖，但它并没有包括用于构建和开发的工具。例如，通常在运行时镜像中不会包含编译器和开发工具链。主要用于部署和运行已经构建好的模型。
这种镜像适用于那些已经准备好进行推理或训练的生产环境，因为它包含运行深度学习模型所需的最小依赖。
与之对应的是其他类型的镜像，比如：
devel：开发环境镜像，包含更多用于构建、编译和开发的工具和依赖（如编译器、构建工具、调试工具等）。用于开发和调试代码。
runtime 镜像没有这些开发工具，通常体积更小，适合生产环境使用。

目前环境的 torch版本是2.6.0

目前环境中是 CUDA Version: 12.0

执行docker pull 拉取镜像

docker pull pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

然后运行容器，

docker run -itd  -v /home/ubuntu/Wheat-disease-recognition:/workspace --name wheat_disease_recognizer pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

> e00974dfabd1af8287a1dd22323b54f96aa62dd7e158e037be5e32b9a109c744

docker exec -it e00 bash

然后在容器中搭建环境

首先是挂在了目录，train.py 这些文件都可以在/workspace里看到

 pip install flask

pip install scikit-learn

没写端口映射，那就重新构建容器

docker stop e00974dfab

docker rm e00974dfab

-p 8080:80

知道咋运行了

然后写Dockerfile

在Dockerfile的目录下执行 docker build -t wheat_disease_recognizer:V0.1.0-20250213 .

docker images查看已经打包好的镜像

docker run -itd  -p 80:80 --gpus all --name wheat_disease_recognizer wheat_disease_recognizer:V0.1.0-20250213

运行报错

Traceback (most recent call last):
  File "/workspace/web_app.py", line 5, in <module>
    from models.ResNet import ResNet
ModuleNotFoundError: No module named 'models'



docker run -itd  -p 80:80 --name wheat_disease_recognizer --entrypoint /bin/bash wheat_disease_recognizer:V0.1.0-20250213



###  保存镜像

docker save -o ./wheat_disease_recognizer_v0.1.1.tar wheat_disease_recognizer:V0.1.1-20250214

再次打包镜像

docker build -t wheat_disease_recognizer:V0.1.1-20250214 .

docker run -itd  -p 80:80 --gpus all --name wheat_disease_recognizer wheat_disease_recognizer:V0.1.1-20250214

# 提交镜像

docker login登录

打标签

docker tag 7d0f8fedb185 kangfengjian/wheat_disease_recognizer:V0.1.0-20250213

docker push kangfengjian/wheat_disease_recognizer:V0.1.0-20250213
