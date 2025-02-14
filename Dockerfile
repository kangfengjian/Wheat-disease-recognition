FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# 设置工作目录
WORKDIR /workspace

#设置时区
# ENV TZ=Asia/Shanghai
# ARG DEBIAN_FRONTEND=noninteractive

# 复制到容器中
# COPY Source/classification.py /home/sdses/
# COPY Build/imagenet_classes.txt /home/sdses/
# COPY Build/resnet18-f37072fd.pth /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
# COPY Build/torch-1.11.0+cu113-cp38-cp38-linux_x86_64.whl /tmp/
# COPY Build/torchvision-0.12.0+cu113-cp38-cp38-linux_x86_64.whl /tmp/

COPY ./ /workspace/ 

# 安装所需的依赖和工具
# COPY Build/3bf863cc.pub /home/sdses
# COPY Build/7fa2af80.pub /home/sdses
# RUN apt-key add 3bf863cc.pub \
#     && apt-key add 7fa2af80.pub 

# RUN apt-get update
# RUN apt-get install -y software-properties-common && \
#     add-apt-repository -y ppa:deadsnakes/ppa && \
#     apt-get update && \
#     apt-get install -y python3.8 python3.8-dev python3-pip && \
#     apt-get install -y python3-distutils python3-setuptools && \
#     update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
#     update-alternatives --set python3 /usr/bin/python3.8 && \
#     ln -s /usr/bin/pip3 /usr/bin/pip 
# RUN apt-get install -y wget vim openssh-server git libgl1 libgl1-mesa-glx libglib2.0-0 iputils-ping build-essential cmake libjpeg-dev zlib1g-dev libpng-dev
# RUN rm -rf /var/lib/apt/lists/*
# RUN rm -f /home/sdses/3bf863cc.pub /home/sdses/7fa2af80.pub

# 安装 Python 包管理工具和基础依赖
# RUN pip install --no-cache-dir --upgrade pip setuptools \
    # && pip install --no-cache-dir Cython

# 安装 pip 依赖
# RUN pip3 install /tmp/torch-1.11.0+cu113-cp38-cp38-linux_x86_64.whl && \
#     pip3 install /tmp/torchvision-0.12.0+cu113-cp38-cp38-linux_x86_64.whl && \
#     rm -rf /tmp/*.whl

# 安装Python依赖包
# RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-build
# RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple Opencv-python==4.10.0.84
# RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple Flask==3.0.3
RUN pip install flask
RUN pip install scikit-learn
RUN pip install torchvision

# 启动入口
ENTRYPOINT ["python"]
CMD ["web_app.py"]
