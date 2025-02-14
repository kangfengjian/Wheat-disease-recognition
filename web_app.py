from flask import Flask, request, jsonify
import base64
#import cv2
import numpy as np
from models.ResNet import ResNet
import torch
import torchvision.transforms as transforms
# from torchvision import models
from io import BytesIO
import json
import requests
from PIL import Image  
import io

# Flask应用初始化
app = Flask(__name__)

class GenericClassifier:
    def __init__(self, weight= ''):
        """
        初始化分类模型。
        :param model_name: 模型名称，例如 'resnet18'
        :param device: 使用的设备，例如 'cpu' 或 'cuda'
        """
        self.device = self.try_gpu()
        # self.transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        self.status = "initializing"
        self.status_msg = "initializing..."
        try:
            self.model = ResNet
            # weight = 'weights/{}best_weights.pth'.format()
            self.model.load_state_dict(torch.load(weight,weights_only=True,map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.status = "ready"
            self.status_msg = "normal"
        except Exception as e:
            self.status = "abnormal"
            self.status_msg = str(e)

    # 检查GPU的可用性
    def try_gpu(self,i=0):
        """如果存在 GPU，则返回第 i 个 GPU，否则返回 CPU"""
        if torch.cuda.device_count() > i:
            return torch.device(f'cuda:{i}')
        return torch.device('cpu')
    
    def load_image(self, image_data, image_type):
        """
        加载图像。
        :param image_data: 图像数据
        :param image_type: 图像类型，0为Base64，1为URL
        :return: OpenCV图像
        """
        if image_type == 0:  # Base64类型
            img_data = base64.b64decode(image_data)
            # nparr = np.frombuffer(img_data, np.uint8)
            # image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image = Image.open(io.BytesIO(img_data))
            # nparr = np.frombuffer(img_response.content, np.uint8)
            # image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image = image.convert('RGB')  # 首先统一为RGB的形式，然后进行处理
            # img = transformer(img)  # 数据增强处理
            image = transforms.Resize((256, 256))(image) # 统一大小
            image = transforms.ToTensor()(image) 
        elif image_type == 1:  # URL类型
            img_response = requests.get(image_data, verify=False)
            print('*'*100)
            print(type(img_response.content))
            print('*'*100)
            image = Image.open(io.BytesIO(img_response.content))
            # nparr = np.frombuffer(img_response.content, np.uint8)
            # image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image = image.convert('RGB')  # 首先统一为RGB的形式，然后进行处理
            # img = transformer(img)  # 数据增强处理
            image = transforms.Resize((256, 256))(image) # 统一大小
            image = transforms.ToTensor()(image) 
        else:
            raise ValueError("Invalid image_type. Supported: 0 (Base64), 1 (URL).")

        if image is None:
            raise ValueError("Failed to load image.")
        return image

    def preprocess(self, image):
        """
        预处理图像。
        :param image: OpenCV图像
        :return: 预处理后的张量
        """
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict(self, input_tensor):
        """
        推理方法。
        :param input_tensor: 输入张量
        :return: 分类结果
        """
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            return probabilities.cpu().numpy()
    
    def process(self, image_data, image_type):
        """
        处理图像数据，返回分类结果。
        :param image_data: 图像数据
        :param image_type: 图像类型，0为Base64，1为URL
        :return: 分类结果
        """
        X = self.load_image(image_data, image_type)
        # img = Image.open('../data/wheat_disease/白粉病/2.jpg')  # 打开图片
        # img = img.convert('RGB')  # 首先统一为RGB的形式，然后进行处理
        # img = transforms.Resize((256, 256))(img) # 统一大小
        # X = transforms.ToTensor()(img)
        X = X.unsqueeze(0)
        X = X.to(self.device)
        y_hat=self.model(X)
        probabilities = torch.softmax(y_hat, dim=1)
        top_idx = torch.argmax(probabilities, dim=1).item()
        return top_idx, probabilities[0][top_idx]

# 单例模式，确保分类器唯一
class ClassifierSingleton:
    _instance = None

    @staticmethod
    def get_instance():
        if ClassifierSingleton._instance is None:
            ClassifierSingleton._instance = GenericClassifier('weights/20250214_110830_best_weights.pth')
        return ClassifierSingleton._instance

# API路由：版本信息
@app.route('/api/algo/instance/version', methods=['GET'])
def query_version_info():
    """
    查询版本信息。
    """
    try:
        version_info = {
            "code": "0",
            "msg": "success",
            "data": {
				"version":"V0.1.0-20250213"
			}
        }
    except Exception as e:
        version_info = {
            "code": "1",
            "msg": f"Error reading version info: {str(e)}",
            "data": {}
        }

    return jsonify(version_info)

# API路由：健康检查    
@app.route('/api/algo/instance/healthy', methods=['GET'])
def get_health_status():
    """
    获取分类器的健康状态
    """
    classifier = ClassifierSingleton.get_instance()
    health_status = {
        "status": classifier.status,
        "msg": classifier.status_msg
    }
    return jsonify({"code": "0", "msg": "success", "data": health_status})

# API路由：推理
@app.route('/api/algo/task/picture/analysis', methods=['POST'])
def picture_analysis():
    """
    执行图片分析任务，具体返回报文根据文档调整。
    """
    data = request.get_json()
    image_type = data.get('pictureType')
    image_data = data.get('pictureData')

    if image_type not in [0, 1]:
        raise ValueError("Invalid imageType. Supported: 0 (Base64), 1 (URL).")
    if not image_data:
        raise ValueError("Missing image data.")

    result, prob = ClassifierSingleton.get_instance().process(image_data, image_type)

    return jsonify({
        "code": "0",
        "msg": "success",
        "data":{
            'expand':{
                "class": int(result),
                "prob": float(prob)
                }
        }
    })
    # try:
    #     data = request.get_json()
    #     image_type = data.get('pictureType')
    #     image_data = data.get('pictureData')

    #     if image_type not in [0, 1]:
    #         raise ValueError("Invalid imageType. Supported: 0 (Base64), 1 (URL).")
    #     if not image_data:
    #         raise ValueError("Missing image data.")

    #     result, prob = ClassifierSingleton.get_instance().process(image_data, image_type)

    #     return jsonify({
    #         "code": "0",
    #         "msg": "success",
    #         "data": {
    #             "class": int(result),
    #             "prob": float(prob)
    #         }
    #     })
    # except Exception as e:
    #     return jsonify({"code": "1", "msg": str(e)}), 400

# 主程序入口
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
