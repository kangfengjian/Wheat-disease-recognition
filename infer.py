import torch
from torch import nn
# from d2l import torch as d2l
from pathlib import Path
from torchvision import datasets, transforms  
from sklearn.model_selection import train_test_split # pip install scikit-learn 
from torch.utils.data import Dataset, DataLoader  
from PIL import Image  
import json
from datetime import datetime
import argparse 
from torch.nn import functional as F
import time
import mimetypes
from models.ResNet import ResNet

# 数据加载
class wheatDiseaseDataset(Dataset):  
    def __init__(self, imgs):  
        # self.transforms = [lambda x:x,]
        self.data = list()
        if not(isinstance(imgs,list) or isinstance(imgs,tuple)):  # 将非列表和元组元素转变为列表
            imgs = [imgs]
        if isinstance(imgs[0],str) or isinstance(imgs[0],Path): # 如果列表中的元素是字符串，说明是图片的路径
            pics = list()  # 这个列表里存储图片的路径，文件路径
            for pic in imgs:
                pic_path = Path(pic)
                if pic_path.is_file():
                    pics.append(pic_path)
                elif pic_path.is_dir():
                    pics.extend([i for i in pic_path.iterdir() if is_binary_file(i)])
            for pic in pics:
                # transformer = lambda x:x
                img = Image.open(pic)  # 打开图片
                img = img.convert('RGB')  # 首先统一为RGB的形式，然后进行处理
                # img = transformer(img)  # 数据增强处理
                img = transforms.Resize((256, 256))(img) # 统一大小
                img = transforms.ToTensor()(img) 
                self.data.append([img,str(pic)]) # 原图

    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):  
        img,pic = self.data[idx]
        return img,pic
    

def is_binary_file(file_path):
    mime_type, encoding = mimetypes.guess_type(file_path)
    return mime_type is None or mime_type.startswith('application/') or 'image' in mime_type


# 检查GPU的可用性
def try_gpu(i=0):
    """如果存在 GPU，则返回第 i 个 GPU，否则返回 CPU"""
    if torch.cuda.device_count() > i:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def infer(inferdata_root,weight):
    # 检查设备
    device = try_gpu()
    # print('infer on {}'.format(device))
    # 加载数据
    # print('load data...')
    # inferdata_root = Path('../data/wheat_disease/')
    batch_size = 128
    inferDataset = wheatDiseaseDataset(inferdata_root)
    infer_iter = DataLoader(dataset=inferDataset,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)
    # 加载模型和训练好的参数
    # print('generate model...')
    net  = ResNet
    # weight = 'weights/{}best_weights.pth'.format()
    net.load_state_dict(torch.load(weight,weights_only=True))
    net.to(device)
    # 开始推理
    # print('start infer...')
    net.eval()
    result = list()
    for X,img_name in infer_iter:
        X = X.to(device)
        y_hat=net(X)
        result+=zip(img_name,torch.argmax(torch.softmax(y_hat, dim=1), dim=1).tolist())
    with open('./data/classes.txt','r',encoding='utf-8') as rf:
        classes = [cls.strip() for cls in rf.read().split('\n') if cls]
    result = [[img,classes[label]] for img,label in result]
    return result


if __name__ == '__main__':
    inferdata_root = dict()
    # with open('./data/train.txt','r',encoding='utf-8') as rf:
    #     for line in rf:
    #         inferdata_root[line.split(',')[0]] = line.split(',')[1]
    with open('./data/test.txt','r',encoding='utf-8') as rf:
        for line in rf:
            inferdata_root[line.split(',')[0]] = line.split(',')[1]
    weight = 'weights/20250212_164237_best_weights.pth'
    result = infer(list(inferdata_root.keys()),weight)
    acc = sum(1 for img,label in result if label==inferdata_root[img.replace('\\','/')])/len(result)
    print(acc)