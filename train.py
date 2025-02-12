import torch
from torch import nn
# from d2l import torch as d2l
from pathlib import Path
from torchvision import transforms  
from torch.utils.data import Dataset, DataLoader  
from PIL import Image  
import argparse 
import time
from models.ResNet import ResNet
# import imgaug.augmenters as iaa
import numpy as np
from datetime import datetime


# 定义数据类，实现初始化、len、getitem方法
class wheatDiseaseDataset(Dataset):  
    def __init__(self, img_txt, classes_txt):  
        self.transforms = [lambda x:x,]
        self.data = list()
        with open(classes_txt,'r',encoding='utf-8') as rf:
            self.classes = [cls.strip() for cls in rf.read().split('\n') if cls]
        with open(img_txt,'r',encoding='utf-8') as rf:
            for pic,class_name in [i.split(',') for i in rf.read().split('\n') if i]:
                self.data.append([Path(pic),self.classes.index(class_name),lambda x:x]) # 原图

    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):  
        pic_path,class_idx,transformer = self.data[idx]
        img = Image.open(pic_path)  # 打开图片
        img = img.convert('RGB')  # 首先统一为RGB的形式，然后进行处理
        img = transformer(img)  # 数据增强处理
        img = transforms.Resize((96, 96))(img) # 统一大小
        img = transforms.ToTensor()(img) 
        return img,class_idx
    
# 检查GPU的可用性
def try_gpu(i=0):
    """如果存在 GPU，则返回第 i 个 GPU，否则返回 CPU"""
    if torch.cuda.device_count() > i:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(14, 10),model_name='',index=''):
        """Defined in :numref:`sec_utils`"""
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        self.model_name=model_name
        self.index = index

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        self.fig.savefig('log/log_{}_{}.png'.format(self.model_name,self.index))

def test(net, test_iter, device):
    result = [[],[]]
    net.eval() # 设置为评估模式
    for i,(X,Y) in enumerate(test_iter):
        X,Y = X.to(device),Y.to(device)
        y_hat = net(X)
        result[0]+=Y.tolist()
        result[1]+=torch.argmax(torch.softmax(y_hat, dim=1), dim=1).tolist()
    acc = sum(result[0][x]==result[1][x] for x in range(len(result[0])))/len(result[0])
    return acc


        
    
def process_bar(progress,total):
        # 计算当前进度条的百分比  
        percent = (progress / total) * 100  
        # 构造进度条字符串，使用空格来填充剩余部分  
        bar = f'[{">" * int(percent)}' + ' ' * (100 - int(percent)) + f']'  +' {}/{}'.format(progress,total)
        # 使用\r回到行首，然后打印进度条  
        print(f'\r{bar}', end='', flush=True)    

def init_weights(m):
    if type(m) == nn.Linear or type(m)==nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)



if __name__ == '__main__':
    # 检查设备
    device = try_gpu()
    print('train on {}'.format(device))
    # 加载数据
    print('load data...')
    datasets_root = Path('../data/traffic_sign/')
    data_root = Path('./data/traffic_sign_B/')
    # 加载数据
    batch_size = 128
    trainDataset = wheatDiseaseDataset('./data/train.txt','./data/classes.txt')
    train_iter = DataLoader(dataset=trainDataset,batch_size=batch_size,shuffle=True)
    testDataset = wheatDiseaseDataset('./data/test.txt','./data/classes.txt')
    test_iter = DataLoader(dataset=testDataset,batch_size=batch_size,shuffle=True)
    # 构建模型
    print('generate model...')
    lr, num_epochs = 0.05, 1
    net  = ResNet
    net.apply(init_weights)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    # 开始训练
    print('start train...')
    # model_name = 'ResNet-B-0002'
    # animator = Animator(xlabel='epoch',xlim=[1,num_epochs],legend=['train loss','train acc','test acc'],model_name=model_name,index=cross_index)
    num_batches = len(train_iter)
    best_acc = 0
    # with open('log/log_{}_{}.log'.format(model_name,cross_index),'w',encoding='utf-8') as wf:
        # wf.write('')
    for epoch in range(num_epochs):
        # start_time = time.perf_counter()
        # out_str = '{}-cross_{}_epoch:{}/{}\t'.format(model_name,cross_index,epoch,num_epochs)
        # metric = d2l.Accumulator(3)
        result = [[],[],[]]
        net.train()
        for i,(X,Y) in enumerate(train_iter):
            X,Y = X.to(device),Y.to(device)
            y_hat = net(X)
            l = loss(y_hat,Y)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            result[0]+=Y.tolist()
            result[1]+=torch.argmax(torch.softmax(y_hat, dim=1), dim=1).tolist()
            result[2]+=[l.tolist()]
            process_bar(i,num_batches)
        process_bar(i+1,num_batches)
        print()
        acc = sum(result[0][x]==result[1][x] for x in range(len(result[0])))/len(result[0])
        avg_loss = sum(result[2])/len(result[2])
        test_acc = test(net,test_iter,device)
        print(f'epoch {epoch+1}:loss {avg_loss:.3f},train acc {acc:.3f},test acc {test_acc:.3f}')
        if test_acc>best_acc:
            best_acc = test_acc
            weights_path = Path('./weights/')
            weights_path.mkdir(parents=True, exist_ok=True)
            if best_acc>0:
                torch.save(net.state_dict(),weights_path/Path('{}_best_weights.pth'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))))
        # torch.save(net.state_dict(), weights_path/Path('{}_{}_last_weights.pth'.format(model_name,index)))
    #     animator.add(epoch+1,(None,None,test_acc))
    #     with open('log/log_{}_{}.log'.format(model_name,cross_index),'a',encoding='utf-8') as wf:
    #         wf.write(out_str+'\n')
    #     process_time = time.perf_counter()-start_time
    #     out_str+=f'loss {train_l:.3f},train acc {train_acc:.3f},'f'test acc {test_acc:.3f},'f'time {process_time:.3f}s'
    #     print(f'\r'+' '*120, end = '', flush=True)  
    #     print(f'\r'+out_str)  
    # print(weights_path/Path('{}_{}_best_weights.pth'.format(model_name,cross_index)))
    