from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split # pip install scikit-learn
from sklearn.model_selection import KFold

'''
将训练集按照1:9划分为训练集和验证集,并将训练集、验证集、测试集以地址形式存储在项目中
当 exist_ok==True 时，若已存在文件，则不重新生成；若不存在则生成
当 exist_ok==False 时，生成
将类别信息存储在classes.txt中
'''

def process(data_path):
    '''针对第一种数据集的处理，这种数据集的组织形式是一个数据集文件夹，里边是各个类别名称的文件夹，文件夹内部是各个图片文件
    程序的目标是生成项目中data目录下的 classes.txt、train.txt、test.txt三个文件
    第一步是读取到所有的类别
    '''
    # 读取文件信息，保存类别和图片信息
    classes = list()
    images = list()
    for item in data_path.rglob('*'):
        if item.is_dir():
            classes.append(item.name)
        else:
            images.append([Path.joinpath(data_path,item.relative_to(data_path)),item.parent.name])
    # 按照1:9划分训练集和验证集
    train_files, val_files = train_test_split(images, test_size=0.1, train_size=0.9, random_state=42, shuffle=True)
    # 生成类别信息
    Path('./data').mkdir(parents=True, exist_ok=True)
    classes.sort()
    with open('./data/classes.txt','w',encoding='utf-8') as wf:
        wf.write('\n'.join(classes))
    # 保存训练集信息
    with open('./data/train.txt', 'w', encoding='utf-8') as wf:
        wf.write('\n'.join([f'{file},{label}' for file, label in train_files]))
    # 保存测试集信息
    with open('./data/test.txt', 'w', encoding='utf-8') as wf:
        wf.write('\n'.join([f'{file},{label}' for file, label in val_files]))

if __name__ =='__main__':
    datasets_root = Path('../data/小麦病害训练集/')
    process(datasets_root)