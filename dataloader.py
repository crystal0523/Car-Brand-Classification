
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:57:06 2020

@author: crystal
"""
import torchvision
import time
import pandas as pd
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt

labelencoder = LabelEncoder()


def getData():
    df = pd.read_csv('training_labels.csv')
    img = df['id']
    #print(img)
    label = labelencoder.fit_transform(df['label'])
    onehot_encoder = OneHotEncoder(sparse=False)
    label = label.reshape(len(label), 1)
    onehot_encoded = onehot_encoder.fit_transform(label)
    
    return np.squeeze(img.values), np.squeeze(onehot_encoded)#刪除數組形狀中的單維度條目

#a,b=getData()
#print(a,b)


class CarLoader(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.img_name, self.label = getData()
        #print(self.img_name)
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        length = len(str(self.img_name[index]))
        x = 6-length
        path = self.root+'0'*x + str(self.img_name[index]) + '.jpg'
        img = Image.open(path)
        img = img.convert('RGB')#'1'
       
        #data augmentation會在dataloader裡面(for imgs, labels in enumeratr(train_loader)裡面做，也就是圖片量會是原來的倍數)
        transform2 = transforms.Compose([
            transforms.Resize((600,600)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15, resample=Image.BICUBIC, expand=False),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
        img = transform2(img)
        #img.show()
        label = self.label[index]
        label = np.argmax(label, axis=0)
        #print(self.img_name[index],labelencoder.inverse_transform([int(label)]))
        
        return img, label
    
#path='./training_data/'
#Car=CarLoader(path)
#Car.__getitem__(1000)

