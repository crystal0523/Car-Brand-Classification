
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
    
    df= pd.read_csv('training_labels.csv')
    img = df['id']
    #print(img)
    label = labelencoder.fit_transform(df['label'])
    #print('transform to integer',label)
    #print(max(label))
    onehot_encoder = OneHotEncoder(sparse=False)
    label = label.reshape(len(label), 1)
    onehot_encoded = onehot_encoder.fit_transform(label)
    return np.squeeze(img.values), np.squeeze(onehot_encoded)#刪除數組形狀中的單維度條目

#a,b=getData()
#print(a,b)
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

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
       
        length=len(str(self.img_name[index]))
        x=6-length
        path=self.root+'0'*x+str(self.img_name[index])+'.jpg'
        img = Image.open(path)
        img=img.convert('RGB')#'1'
        #print(img)
        
        #width = img.size[0]   # 获取宽度
        #height = img.size[1]   # 获取高度
        #img = img.resize((int(width*0.2), int(height*0.2)), Image.ANTIALIAS)
        #print(img.size)
        #data augmentation會在dataloader裡面(for imgs, labels in enumeratr(train_loader)裡面做，也就是圖片量會是原來的倍數)
        transform2 = transforms.Compose([
            transforms.Resize((600,600)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15, resample=Image.BICUBIC, expand=False),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
            #AddGaussianNoise(0., 1.)])
            
        
        img=transform2(img)
        #img.show()
       
        label=self.label[index]
        label=np.argmax(label, axis=0)
        #print(label,type(label))
        #print('shape:',img.shape)
        #print('img',img)
        #print(self.img_name[index],labelencoder.inverse_transform([int(label)]))
        return img, label
    
#path='./training_data/'
#Car=CarLoader(path)


#Car.__getitem__(1000)



'''#影象一半的概率翻轉，一半的概率不翻轉
            
           transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
            
            Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
           
        transforms.ColorJitter(brightness=(0, 36), contrast=(0, 10), saturation=(0, 25), hue=(-0.5, 0.5)),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
            ])image = (image - mean) / std, mean,std:0.5,0.5 歸一化到[-1.0, -1.0]
train_loader=Data.DataLoader(Car,batch_size= 10,shuffle= True, num_workers=0)
start = time.time()
'''