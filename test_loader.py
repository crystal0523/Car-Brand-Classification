# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 13:17:56 2020

@author: USER
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:57:06 2020

@author: crystal
"""
import torchvision
import pandas as pd
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch
import os
import torch.utils.data as Data
import matplotlib.pyplot as plt

path_to_test_data = './testing_data/'


def getData():
    
    datalist = [f.split('.jpg')[0] for f in os.listdir(path_to_test_data)]
    pic_id = []
    
    for pic in datalist:
        pic_id.append(pic)
    df = pd.DataFrame(pic_id,columns=['id'])
    img = df['id']
    
    return np.squeeze(img.values)#刪除數組形狀中的單維度條目

#getData()
#print(a, b)


class CarTestLoader(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.img_name = getData()
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        path = self.root + str(self.img_name[index]) + '.jpg'
        img = Image.open(path)
        img = img.convert('RGB')
        #print(img.size)
        
        transform2 = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(), #Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        img = transform2(img)
      
        return img,self.img_name[index]
    
