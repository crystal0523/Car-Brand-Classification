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
path_to_test_data='./testing_data/'

def getData():
    
    datalist = [f.split('.jpg')[0] for f in os.listdir(path_to_test_data)]
    pic_id=[]
    
    for pic in datalist:
        pic_id.append(pic)
    #print(pic_id)
    df =pd.DataFrame(pic_id,columns=['id'])
    #print(df)
    img = df['id']
    return np.squeeze(img.values)#刪除數組形狀中的單維度條目

getData()
#print(a,b)

class CarTestLoader(data.Dataset):
    def __init__(self, root):
        
        self.root = root
        self.img_name= getData()
        #print(self.img_name)
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
       
        
        path=self.root+str(self.img_name[index])+'.jpg'
        img = Image.open(path)
        img=img.convert('RGB')
        #print(img.size)
        
        width = img.size[0]   # 获取宽度
        height = img.size[1]   # 获取高度
        #img = img.resize((int(width*0.2), int(height*0.2)), Image.ANTIALIAS)
        #print(img.size)
        transform2 = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(), #Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        img=transform2(img)
      
        return img,self.img_name[index]
    
'''path='./testing_data/'
Car=CarTestLoader(path)
test_loader=Data.DataLoader(Car,batch_size= 10,shuffle= True, num_workers=0)

def show_img(img):
    plt.figure(figsize=(18,15))
    # unnormalize
    img = img / 2 + 0.5  
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.imshow(np.transpose(npimg))
    plt.show()
data = iter(test_loader)
images = data.next()

show_img(torchvision.utils.make_grid(images))'''