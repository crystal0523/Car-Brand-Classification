# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:35:35 2020

@author: USER
"""

import torchvision
import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data
from dataloader import CarLoader, labelencoder
from test_loader import CarTestLoader
import numpy as np
import csv
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 30
BATCH_SIZE = 16




path='./training_data/'
path1='./testing_data/'

Car=CarLoader(path)
Car1=CarTestLoader(path1)

train_loader=Data.DataLoader(Car,batch_size= BATCH_SIZE,shuffle= True, num_workers=4)
test_loader=Data.DataLoader(Car1,batch_size= BATCH_SIZE,shuffle= True, num_workers=4)


def train(model,learning_rate):
    
    best_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9,weight_decay=5e-4)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr= learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.9, centered=False)
    train_acc=[]
    for epoch in range(EPOCHS):
        
        #model.train()
        correct=0
        total=0     
        
        for i, (images, labels) in enumerate(train_loader):
            
            
            #print('img shape',images.shape,labels.shape)
            images = images.to(device)
            labels = labels.to(device)
        
            optimizer.zero_grad()
            outputs = model(images)
        
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        acc=100 * correct / total 
        train_acc.append(correct / total)
        if acc> best_acc:
             best_acc = acc
             print('model save')
             torch.save({'state_dict': model.state_dict()}, 'resnet50_spp.pth.tar')
             
            
        print ("Epoch [{}/{}],  train_loss: {:.4f}".format(epoch+1, EPOCHS, loss.item()))
        print('train accuracy{} '.format(correct / total))
                     
    print('training complete')
    plt.plot(train_acc)
    plt.xlabel('epoch')
    plt.savefig('resnet50.jpg')

def test(model):
    
    
    checkpoint = torch.load('resnet50_spp.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])#
    model.eval()
    with open('output_resnet50.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id','label'])
       
    with torch.no_grad():
        
        for images,names in test_loader:
            images = images.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            with open('output_resnet50.csv', 'a', newline='') as csvfile:
               
               writer = csv.writer(csvfile)
              
               for i in range(len(predicted)):
                   
                   t=labelencoder.inverse_transform([int(predicted[i])])
                   
                   t=str(t)
                   t=t.strip('[\'\']')
                   print(names[i],t)
                   writer.writerow([names[i],t])
   
 
if __name__=="__main__":
    
    model = torchvision.models.resnet50(pretrained=True)
    model.avgpool=SPPLayer(3)
    model.fc.out_features = 196
    model.to(device)
    print(model)
    train(model,1e-3)
    #test(model)