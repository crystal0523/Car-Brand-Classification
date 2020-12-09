# Car-Brand-Classification
Use pretrained wide residual network to do car brand classification problem
## Dataset
This project use a lot of pictures of different car brand, including 11,185 training images and 5,000 testing images.
<p>-------------------
  
  <img src='000074.jpg'>  
  <img src='000099.jpg'>
  <img src='0000160.jpg'>
  
</p>-------------------

## Environment
```
torch 1.6.0
```
## Installation
Clone this repo.
  
  
## Pretrained model
 By the following command
 ```
 model = torchvision.models.wide_resnet50_2(pretrained=True)
 ```
## Train
Run ```wide_residual_network.py```
## Testing Results
Since it's a kaggle competition, you have to submit your result on kaggle website.
  
