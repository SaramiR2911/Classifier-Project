import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data

import json
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import time
import copy
import argparse
from collections import OrderedDict

import matplotlib.pyplot as plt

import PIL
from PIL import Image 

ap = argparse.ArgumentParser(
    description='predict-file')
ap.add_argument('--input_img', default='./flowers/test/13/image_05787.jpg', nargs='*', action="store", type = str)
ap.add_argument('checkpoint', default='/home/workspace/checkpoint.pth', nargs='*', action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')

pa = ap.parse_args()
path_image = pa.input_img
number_of_outputs = pa.top_k
path = pa.checkpoint

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    
    model.class_to_idx= checkpoint['class_to_idx']
    lr= checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    dropout = checkpoint['dropout']
    hidden_layer1 = checkpoint['output_size']
    classifier = nn.Sequential(OrderedDict([ ('fc1', nn.Linear(25088, 4069)), 
                                            ('relu', nn.ReLU()), 
                                            ('dropout', nn.Dropout(dropout)), 
                                            ('fc2', nn.Linear(4069, 1024)), 
                                            ('relu', nn.ReLU()), 
                                            ('dropout', nn.Dropout(dropout)), 
                                            ('fc3', nn.Linear(1024, hidden_layer1)), 
                                            ('output', nn.LogSoftmax(dim=1)) ]))
    # Put the classifier on the pretrained network
    model.classifier = classifier
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
   
    return model

model = load_checkpoint(path)

with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)
    
def process_image(path_image):
    im = Image.open(path_image)    
    image_adj = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                      [0.229, 0.224, 0.225])])
    
    img_tensor = image_adj(im)
    return img_tensor
    
img = process_image(path_image)

def predict(image_path, model, number_of_outputs=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image_pred = process_image(image_path)# open the image and process it
    
    image_pred.unsqueeze_(0)
    outputs = model(image_pred)
    probs = torch.exp(outputs)
    top_probs, top_labs = probs.topk(number_of_outputs, dim=1)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labs, top_flowers

probs, labels, flowers = predict(path_image, model, number_of_outputs )
print(probs, labels, flowers)

def plot_solution(path_image, model):
    # Set up plot
    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)
    # Set up title
    flower_num = image_path.split('/')[2]
    title_ = cat_to_name[flower_num]
    # Plot flower
    img = process_image(image_path)
    imshow(img, ax, title = title_)
    # Make prediction
    probs, labs, flowers = predict(image_path, model) 
    # Plot bar chart
    plt.subplot(2,1,2)
    sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0]);
    plt.show()
    
plot_solution(path_image, model)
