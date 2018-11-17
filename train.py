import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data


import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import time
import copy
import argparse
from collections import OrderedDict

ap = argparse.ArgumentParser(description='Train.py')

ap.add_argument('data_dir', nargs='*', action="store", default="./aipnd-project/flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.0005)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=3)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)

pa = ap.parse_args()

where = pa.data_dir
path = pa.save_dir
learnr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs

if structure != "vgg16" :
    raise ValueError('Unexpected network architecture. Only vgg16 is supported.', arch)

def load_data(where  = "./flowers" ):

    data_dir = where
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Apply the required transfomations to the test dataset in order to maximize the efficiency of the learning
    #process


    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                      [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                      [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    image_datasets = datasets.ImageFolder(data_dir + '/train', transform=data_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)


    return trainloaders , validloader, testloader

trainloaders, validloader, testloader = load_data(where)

def nn_setup(structure='vgg16',dropout=0.5, hidden_layer1 = 120, learnr = 0.0005):
   
    if structure != "vgg16" :
        print("Im sorry but {} is not a valid model.Only vgg16 is supported.".format(structure))
    else:
        model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
        
        classifier = nn.Sequential(OrderedDict([ ('fc1', nn.Linear(25088, 4069)), 
                                        ('relu', nn.ReLU()), 
                                        ('dropout', nn.Dropout(dropout)), 
                                        ('fc2', nn.Linear(4069, 1024)), 
                                        ('relu', nn.ReLU()), 
                                        ('dropout', nn.Dropout(dropout)), 
                                        ('fc3', nn.Linear(1024, hidden_layer1)), 
                                        ('output', nn.LogSoftmax(dim=1)) ]))
    
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learnr)
        model.cuda()
        
        return model, criterion, optimizer

model, optimizer, criterion = nn_setup(structure,dropout,hidden_layer1,learnr)
    
def train_network(model, criterion, optimizer, epochs = 3, print_every=20, loader=trainloaders):
    epochs = epochs
    print_every = 20
    steps = 0

     # change to cuda
    model.to('cuda')

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloaders):
            steps += 1
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(running_loss)
            if steps % print_every == 0:
                model.eval()
                vlost = 0
                accuracy=0

                for ii, (inputs2,labels2) in enumerate(validloader) :
                    optimizer.zero_grad()
                    inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                    model.to('cuda:0')
                    with torch.no_grad():    
                        outputs = model.forward(inputs2)
                    vlost = criterion(outputs,labels2)
                    ps = torch.exp(outputs).data
                    equality = (labels2.data == ps.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
                vlost = vlost / len(validloader)
                accuracy = accuracy /len(validloader)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                                  "Loss: {:.4f}".format(running_loss/print_every),
                                  "Validation Lost {:.4f}".format(vlost),
                                  "Accuracy: {:.4f}".format(accuracy))
                          
                running_loss = 0
                        

train_network(model, optimizer, criterion, epochs, 20, trainloaders)
                            
def save_checkpoint(path='checkpoint.pth',structure ='vgg16', hidden_layer1=120,dropout=0.5,lr=0.0005,epochs=3):
    model.class_to_idx = image_datasets.class_to_idx
    checkpoint = {'arch': 'vgg16',
                  'input_size': 25088,
                  'output_size': hidden_layer1,
                  'dropout':dropout,
                  'epochs' : epochs,
                  'learning_rate' : lr,
                  'optimizer_dict' : optimizer.state_dict,
                  'class_to_idx' : model.class_to_idx,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, 'checkpoint.pth')

save_checkpoint(path,structure,hidden_layer1,dropout,lr)

print("All Set and Done. The Model is trained and checkpoint is saved.")

def load_checkpoint(path='checkpoint.pth'):
        checkpoint = torch.load(path)
        if checkpoint['arch'] == 'vgg16':
            model = models.vgg16(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False

        model.class_to_idx= checkpoint['class_to_idx']
        lr= checkpoint['learning_rate']
        epochs = checkpoint['epochs']
        dropout = checkpoint['dropout']
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

