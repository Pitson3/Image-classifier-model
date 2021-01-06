# Imports here
import matplotlib.pyplot as plt 

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
import numpy as np
from PIL import Image 
import os, random
from torch.autograd import Variable

import argparse

# Define command line arguments
parser = argparse.ArgumentParser(description="Trains a new network on a dataset and save the model as a checkpoint")
parser.add_argument('data_dir', type=str, help='The path to the dataset directory')
parser.add_argument('--gpu', action='store_true', help='Use an optional gpu if available')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--print_rate', type=int, help='Print rate')
parser.add_argument('--arch', type=str, help='Model architecture')
parser.add_argument('--learning_rate', type=float, help='Learning rate value')
parser.add_argument('--hidden_units', type=int, help='Number of hidden layer\'s units')
parser.add_argument('--checkpoint', type=str, help='Commit the trained model checkpoint to file')

#Convert argument strings to objects and assign them as attributes of the namespaces then return the populated namespace without errors on additional args
args, _ = parser.parse_known_args()

def load_model(arch_name='vgg16',output_labels=102,hidden_units=1000):
    """
    Load a trained model
    """
    
    #Step 1:Loading a pretrained network, i.e. vgg16
    #Inspired by the firt project in this course and 
    if args.arch == 'vgg16' or arch_name=='vgg16':
        model = models.vgg16(pretrained=True)
    elif args.arch == 'resnest18' or arch_name=='resnet18':
        model = models.resnet18(pretrained=True)
    elif args.arch == 'alexnet' or arch_name=='alexnet':
        model = models.alexnet(pretrained=True)
    else:
        raise ValueError(arch_name, " is currently not supported try revisting python train() -h for help")
    

    #Sub-step 1:1 -> Freeze the parameters to escape backpropagating through them 
    for param in model.parameters():
        param.requires_grad = False

    
    #Step 2->Define  a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    classifier = nn.Sequential(nn.Linear(25088, hidden_units), nn.ReLU(),
                           nn.Dropout(p=0.07),
                           nn.Linear(hidden_units,output_labels),
                           nn.LogSoftmax(dim=1)
                          )

    #Sub-step 2.1-> Revamp the classifier with the new classifier 
    model.classifier = classifier

    return model, classifier
    
def train(image_datasets, arch_name='vgg16', hidden_units=1000, output_labels=102, epochs=3, learning_rate=0.07, print_rate=20, gpu=True, checkpoint='checkpoint_cmd.pt'):
    """
    The module trains a new network on a dataset and save the model as a checkpoint
    Parameters: image_dataset, model architecture(resnes18,vgg16 or alexnet), hidden layer units, output labels, epochs, print_rate, device and the checkpoint
    
    Prints out training loss, validation loss, and validation accuracy as the network trains
    """
    
    """"Part 1: Loading all the required parameters"""
    
    #Assigns the parameters based on the command line args     
    if args.hidden_units:
        hidden_units = args.hidden_units

    if args.epochs:
        epochs = args.epochs
    else:
        epochs = epochs
        
    if args.print_rate:
        print_rate = args.print_rate
    else:
        print_rate
            
    if args.learning_rate:
        learning_rate = args.learning_rate
    else:
        learning_rate = learning_rate

    if args.gpu:
        gpu = args.gpu
    else: 
        gpu = gpu

    if args.checkpoint:
        checkpoint_data = args.checkpoint
    else:
        checkpoint_data = checkpoint 
    
    
    
    if args.data_dir:
        
        
        # TODO: Using the image datasets and the trainforms, define the dataloaders
        dataloaders = {idx: torch.utils.data.DataLoader(image_datasets[idx], batch_size=64, shuffle=True) for idx in list(image_datasets.keys())
              }
        
        
        """"Part 2: Training and building the network"""
        # TODO: Build and train your network

        #Step 1:Loading a pretrained network, i.e. vgg16
        #Inpired by the first project in this course and 
        if args.arch == 'vgg16' or arch_name=='vgg16':
            model = models.vgg16(pretrained=True)
        elif args.arch == 'resnest18' or arch_name=='resnet18':
            model = models.resnet18(pretrained=True)
        elif args.arch == 'alexnet' or arch_name=='alexnet':
            model = models.alexnet(pretrained=True)
        else:
            raise ValueError(arch_name, " is currently not supported try revisting python train() -h for help")

        
        #model,optimizer = load_model(arch_name,output_labels,hidden_units)
        model, classifier = load_model(arch_name,output_labels,hidden_units)
        
        
        #Step 3->Train the classifier layers using backpropagation using the pre-trained network to get the features

        #Sub-step 3.1: Track the loss and accuracy on the validation set to determine the best hyperparameters
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)

        #Sub-step 3.2-> Initializing other essential variable for training the model
        #epochs = epochs
        steps = 0 
        running_loss = 0
        accuracy = 0
        #print_rate = print_rate

        #sub-step 3.3-> checking and assigning device to train the model on
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda:0" if gpu else "cpu")
        model.to(device)
        
        #Initializing the start time counter and starting the training 

        start_time = time.time()

        print("Training started\n")
        
        for epo in range(epochs):
            model.train()
            
            for images, labels in iter(dataloaders['train']):
                images, labels = images.to(device), labels.to(device)
        
                steps +=1
        
                optimizer.zero_grad()
        
                #Feed-Forward and Backpropagation
                output = model.forward(images)
        
                loss = criterion(output, labels)
        
                loss.backward()
        
                #Tweak parameters
                optimizer.step()
        
                #Updating the running_loss through adding the loss 
                running_loss += loss.item()
        
                #Calculating the probability via converting the criterion to the softmax function
                ps = torch.exp(output).data
                equality = (labels.data == ps.max(1)[1])
        
                #accuracy = equality.type_as(torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor).mean()
                #accuracy = equality.type_as(torch.cuda.FloatTensor() if gpu else torch.FloatTensor).mean()
                accuracy = equality.type_as(torch.cuda.FloatTensor()).mean()
                
                if steps % print_rate == 0:
                    #Turn evaluation mode on or off
                    mode = model.eval()
            
                    #printing based on the training or validation mode
                    if mode == 0:
                        print("Epoch: {}/{}... ".format(epo+1, epochs),"\nTraining Loss: {:.3}".format(running_loss/print_rate))
                    else:
                        print("Validation Loss: {:.3f}  ".format(running_loss/print_rate),"Accuracy: {:.3f}".format(accuracy))
            
            
                    running_loss = 0
            
                    #Returning the model to training 
                    model.train()
        
        #Capture the total time taken to train the network 
        elapsed_time = time.time()-start_time

        print("\nFinished training the network in {:.0f}m {:.0f}s".format(elapsed_time/60,elapsed_time % 60))
        
        
        """Part 3: saving the trained model as a checkpoint_cmd.pth"""
        # TODO: Save the checkpoint 
        model.class_to_idx = image_datasets['train'].class_to_idx

        #Initializing checkpoint data/dict, inspired by Matt Leonard, najeebhassan, ksatola, and mpho mphego's works 
       
        checkpoint = {'input_size': 25088,
              'output_size': 102,
              'arch_name': 'vgg16',
              'learning_rate': 0.07,
              'batch_size': 64,
              'classifier' : classifier,
              'epochs': epochs,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}
        
        print("Started saving the checkpoint data")
        
        torch.save(checkpoint, checkpoint_data)
        
        print("Finished saving the checkpoint data")
        
        return model

#Calling the train() function based on the required parameters 
if args.data_dir:
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {'train' : transforms.Compose([transforms.RandomRotation(40),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                                         (0.229, 0.224, 0.225)
                                                                        ) 
                                                    
                                                    ]),
                   'valid' : transforms.Compose([transforms.Resize(256),
                                                      transforms.CenterCrop(224),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.485, 0.456, 0.406),
                                                                         (0.229, 0.224, 0.225)
                                                                        )
                                                    
                                                    ]),
                   'test' : transforms.Compose([transforms.Resize(256),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                                         (0.229, 0.224, 0.225)
                                                                        ) 
                                                    
                                                    ]),
                      }
    
    # TODO: Load the datasets with ImageFolder
    image_datasets = {
                idx: datasets.ImageFolder(args.data_dir + '/' + idx, transform=data_transforms[idx]) for idx in list(data_transforms.keys())
                 }
    #Calling the function train()
    train(image_datasets)
    
else:
    print("The data directory is required. Make sure you supply it via commnd line before proceeding. For help call python train.py -h")
        