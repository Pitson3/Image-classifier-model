
import matplotlib.pyplot as plt 

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
import json

#from train import train 
import numpy as np
from PIL import Image 
import os, random
from torch.autograd import Variable
import argparse

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('image', type=str, help='The image to be prticted')
parser.add_argument('--checkpoint', type=str, help='The saved model checkpoint to be used for prediction')
parser.add_argument('--topk', type=int, help='Return top K predictions')
parser.add_argument('--labels', type=str, help='JSON file containing with label names')
parser.add_argument('--gpu', action='store_true', help='Allow the usage of the gpu device if available')

#Convert argument strings to objects and assign them as attributes of the namespaces then return the populated namespace without errors on additional args
args, _ = parser.parse_known_args()

#Define a prediction function for image based on the pre-trained model
def predict(image_path, checkpoint='checkpoint_cmd.pt', topk=5, gpu=False , cat_to_name='cat_to_name.json'):
    """
    The module predicts the class of an image using a pretrained neural network
    
    Inputs Parameters: Image, saved checkpoint, topk(probaility) values, gpu status and category labels for the images
    
    Output: Prints the inputed image and the probailities of top 5 images inclusive of the input image
    """
    
    """"Part 1: Loading all the required parameters"""
    
    #Assigns the parameters based on the command line args     
    if args.topk:
        topk = args.topk
    else:
        topk = topk
        
    if args.labels:
        cat_to_name = args.labels
    else:
        cat_to_name = cat_to_name
            
    if args.checkpoint:
        checkpoint_data = args.checkpoint
    else:
        checkpoint_data = checkpoint

    if args.gpu:
        gpu = args.gpu
    else: 
        gpu = gpu
        
    #Load image classes 
    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
 
    if args.image:
        image_path = args.image
        
        
        #Rebuilding the model from a saved checkpoint
        model = rebuild_model_from_checkpoint(checkpoint_data)
        
        #model, optimizer = rebuild_model_from_checkpoint(checkpoint_data)


        # TODO: Implement the code to predict the class from an image file

        #Switch to evaluation mode and choose the device to run the program on
        model.eval()


        device = torch.device("cuda:0" if gpu else "cpu")
        model.to(device)

        #Load an image path and process it 
        image = Image.open(image_path)

        image = process_image(image)

        #print(image)

        #Convert a numpy array to a tensor
        #if torch.cuda.is_available():
        #    image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
        #else:
        #    image = torch.from_numpy(image).type(torch.FloatTensor).cpu()

        image = torch.from_numpy(image).type(torch.cuda.FloatTensor() if gpu else torch.FloatTensor)
        #image = torch.from_numpy(image).type(torch.cuda.FloatTensor())
        #Formating tensor for input into  model and choose device 
        image = image.unsqueeze(0)
        #image.to(device)

        #Inputing the tensor via feed-forwarding and returning the output
        #Overlooked torch.auto_grad.Variable due to deprecation issues 
        output = model.forward(image)


        #Capture the probability based on our loss function 
        prob = torch.exp(output).data

        #Return topk probalities and indices
        probs, indices = torch.topk(prob,topk)



        #MOdified version of https://ksatola.github.io/projects/image_classifier_with_deep_learning.html
        #Convert from Tensors to Numpy arrays
        top_probabilities = probs.cpu().detach().numpy().tolist()[0]

        #print("model.class_to_idx ", model.class_to_idx)
        #Altering classes to indeces
        indexed_classes = {model.class_to_idx[i]: i for i in model.class_to_idx}

        #print("Indexed classes ", indexed_classes)
        #print("tensor! ",indices) 

        #transfer indices to labels 

        #print("Concersion ", indices.cpu().numpy().tolist()[0])
        labels = [] 
        for index in indices.cpu().numpy().tolist()[0]:
            labels.append(indexed_classes[index])
        #print("Labels b4 ", labels)

        #return top_probabilities, labels 
        
        #Checking images for sanity 
        #check_images_for_sanity(image_path,model,cat_to_name)
        
        #Load image from path 
        #image = Image.open(image_path)
    
        #Get the image name based on path
        #flower_num = image_path.split('/')[-2]
    
        #Extract probabilities and labels through the predict function  
        #probs, classes = predict(image_path, model)
    
        #Return the indices of the maximum values
        max_index = np.argmax(top_probabilities)
    
        #Get maximum probability
        max_probability = top_probabilities[max_index]
    
        #Return the folder/flower integer nun
        flower_num = labels[max_index]
    
        print(top_probabilities)
        #print(labels)
        #print(flower_num)
        #print(cat_to_name)
        #print(cat_to_name[flower_num])
        
        return top_probabilities, labels
        
    else:
        print("The image path is required!!")
        
    
# TODO: Write a function that loads a checkpoint and rebuilds the model
def rebuild_model_from_checkpoint(file_name):
    checkpoint = torch.load(file_name)
    learning_rate = checkpoint['learning_rate']

    #Consolidating the data into a model 
    model = getattr(models, checkpoint['arch_name'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    #Consolidating the optimizer data 
    #optimizer = model.load_state_dict(checkpoint['optimizer'])

    #return model, optimizer 
    return model

#Process the image 
def process_image(image):
    ''' 
        Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a transposed Numpy array
        '''

    # TODO: Process a PIL image for use in a PyTorch model
    #image = Image.open(image)
    #image = image.thumbnail(256)->Kept for debugging as it is not subscribtable

    #Alter the image size 
    image = image.resize((256,256))

    #print(image) 

    #Crop the image with proper parameters 
    left_margin = 0.5*(image.width-224)
    bottom_margin = 0.5*(image.width-224)
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224 

    image = image.crop((left_margin, bottom_margin, right_margin, top_margin))


    #print(image)

    image = np.array(image)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = (image - mean) / std

    return image.transpose(2,0,1)


#Calling the predict function 
if args.image:
     predict(args.image)
else:
    raise ValueError("The image path must be passed in the arguments")