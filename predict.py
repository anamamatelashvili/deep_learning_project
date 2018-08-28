import argparse
parser = argparse.ArgumentParser()
parser.add_argument('image', action="store", type=str, help='Path to image file.')
parser.add_argument('checkpoint', action="store", type=str, help='Path to checkpoint file.')
parser.add_argument('--top_k', action="store", dest="input_top_k", default=1, type=int, help='Number of most probable classes to predict. Positive integer.')
parser.add_argument('--category_names', action="store", dest="input_cat_names", default='cat_to_name.json', type=str, help='Path to a json file with category names.')
parser.add_argument('--gpu', action="store_true", default=False, dest='input_device', help='Use gpu.')
predict_inputs = parser.parse_args()
if predict_inputs.input_device == True:
    input_device = 'cuda'
else:
    input_device = 'cpu'


# Imports here
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import copy
import time
import json

from collections import OrderedDict

from PIL import Image
import random

random.seed(2)
np.random.seed(2)
torch.manual_seed(2)
torch.cuda.manual_seed_all(2)
torch.backends.cudnn.deterministic=True


def pick_arch(architecture):
    """Pick from the following model architectures: vgg11, vgg13, vgg16, vgg19"""
    if architecture == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif architecture == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif architecture == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        model = None
        print('Wrong model architecture. Pick vgg11, vgg13, vgg16 or vgg19.')
    
    return model

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    model = pick_arch(architecture=checkpoint['architecture'])

    for param in model.parameters():
        param.requires_grad = False

    input_size = list(model.classifier.parameters())[0].size()[1]
    output_size = len(checkpoint['class_to_idx'])

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, checkpoint['hu'])),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(p=0.5)),
                              ('fc2', nn.Linear(checkpoint['hu'], output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    width, height = pil_image.size
    l = int(256 * max(width, height) / min(width, height))
    pil_image.thumbnail(size = (l,l))
    width, height = pil_image.size 
    pil_image = pil_image.crop(box = (width / 2 - 112, height / 2 - 112, width / 2 + 112, height / 2 + 112))
    
    np_image = np.array(pil_image)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image / 255.0 - mean) / std
    
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image


def predict2(image_path, checkpoint, topk, cat_names, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model = load_checkpoint(checkpoint)
    model.to(device)
    model.eval()
    image = process_image(image_path)
    img_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    img_tensor.unsqueeze_(0)
    image_variable = Variable(img_tensor)
    image_variable = image_variable.to(device)
    with torch.no_grad():
        outputs = model(image_variable)  
          
    probs, classes = outputs.topk(topk) 
    
    probs = list(np.exp(np.asarray(probs).flatten()))
    classes = list(np.asarray(classes).flatten())
    
    class_to_idx = model.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    classes = [idx_to_class[k] for k in classes]
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)
    classes = [cat_to_name[k] for k in classes]
    model.to('cpu')
    return probs, classes


    
        
  
    



if __name__ == "__main__":
    probs, classes = predict2(image_path=predict_inputs.image, checkpoint=predict_inputs.checkpoint, topk=predict_inputs.input_top_k, cat_names=predict_inputs.input_cat_names, device=input_device)
    
    if predict_inputs.input_top_k > 0:
        print('The predictions are:')
        for i in range(len(probs)):
            print('Class: {} with probability: {:.2%}'.format(classes[i], probs[i]))
    else:
        print('Enter a positive integer for --top_k')
    









