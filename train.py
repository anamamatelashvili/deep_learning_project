import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_directory', action="store", type=str, help='Path to directory with data.')
parser.add_argument('--save_dir', action="store", dest="save_directory", default='checkpoint_new.pth', type=str, help='Path for storing checkpoint.')
parser.add_argument('--arch', action="store", dest="input_architecture", default='vgg16', type=str, help='Model architecture. One of: vgg11, vgg13, vgg16 or vgg19.')
parser.add_argument('--learning_rate', action="store", dest="input_lr", default=0.0001, type=float, help='Learning rate. Positive double.')
parser.add_argument('--hidden_units', action="store", dest="input_hu", default=4096, type=int, help='Number of hudden units in the classifier. Positive integer.')
parser.add_argument('--epochs', action="store", dest="input_num_epochs", default=10, type=int, help='Number of epochs. Positive integer.')
parser.add_argument('--step_size', action="store", dest="input_step_size", default=5, type=int, help='Scheduler step size. Positive number.')
parser.add_argument('--gpu', action="store_true", default=False, dest='input_device', help='Use gpu.')
train_inputs = parser.parse_args()
if train_inputs.input_device == True:
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


def load_data(data_directory=train_inputs.data_directory):
    data_dir = data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {'train' : transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])]),
                       'valid' : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(), 
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])}

    # TODO: Load the datasets with ImageFolder
    image_datasets = {'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                      'valid' : datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=0, worker_init_fn=np.random.seed(2)),
                   'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=True, num_workers=0, worker_init_fn=np.random.seed(2))}

    dataset_sizes = {'train': len(image_datasets['train']),
                     'valid': len(image_datasets['valid'])} 
    
    return image_datasets, dataloaders, dataset_sizes



# Defaults for functions to follow: architecture and parameters of the best_model
architecture = train_inputs.input_architecture
lr = train_inputs.input_lr
hu = train_inputs.input_hu
step_size = train_inputs.input_step_size
num_epochs = train_inputs.input_num_epochs


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


def build_model(output_size, architecture=architecture, hu=hu):
    model = pick_arch(architecture)

    for param in model.parameters():
        param.requires_grad = False

    input_size = list(model.classifier.parameters())[0].size()[1]

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hu)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(p=0.5)),
                              ('fc2', nn.Linear(hu, output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    
    return model



def train_model(image_datasets, dataloaders, dataset_sizes, model=None, criterion=None, optimizer=None, scheduler=None,
                num_epochs=num_epochs, device=input_device):
    since = time.time()
    
    if model == None:
        output_size = len(image_datasets['train'].class_to_idx)
        model = build_model(output_size)
        
    if criterion == None:
        criterion = nn.NLLLoss()
        
    if optimizer == None:
        optimizer = optim.Adam(model.classifier.parameters(), lr)
        
    if scheduler == None:
        scheduler = lr_scheduler.StepLR(optimizer, step_size, gamma=0.1)    
    
    model = model.to(device)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_num_epochs = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_num_epochs = epoch + 1

    print('-' * 10)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best validation Accuracy of {:.4f} for {} epochs'.format(best_acc, best_num_epochs))
    print()

    # load best model weights
    model.load_state_dict(best_model_wts)
    model.to('cpu')
    return model, best_acc, best_num_epochs

   

if __name__ == "__main__":
    image_datasets, dataloaders, dataset_sizes = load_data()
    best_model, best_acc, best_num_epochs = train_model(image_datasets, dataloaders, dataset_sizes)

    # TODO: Save the checkpoint 
    checkpoint = {'state_dict': best_model.state_dict(),
                  'hu': hu,
                  'architecture': architecture,
                  'num_epochs': best_num_epochs,
                  'lr': lr,
                  'step_size': step_size,
                  'class_to_idx': image_datasets['train'].class_to_idx}

    torch.save(checkpoint, train_inputs.save_directory)




