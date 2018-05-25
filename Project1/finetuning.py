import torch
import torchvision

import sklearn.svm

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import numpy as np
import matplotlib.pyplot as plt

from util import plot_confusion_matrix
from torchvision import datasets, models, transforms

import torchvision.models
model = torchvision.models.alexnet(pretrained=True)

resize = transforms.Resize((224, 224))
transform = transforms.Compose([resize, transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
batch_size=4

#num_classes=extract from folder

new_layer=torch.nn.Linear(4096, 31)
print('newlayer',new_layer)
print('initial model', model)

old_list=list(model.classifier.children())[:-1]
print('old list', old_list)
new_list=old_list.append(new_layer)
print('new list', new_list)


new_classifier = nn.Sequential(*new_list)
model.classifier=new_classifier

print('new model', model)
