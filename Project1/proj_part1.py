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
'''
def cifar100(seed):
    np.random.seed(seed)
    resize = transforms.Resize((224, 224))
    transform = transforms.Compose(
        [resize, transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    trainset = torchvision.datasets.ImageFolder(root='./faces/train',
                                             transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageFolder(root='./faces/test',
                                            transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader
'''

new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier= new_classifier

print('new_classifier', new_classifier)
#change forward if feature extractor is not being used by the nn
'''
def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), 256 * 6 * 6)
    #x = self.classifier(x)
    return x
'''
trainset = torchvision.datasets.ImageFolder(root='../faces/train',
                                             transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(root='../faces/test',
                                            transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

#loader = torch.utils.data.DataLoader(datasets.ImageFolder(data_dir, preprocessor),batch_size=batch_size,shuffle=True)

target_numpy = None
in_data_numpy = None

for i,(in_data, target) in enumerate(trainloader):
    print("i: ", i)
    input_var = torch.autograd.Variable(in_data,volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)

    output = model(input_var)

    target_new = target_var.data.numpy()
    in_data_new = output.data.numpy()

    if target_numpy is None:
        target_numpy=target_new
    else:
        target_numpy = np.append(target_numpy, target_new,axis=0)
    if in_data_numpy is None:
        in_data_numpy = in_data_new
    else:
        in_data_numpy = np.append(in_data_numpy, in_data_new,axis=0)
print('shapetargetnumpy', target_numpy.shape)
print('shapeindatanumpy', in_data_numpy.shape)

test_target_numpy = None
out_data_numpy = None

for i,(test_data, test_target) in enumerate(testloader):
    print("i: ", i)
    output_var = torch.autograd.Variable(test_data,volatile=True)
    test_target_var = torch.autograd.Variable(test_target, volatile=True)

    output = model(output_var)

    test_target_new = test_target_var.data.numpy()
    out_data_new = output.data.numpy()

    if test_target_numpy is None:
        test_target_numpy=test_target_new
    else:
        test_target_numpy = np.append(test_target_numpy, test_target_new,axis=0)
    if out_data_numpy is None:
        out_data_numpy = out_data_new
    else:
        out_data_numpy = np.append(out_data_numpy, out_data_new,axis=0)
    # print('numpindatatype', in_data_numpy.dtype)
    # print('targetshape', target.shape)
    # print('targetnumpydtype', target_numpy.dtype)
    # break

#print('targetnumpyshape', target_numpy.shape)
#print('indatashape', in_data.shape)
#print('indatatype', in_data.dtype)


'''
    print('index=', i)
    if (target_numpy ==None):
        target_numpy_ele = target[i].numpy()
    else:
        target_numpy_ele = target[i].numpy()
    np.append(target_numpy, target_numpy_ele, axis=0)
    print('target_numpy', target_numpy)
    print('target_numpy.shape', target_numpy.shape)
'''
'''
    else:
        target_numpy = target.numpy()
    np.append(target_numpy, target_numpy,axis=0)
    if (in_data_numpy ==None):
        in_data_numpy = in_data.numpy()
    else:
        in_data_numpy = in_data.numpy()
    in_data_numpy.append(in_data.numpy,axis=0)
'''
print target_numpy.shape


x_train, y_train, x_test, y_test = in_data_numpy, target_numpy, out_data_numpy, test_target_numpy
model = sklearn.svm.SVC(C=4.0, kernel='linear')
model.fit(x_train, y_train)
y_pred_lin = model.predict(x_test)

from sklearn.metrics import accuracy_score
a_svm = accuracy_score(y_test, y_pred_lin)
print('Accuracy_linear: %.2f' % a_svm)

new_layer=