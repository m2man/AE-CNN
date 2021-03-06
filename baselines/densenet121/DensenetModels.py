import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import torchvision


#RESNETS

class Resnet18(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(Resnet18, self).__init__()
        
        self.resnet18 = torchvision.models.resnet18(pretrained=isTrained)

        kernelCount = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet18(x)
        return x


class Resnet34(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(Resnet34, self).__init__()
        
        self.resnet34 = torchvision.models.resnet34(pretrained=isTrained)

        kernelCount = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet34(x)
        return x


class Resnet50(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(Resnet50, self).__init__()
        
        self.resnet50 = torchvision.models.resnet50(pretrained=isTrained)

        kernelCount = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet50(x)
        return x


class Resnet101(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(Resnet101, self).__init__()
        
        self.resnet101 = torchvision.models.resnet101(pretrained=isTrained)

        kernelCount = self.resnet101.fc.in_features
        self.resnet101.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet101(x)
        return x


class Resnet152(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(Resnet152, self).__init__()
        
        self.resnet152 = torchvision.models.resnet152(pretrained=isTrained)

        kernelCount = self.resnet152.fc.in_features
        self.resnet152.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet152(x)
        return x






#DENSENETS

class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(DenseNet121, self).__init__()
        
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)

        kernelCount = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x


class DenseNet161(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(DenseNet161, self).__init__()
        
        self.densenet161 = torchvision.models.densenet161(pretrained=isTrained)

        kernelCount = self.densenet161.classifier.in_features
        self.densenet161.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet161(x)
        return x


class DenseNet169(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(DenseNet169, self).__init__()
        
        self.densenet169 = torchvision.models.densenet169(pretrained=isTrained)

        kernelCount = self.densenet169.classifier.in_features
        self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet169(x)
        return x




class DenseNet201(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(DenseNet201, self).__init__()
        
        self.densenet201 = torchvision.models.densenet201(pretrained=isTrained)

        kernelCount = self.densenet201.classifier.in_features
        self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet201(x)
        return x






#INCEPTION

class InceptionV3(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(InceptionV3, self).__init__()
        
        self.inceptionv3 = torchvision.models.inception_v3(pretrained=isTrained)
        self.inceptionv3.transform_input = True

        kernelCount = self.inceptionv3.AuxLogits.fc.in_features
        self.inceptionv3.AuxLogits.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        
        kernelCount = self.inceptionv3.fc.in_features
        self.inceptionv3.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.inceptionv3(x)
        return x
