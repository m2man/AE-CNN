import os
import numpy as np
import time
import sys
import torch

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from ChexnetTrainer_Binary import ChexnetTrainer



#-------------------------------------------------------------------------------- 

def main ():
    
    runTrain()
    #runTest()
  
#--------------------------------------------------------------------------------   

def runTrain():
    
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    #---- Path to the directory with images
    
    #---- Paths to the files with training, validation and testing sets.
    #---- Each file should contains pairs [path to image, output vector]
    #---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    datatxt = '/home/dxtien/dxtien_research/nmduy/chexnet/dataset/'
    pathFileTrain = datatxt + 'binary_train.txt'
    pathFileVal = datatxt + 'binary_validate.txt'
    pathFileTest = datatxt + 'binary_test.txt'
    pathDirData = '/home/dxtien/dxtien_research/COVID/CXR8/'

    #---- Neural network parameters: type of the network, is it pre-trained 
    #---- on imagenet, number of classes
    nnIsTrained = True
    nnClassCount = 1
    
    #---- Training settings: batch size, maximum number of epochs
    trBatchSize = 16
    trMaxEpoch = 50
    
    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    transResize = 224
    transCrop = 1024 #896
    

    # ----- pathModel - path to the AE-CNN model which is to be tested 
    # -----checkpoint1 - is like a dummy. If not none then it loads encoder and decoder from encoder.pth and decoder.pth seperately into AE-CNN encoder and decoder
    # -----checkpoint2 - if not none, loads the classifier of AE-CNN from pretrained classifier on ChestX-Ray14 dataset
    # -----checkpoint3 - if not none, loads the full AE-CNN weights for resuming the training from a saved instance
    #pathModel = 'm-Autoencoder-pE-pD-pd121-epoch-15-auc-0.843020353288727-16072018-233246.pth.tar'
    pathModel = 'DENSENET121-' + timestampLaunch + '.pth.tar'
    #checkpoint1 = 'm-Autoencoder-epoch-17-loss-0.00026260194169445625-15072018-133705.pth.tar'
    checkpoint1 = None
    #checkpoint2 = 'm-DENSE-NET-121-epoch-6-auc-0.8391076085811494-15072018-184230.pth.tar'
    checkpoint2 = None
    #checkpoint3 = 'm-Autoencoder-pE-pD-pd121-epoch-14-auc-0.8429981074471875-16072018-233246.pth.tar'
    checkpoint3 = None

    print ('Training DENSENET121')
    ChexnetTrainer.train(pathDirData, pathFileTrain, pathFileVal, nnClassCount, trBatchSize, trMaxEpoch, timestampLaunch)

    # -----ChexnetTrainer.store() - function to store the output of encoder of AE-CNN. This can be used for visualisation of latent code produced by encoder
    # ChexnetTrainer.store(parentPath, pathModel, nnArchitecture, nnClassCount, trBatchSize, device)

    # -----uncomment the below block and comment out the training block for testin the AE-CNN from pathModel.
    # print ('Testing the trained model')
    # ChexnetTrainer.test(parentPath, None, pathModel, pathDirDataTest, pathFileTest, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, timestampLaunch, threshold, device)

#-------------------------------------------------------------------------------- 


if __name__ == '__main__':
    main()

