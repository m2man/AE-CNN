import os
import numpy as np
import time
from time import time as now
import sys
import random
import copy
from sys import getsizeof

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets
import torch.nn.functional as func

from PIL import Image,ImageFile
from matplotlib import pyplot as plt
from skimage import io

from sklearn.metrics.ranking import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve as prc
from sklearn.metrics import average_precision_score as ap
from DensenetModels import AECNN, AECNN0, Resnet18
from DatasetGenerator_Binary import DatasetGenerator


class ChexnetTrainer ():

    # ----- function to compute AUROC, AUPRC, AP
    def computeAUROC (dataGT, dataPRED, classCount):
        
        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
        return outAUROC


    def shufflefinal(it, il, N):
        for loop in range(N//2):

            irand1 = np.random.randint(0, N)
            irand2 = np.random.randint(0, N)

            ivar = it[irand1].clone()
            it[irand1] = it[irand2]
            it[irand2] = ivar
            
            ivar = il[irand1].clone()
            il[irand1] = il[irand2]
            il[irand2] = ivar

        return it, il

    #takes 256x256 and returns 224x224 (cropping)
    def trans_train(x, transCrop):
        transformList = []
        transformList.append(transforms.ToPILImage())
        transformList.append(transforms.RandomCrop(transCrop) )
        # transformList.append(transforms.RandomRotation(5))
        transformList.append(transforms.RandomRotation(5))
        # transformList.append(transforms.ColorJitter(brightness=0, contrast=0.25))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformSequence=transforms.Compose(transformList)
        
        # y = torch.zeros( size=(x.shape), dtype = torch.float32)
        y = torch.zeros(size=(x.shape[0], x.shape[1], transCrop, transCrop), dtype = torch.float32)
        
        for i in range(x.shape[0]):
            y[i] = transformSequence(x[i])
        
        return y


    #takes 224x224/299x299 and returns 224x224/299x299 (simple testing)
    def trans_val(x):

        transformList = []
        transformList.append(transforms.ToPILImage())
        # transformList.append(transforms.Resize(transCrop))
        transformList.append(transforms.ToTensor())

        transformSequence=transforms.Compose(transformList)
        
        y = torch.zeros_like(x, dtype = torch.float32)
        
        for i in range(x.shape[0]):
            y[i] = transformSequence(x[i])
        
        return y

    def trans_test(x, transCrop):

        transformList = []
        transformList.append(transforms.ToPILImage())   
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
    
        transformSequence=transforms.Compose(transformList)
        
        y = torch.zeros( size=(x.shape[0], 10, x.shape[1], transCrop, transCrop), dtype = torch.float32)

        for i in range(x.shape[0]):
            y[i] = transformSequence(x[i])
        
        return y


    # -----ChexnetTrainer.store() - function to store the output of encoder of AE-CNN. This can be used for visualisation of latent code produced by encoder
    def store(file_path, pathImg, pathModel, nnClassCount, device, savename = 'latent_encoder', basewidth=1024):
        # pathImg is directory to parent datafolder
        # file_path is txt file

        model = AECNN0(nnClassCount)
        model.to(device)

        #--LOAD MODEL
        if pathModel != None:
            modelCheckpoint = torch.load(pathModel, map_location=lambda storage, loc: storage)
            model.load_state_dict(modelCheckpoint['state_dict'])
            print("LOADED FROM PATHMODEL:", pathModel)
        else:
            print("PATHMODEL could not be loaded:", pathModel)

        list_images_name = []
        file_ptr = open(file_path,'r')
        for line in file_ptr:
            img_name = line.split(" ")[0]
            img_name = pathImg + img_name if pathImg[-1] == '/' else pathImg + '/' + img_name
            list_images_name.append(img_name)
        file_ptr.close()
        numb_images = len(list_images_name)

        output_encoding = torch.ByteTensor(numb_images, 3, 256, 256)

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        #basewidth = 896 -> 224 --> can keep 1024 as original -> 256

        for idx, image_name in enumerate(list_images_name):
            img = Image.open(image_name)
            img = img.resize((basewidth, basewidth), Image.ANTIALIAS)
            img=img.convert('L')

            np_image=np.array(img)
            image_tensor=torch.from_numpy(np_image)

            input = image_tensor.float()
            input/=255
            input = input.view(1, 1, input.shape[0], input.shape[1])

            varInput = torch.autograd.Variable(input)
            varInput = varInput.to(device)


            varOutput = model.encoder(varInput)
            #print(varOutput.shape)
            output_img = (varOutput*255).byte()
            # print(type(output_img))
            # print(output_img.dtype)

            output_encoding[idx] = output_img
            

        print("No of images encoded:", numb_images)

        # torch.save(output_encoding,"/home/deepsip/ChestX/database/val_only_224_autoencoder.pth")
        torch.save(output_encoding, f"{savename}.pth")


    def train (pathDirData, pathFileTrain, pathFileVal, nnClassCount, trBatchSize, trMaxEpoch, launchTimestamp, checkpoint=None):
        nnArchitecture = 'DENSENET121'
        print("Inside train function:")
    
        #-------------------- SETTINGS: NETWORK ARCHITECTURE
        model  = AECNN0(nnClassCount)
        model.cuda()

        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.2, patience = 2, mode = 'min', verbose=True)
                
        loss1 = torch.nn.MSELoss(size_average = True)
        loss2 = torch.nn.BCELoss(size_average = True)

        #---- Load checkpoint 
        #---checkpoint1 is for all
        epochStart = 0
        

        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
            epochStart = modelCheckpoint['epoch']
            print("LOADED FROM CHECKPOINT:", checkpoint)

        #-------------------- SETTINGS: DATA TRANSFORMS
        # normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        transformList = []
        transformList.append(transforms.RandomCrop(1024))
        transformList.append(transforms.RandomRotation(5))
        transformList.append(transforms.ToTensor())
        transformSequence=transforms.Compose(transformList)



        #-------------------- SETTINGS: DATASET BUILDERS
        datasetTrain = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain, transform=transformSequence)
        datasetVal =   DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileVal, transform=transformSequence)
              
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=24, pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=24,  pin_memory=True)

        #---- TRAIN THE NETWORK
        
        lossMIN = 100000    
        auc_per_5_epochs = []
        auc_per_epoch = []
        loss_train_per_epoch = []
        loss_val_per_epoch = []

        print('\n')
        
        info = f"===== {launchTimestamp} =====\n"
        with open (f"{launchTimestamp}_Log.log", 'a') as f:
                f.write(info)

        for epochID in range (epochStart, trMaxEpoch):
            
        #----------------------------------------------------------------training             
            tic = now()
            lossTrain = ChexnetTrainer.epochTrain(model, dataLoaderTrain, optimizer, loss1, loss2) 
            
            loss_train_per_epoch.append(lossTrain)
            
            torch.save(loss_train_per_epoch, './loss/' + nnArchitecture + '-'+str(trMaxEpoch)+'-'+'loss_train_mean-'+str(launchTimestamp) )
            # print("Saved Loss for this epoch")
            
            toc = now()


            info = f"Epoch: {epochID+1} - Train Loss: {lossTrain}\n"
            with open (f"{launchTimestamp}_Log.log", 'a') as f:
                f.write(info)

            print(f"Time taken epoch {epochID+1}: {toc-tic}")

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
            
            if ((epochID + 1)%5 == 0):

                print ('\nValidating the trained model ')
                aurocMean, lossVal, losstensor = ChexnetTrainer.epochVal(model, dataLoaderVal, loss1, loss2)

                auc_per_5_epochs.append(aurocMean)
                loss_val_per_epoch.append(lossVal)
                torch.save(auc_per_5_epochs, './AUC/' + nnArchitecture +'-'+str(trMaxEpoch)+'-'+'auc_mean-'+str(launchTimestamp))
                torch.save(loss_val_per_epoch, './loss/' + nnArchitecture+'-'+str(trMaxEpoch)+'-'+'loss_val_mean-'+str(launchTimestamp))
                print("AUC saved\nSaving Model")

                #---STEP SCHEDULER
                scheduler.step(losstensor.item())
        
                if lossVal< lossMIN:
                    lossMIN = lossVal
                    torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'm-' + launchTimestamp + '.pth.tar')
                    print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
                    info = f"[Evaluate] Epoch: {epochID+1} - [SAVE] - Val Loss: {lossVal} - Val AUROC: {aurocMean}\n"
                else:
                    print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))
                    info = f"[Evaluate] Epoch: {epochID+1} - [---] - Val Loss: {lossVal} - Val AUROC: {aurocMean}\n"
                
                with open(f"{launchTimestamp}_Log.log", 'a') as f:
                    f.write(info)


    def epochTrain(model, dataLoader, optimizer, loss1, loss2):
        
        model.train()

        lossTrain = 0
        lossTrainNorm = 0


        for batchID, (input, target) in enumerate (dataLoader):
            # train for batch

            target = target.cuda(async = True)
                 
            varInput = torch.autograd.Variable(input).cuda()
            shape = list(varInput.shape)
            varTarget1 = input[:,0,:,:].view(shape[0], 1, shape[2], shape[3]).cuda()
            #print(list(varTarget1.shape))
            varTarget2 = torch.autograd.Variable(target).cuda()

            # varInput dim: bsx1x896x896 (or maybe 1014x1024)
            varOutput1, varOutput2 = model(varInput)
            #rint(list(varOutput1.shape))
            lossvalue1 = loss1(varOutput1, varTarget1)
            lossvalue2 = loss2(varOutput2, varTarget2)
            lossvalue = 0.1*lossvalue1 + 0.9*lossvalue2 # weighting bw MSE and BCE resp.

            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()

            losstensor = lossvalue
            lossTrain += losstensor.item()
            lossTrainNorm += 1

        lossTrainMean = lossTrain/lossTrainNorm
        return lossTrainMean


       
    def epochVal (model, dataLoader, loss1, loss2):
        
        model.eval ()
        
        lossVal = 0
        lossValNorm = 0
        losstensorMean = 0

        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()

        with torch.no_grad():
            for i, (input, target) in enumerate (dataLoader):
                
                target = target.cuda(async = True)
                outGT = torch.cat((outGT, target), 0)

                varInput = torch.autograd.Variable(input).cuda()
                shape = list(varInput.shape)
                varTarget1 = input[:,0,:,:].view(shape[0], 1, shape[2], shape[3]).cuda()
                varTarget2 = torch.autograd.Variable(target).cuda()
                
                varOutput1, varOutput2 = model(varInput)

                outPRED = torch.cat((outPRED, varOutput2), 0)

                lossvalue1 = loss1(varOutput1, varTarget1)
                lossvalue2 = loss2(varOutput2, varTarget2)
                lossvalue = 0.1*lossvalue1 + 0.9*lossvalue2 # weighting bw MSE and BCE resp.

                losstensorMean += lossvalue

                lossVal += lossvalue.item()
                lossValNorm += 1
            
            outGTnp = outGT.cpu().numpy()
            outPREDnp = outPRED.cpu().numpy()

            aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, 1)
            aurocMean = np.array(aurocIndividual).mean()
                
            outLoss = lossVal / lossValNorm
            losstensorMean = losstensorMean / lossValNorm
        
        return aurocMean, outLoss, losstensorMean