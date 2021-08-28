from __future__ import print_function
from __future__ import division

import warnings
warnings.filterwarnings("ignore")

import os
import csv
import random
import shutil
import time
import json
import copy
import pickle
import argparse

import torch, torchvision
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm_base

# from timm.utils import *
from timm.optim import AdamP
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler

from and_mask_utils import get_grads
from adam_flexible_weight_decay import AdamFlexibleWeightDecay
matplotlib.style.use('seaborn')

from torchmetrics.functional import accuracy, confusion_matrix, f1

parser = argparse.ArgumentParser(description="Exploring Different Models Performance on Medical Images Pathology Classification")

### misce
parser.add_argument('--seed', type=int, default=1, help='seed for initializing training for reproducibility')
parser.add_argument('--data_dir', type=str, default="./malaria_dataset/", required=True, help='directory of dataset')
parser.add_argument('--output_dir', type=str, default="./malaria_diagnostics", required=True, help='directory to save checpoints into')
parser.add_argument('--start_epoch', type=int, default=None, help='manual epoch number (useful on restarts)')
parser.add_argument('--arch', nargs="+", default=None, help='architecture of model')

### training
parser.add_argument('--epochs', type=int, default=190, help='number of epochs to run')
parser.add_argument('--cuda', type=bool, default=True, help='use cuda or train on gpus')
parser.add_argument('--gpu', type=int, default=None, help='GPU id to use.')

### dataloading
parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle data')
parser.add_argument('--num_workers', type=int, default=8, help='number of threads')
parser.add_argument('--num_classes', type=int, default=2, help='number of dataset classes')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--input_size', type=int, default=224, help='Input image resolution')

### regularize
parser.add_argument('--feature_extract', action='store_true', default=False, help='whether to use model as feature extractor')
parser.add_argument('--pretrained', action='store_true', default=False, help='whether to use ImageNet Weights')

### ilc
parser.add_argument('--ilc', action='store_true', default=False, help='Train with ILC')
parser.add_argument('--scale_grad_inverse_sparsity', type=int, default=1, help='')
parser.add_argument('--agreement_threshold', type=float, default=0.2, help='')
parser.add_argument('--method', type=str, default='and_mask', help='')
                    
args = parser.parse_args()
print(f'args: {args}')

if args.ilc:
    output_dir = args.output_dir + "_ilc_models_checkpoint"
    models_test =  args.output_dir + "_ilc_models_test.pkl"
else:
    output_dir = args.output_dir + "_baseline_models_checkpoint"
    models_test =  args.output_dir + "_baseline_models_test.pkl"
    
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f'Output dir: {output_dir}')
        
        
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)

device = 'cuda' if args.cuda else 'cpu'
if not torch.cuda.is_available() and args.cuda:
    device = 'cpu'
    print("WARNING: cuda was requested but is not available, using cpu instead.")
    
print(f'Using device: {device}')

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
        
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
### Model Store ###
def get_model_inputimg(model_name, pretrained, num_classes):
    """
    Function to initialize model and input image size. Available models are: 
    ['resnet18', 'resnet34', 'mobilenet_v2', 'mnasnet0_5', 'mnasnet1_0', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 
    googlenet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121']
    """
    model = None
    input_size = 0
    
    if args.feature_extract:
        print(f"\n\n=> Using pre-trained model as feature extractor")
    elif not args.feature_extract:
        print(f"\n\n=> Using pre-trained model {model_name} for finetuning")
    else:
        print(f"\n\n=> Creating model {model_name}")
    
    if "resnet" in model_name:
        model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        set_parameter_requires_grad(model, args.feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)
                )
        
    elif "shufflenet" in model_name:
        model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        set_parameter_requires_grad(model, args.feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)
                )
        
    elif "googlenet" in model_name:
        model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        set_parameter_requires_grad(model, args.feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)
                )

    elif "mobilenet" in model_name:
        model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        set_parameter_requires_grad(model, args.feature_extract)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)
                )
        
    elif "mnasnet" in model_name:
        model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        set_parameter_requires_grad(model, args.feature_extract)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)
                )

    elif "squeezenet" in model_name:
        model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        set_parameter_requires_grad(model, args.feature_extract)
        model.classifier[1] = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
                )

    elif "densenet" in model_name:
        model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        set_parameter_requires_grad(model, args.feature_extract)
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)
                )
    else:
        print(f"Invalid model name. Choose one of the models defined for this work")
        exit()
    
    return model

#############################################################################################################################
def ilc_loss(outputs, labels, criterion, optimizer):
    if args.agreement_threshold > 0.0:
        mean_loss, masks = get_grads(
            agreement_threshold=args.agreement_threshold,
            batch_size=1,
            loss_fn=criterion,
            n_agreement_envs=args.batch_size,
            params=optimizer.param_groups[0]['params'],
            output=outputs,
            target=labels,
            method=args.method,
            scale_grad_inverse_sparsity=args.scale_grad_inverse_sparsity,
            )
    else:
        mean_loss = criterion(outputs, labels)
    
    return mean_loss

def save_checkpoint(state, is_best, ckpt_filename, best_model_ckpt):
    torch.save(state, ckpt_filename)
    if is_best:
        shutil.copyfile(ckpt_filename, best_model_ckpt)
        
def train_model(model, model_name, dataloaders, criterion, scheduler, optimizer, num_epochs):
    best_acc = 0.0
    
    resume = output_dir + '/' + model_name + '_checkpoint.pth'
    start_epoch = 0
    resume_epoch = None
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        if args.gpu is None:
            checkpoint = torch.load(resume)
        else:
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(resume, map_location=loc)
        resume_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_valid_acc']
        if args.gpu is not None:
            best_acc = best_acc.to(args.gpu)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))      
    else:
        print("=> no checkpoint found at '{}'".format(resume))
        
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
           
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_optim_wts = copy.deepcopy(optimizer.state_dict())
    
    for epoch in range(start_epoch, num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0
            
            progress = tqdm(dataloaders[phase])
            for idx, (inputs, labels) in enumerate(progress):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        outputs = model(inputs)
                        if args.ilc:
                            loss = ilc_loss(outputs, labels, criterion, optimizer)
                        else:
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        optimizer.step()                        

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
                progress.set_description(f'Phase: {phase.capitalize()} - Epoch: [{epoch}/{num_epochs-1}] - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                print(f'Validation accuracy improved. Saving model and optimizer state.')
                is_best = True
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_optim_wts = copy.deepcopy(optimizer.state_dict())
            else:
                is_best = False
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        scheduler.step()
        print()
        
        ckpt_filename =  output_dir + '/' + model_name + '_checkpoint.pth'
        best_model_ckpt =  output_dir + '/' + model_name + '_model_best.pth'
        
        save_checkpoint({
                'epoch': epoch + 1,
                'arch': model_name,
                'state_dict': model.state_dict(),
                'best_valid_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, ckpt_filename, best_model_ckpt)
        
    print(f'{model_name} best validation Acc {best_acc:4.4f}')

#     model.load_state_dict(best_model_wts)
#     optimizer.load_state_dict(best_optim_wts)
    
    return model, val_acc_history, best_acc
       
def inference(model_name, model, classes, test_loader):
    correct = 0
    total = 0
    test_prediction = {}
    
    model.eval()
    
    with torch.no_grad():
        progress = tqdm(test_loader)
        for idx, data in enumerate(progress):
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of {model_name} on the {len(test_dataset.samples)} test images: {100 * correct / total:.2f}%')

    test_prediction['Prediction'] = round(100 * correct / total, 2)

    ######### Compute prediction for each class in the dataset ###########
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    class_predictions = {}
    for class_name, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[class_name]
        print(f"Accuracy for class {class_name:5s} is: {accuracy:.2f}%")
        class_predictions[class_name] = round(accuracy, 2)
    
    print()
    
    return test_prediction, class_predictions
#####################################################           #######################################################
if args.arch is not None:
    model_zoo = args.arch
else:
    model_zoo = sorted(['resnet50', 'mobilenet_v2', 'mnasnet1_0', 'shufflenet_v2_x1_0', 'googlenet', 'squeezenet1_1', 'densenet121'])

results_dict = {}
for model_name in model_zoo:
    ######### Model Initialization ############
    model = get_model_inputimg(model_name=model_name, pretrained=args.pretrained, num_classes=args.num_classes) 
    model = model.to(device)

    ####################### Data Transformation and Loading ########################
    data_transforms = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(args.input_size),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], 
                                             [0.229, 0.224, 0.225])
        ]),
        'val': torchvision.transforms.Compose([
            torchvision.transforms.Resize((args.input_size, args.input_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], 
                                             [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(args.data_dir, x),
                            data_transforms[x]) for x in ['train', 'val',]}
    
    dataloaders_dict = {x: torch.utils.data.DataLoader(
                                      image_datasets[x], 
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      pin_memory=True,
                                      num_workers=args.num_workers,) for x in ['train', 'val']}
    
    test_dataset = torchvision.datasets.ImageFolder(
                              root=os.path.join(args.data_dir, 'test'),
                              transform=data_transforms['val']
                              )
    
    classes = test_dataset.classes
    test_dataloader = torch.utils.data.DataLoader(
                            test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers
                            )
    
    dataloaders_dict['val'].shuffle = False
    
    ######### Optimizer ###############
    params_to_update = model.parameters()
    print("Params to learn:")
    if args.feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        print(f"Training entire model")               
    
    if args.ilc:
        optimizer = AdamFlexibleWeightDecay(params_to_update, lr=1e-3, weight_decay=1e-5, weight_decay_order='before')
    else:
        optimizer = AdamP(params_to_update, lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    ############ Loss Function ###########
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    ########## Train and validate ########
    model, _, _ = train_model(model, model_name, dataloaders_dict, criterion, scheduler, optimizer, args.epochs)
        
    ########## Inference ########
    checkpoint_path =  output_dir + '/' + model_name + '_model_best.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    test_prediction, class_predictions = inference(model_name, model, classes, test_dataloader)
    
    results_dict[model_name] = [test_prediction, class_predictions]
    

### pickle models and inference accuracies        
models_test_acc = open(models_test, "wb")
pickle.dump(results_dict, models_test_acc)
models_test_acc.close()

