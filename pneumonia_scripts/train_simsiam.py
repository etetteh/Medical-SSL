from __future__ import print_function
from __future__ import division

import warnings
warnings.filterwarnings("ignore")

import os
import torch
import random
import pickle
import argparse

import timm
import torch
import lightly
import torchvision

import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from glob import glob
from PIL import Image
from pathlib import Path
from timm.optim import AdamP
from os.path import join, exists
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate
from torchmetrics.functional import accuracy, confusion_matrix, f1
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging

parser = argparse.ArgumentParser(description="Self-Supervised Learning with SIMSIAM for Medical Imaging")

parser.add_argument('--seed', type=int, default=0, help='seed for initializing training for reproducibility')
parser.add_argument('--data_dir', type=str, default=None, required=True, help='directory of dataset')
parser.add_argument('--ckpt_simsiam', type=str, default='simsiam_checkpoints', required=True, help='directory of checkpoints')
parser.add_argument('--ckpt_classifier', type=str, default='simsiam_feat', required=True, help='directory of checkpoints')
parser.add_argument('--arch', nargs="+", default=None, help='Architecture of model')

parser.add_argument('--batch_size', type=int, default=64, help='Batch size of samples')
parser.add_argument('--num_workers', type=int, default=8, help='Number of threads')
parser.add_argument('--epochs', type=int, default=190, help='Number of epochs to run')

parser.add_argument('--num_ftrs', type=int, default=512, help='Number of hidden features dimension')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes in dataset')
parser.add_argument('--input_size', type=int, default=224, help='Input image resolution')
parser.add_argument('--jitter', action='store_true', default=False, help='Whether to use colour jitter')

args = parser.parse_args()
print(f'args: {args}')


train_dir = join(args.data_dir, "train")
val_dir = join(args.data_dir, "val")
test_dir = join(args.data_dir, "test")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
pl.seed_everything(args.seed)

gpus = 1 if torch.cuda.is_available() else 0
device = 'cuda' if gpus==1 else 'cpu'

###################################### Data Loaders ##############################################
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=args.input_size,
    vf_prob=0.5,
    rr_prob=0.5,   
)

train_classifier_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(args.input_size),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((args.input_size, args.input_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

simsiam_train_dataset = lightly.data.LightlyDataset(
    input_dir=train_dir
    )

classifier_train_dataset = lightly.data.LightlyDataset(
    input_dir=train_dir,
    transform=train_classifier_transforms
    )

valid_dataset = lightly.data.LightlyDataset(
    input_dir=val_dir,
    transform=test_transforms
    )

test_dataset = lightly.data.LightlyDataset(
    input_dir=test_dir,
    transform=test_transforms
    )
 
##### Load data for SIMSIAM SSL ###########
simsiam_train_dataloader = torch.utils.data.DataLoader(
    simsiam_train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    pin_memory=True,
    num_workers=args.num_workers
    )

classifier_train_dataloader = torch.utils.data.DataLoader(
    classifier_train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    num_workers=args.num_workers
    )

valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=True,
    pin_memory=True,
    num_workers=args.num_workers
    )

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    num_workers=args.num_workers
    )

############################## SIMSIAM Model Function #####################################
class SIMSIAMModel(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        
        # create a simsiam based on model architecture
        self.simsiam = \
            lightly.models.simsiam.SimSiam(self.backbone, num_ftrs=args.num_ftrs,)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        y0, y1 = self.simsiam(x0, x1)
        loss = self.criterion(y0, y1)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optim = AdamP(self.simsiam.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs)
        return [optim], [scheduler]


class Classifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.simsiam_model = model

        for p in self.simsiam_model.parameters():  # reset requires_grad
            p.requires_grad = False

        self.fc = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(args.num_ftrs, args.num_classes)
                )

    def forward(self, x):
        with torch.no_grad():
            y_hat = self.simsiam_model.backbone(x).squeeze()
            y_hat = nn.functional.normalize(y_hat, dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss
        
    def evaluate(self, batch, stage=None):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)
        acc = accuracy(y_hat, y)
        
        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')      

    def configure_optimizers(self):
        optimizer = AdamP(self.fc.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        return [optimizer], [scheduler]
    
def get_model_backbone(model_name, pretrained, num_classes, num_ftrs):
    """
    Function to initialize model and input image size. Available models are: 
    ['resnet18', 'resnet34', 'mobilenet_v2', 'mnasnet0_5',
    'mnasnet1_0', shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'googlenet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121', 'densenet169'
    ]
    """
          
    if "resnet" in model_name:
        model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        dense_inc = list(model.children())[-1].in_features
        backbone = torch.nn.Sequential(*list(model.children())[:-1],
                                      torch.nn.Conv2d(dense_inc, num_ftrs, 1),
                                      torch.nn.AdaptiveAvgPool2d(1)
                                      )
    elif "shufflenet" in model_name:
        model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        dense_inc = list(model.children())[-1].in_features
        backbone = torch.nn.Sequential(*list(model.children())[:-1],
                                      torch.nn.Conv2d(dense_inc, num_ftrs, 4),
                                      torch.nn.AdaptiveAvgPool2d(1)
                                      )
    elif "googlenet" in model_name:
        model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        dense_inc = list(model.children())[-1].in_features
        backbone = torch.nn.Sequential(*list(model.children())[:-1],
                                      torch.nn.Conv2d(dense_inc, num_ftrs, 1),
                                      torch.nn.AdaptiveAvgPool2d(1)
                                      )
    elif 'mobilenet_v2' in model_name:
        model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        dense_inc = list(model.children())[-1][-1].in_features
        backbone = torch.nn.Sequential(*list(model.children())[:-1],
                                      torch.nn.Conv2d(dense_inc, num_ftrs, 4),
                                      torch.nn.AdaptiveAvgPool2d(1)
                                      )
    elif "mnasnet" in model_name:
        model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        dense_inc = list(model.children())[-1][-1].in_features
        backbone = torch.nn.Sequential(*list(model.children())[:-1],
                                      torch.nn.Conv2d(dense_inc, num_ftrs, 4),
                                      torch.nn.AdaptiveAvgPool2d(1)
                                      )
    elif "squeezenet" in model_name:
        model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        dense_inc = list(model.children())[-1][1].in_channels
        backbone = torch.nn.Sequential(*list(model.children())[:-1],
                                      torch.nn.Conv2d(dense_inc, num_ftrs, 6),
                                      torch.nn.AdaptiveAvgPool2d(1)
                                      )
    elif "densenet" in model_name:
        model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        dense_inc = list(model.children())[-1].in_features
        backbone = torch.nn.Sequential(*list(model.children())[:-1],
                                      torch.nn.Conv2d(dense_inc, num_ftrs, 3),
                                      torch.nn.AdaptiveAvgPool2d(1)
                                      )
    else:
        print(f"Invalid model name. Choose one of the models defined for this work")
        exit()
    return backbone

if args.arch is not None:
    model_zoo = sorted(args.arch)
else:
    model_zoo = sorted(['resnet50', 'mobilenet_v2', 'mnasnet1_0', 'shufflenet_v2_x1_0', 'googlenet', 'squeezenet1_1', 'densenet121'])
    
results_dict = {}                       
for model_name in model_zoo:
    backbone = get_model_backbone(model_name, pretrained=True, num_classes=args.num_classes, num_ftrs=args.num_ftrs)   
    print(f'\n Pre-training with self-supervised learning using SimSIAM')
    
    model = SIMSIAMModel(backbone = backbone)
    if args.jitter:
        dirpath_simsiam = join(args.ckpt_simsiam, Path(model_name+'_jitter'))
        checkpoint_simsiam = ModelCheckpoint(
             monitor='train_loss',
             mode='min',
             save_top_k=1,
             save_last=True,
             dirpath=dirpath_simsiam,
             filename=model_name)
        
        try:
            resume_simsiam = join(dirpath_simsiam, 'last.ckpt')
        except:
            print(f"No resume file")
            
        if os.path.exists(resume_simsiam):
            simsiam_trainer = pl.Trainer(max_epochs=args.epochs, 
                             gpus=gpus, gradient_clip_val=1.0, 
                             checkpoint_callback=True, 
                             auto_lr_find=True,
                             progress_bar_refresh_rate=2,
                             callbacks=[checkpoint_simsiam,],
                             resume_from_checkpoint=resume_simsiam)
            try:
                checkpoint = torch.load(sorted(glob(join(dirpath_simsiam, model_name+'*.ckpt')))[-2])
            except:
                checkpoint = torch.load(join(dirpath_simsiam, model_name+'.ckpt'))
            model.load_state_dict(checkpoint['state_dict'])
        else:
            simsiam_trainer = pl.Trainer(max_epochs=args.epochs, 
                             gpus=gpus, gradient_clip_val=1.0, 
                             checkpoint_callback=True, 
                             auto_lr_find=True,
                             progress_bar_refresh_rate=2,
                             callbacks=[checkpoint_simsiam,],
                             )
            
        print(f'Training {model_name} embeddings with jitter')
        simsiam_trainer.fit(
                model,
                simsiam_train_dataloader
                )
        ###################################################################
        print(f'Training {model_name} classifier')
        classifier = Classifier(model.simsiam)
        dirpath_class = join(args.ckpt_classifier, Path(model_name+'_jitter'))
        checkpoint_classifier = ModelCheckpoint(
             monitor='val_acc',
             mode='max',
             save_top_k=1,
             save_last=True,
             dirpath=dirpath_class,
             filename=model_name)
        
        try:
            resume_classifier = join(dirpath_class, 'last.ckpt')
        except:
            print(f"No resume file")
            
        if os.path.exists(resume_classifier):
            class_trainer = pl.Trainer(max_epochs=args.epochs, 
                             gpus=gpus, gradient_clip_val=1.0, 
                             checkpoint_callback=True, 
                             auto_lr_find=True,
                             progress_bar_refresh_rate=2,
                             callbacks=[checkpoint_classifier,],
                             resume_from_checkpoint=resume_classifier)
            try:
                checkpoint = torch.load(sorted(glob(join(dirpath_class, model_name+'*.ckpt')))[-2])
            except:
                checkpoint = torch.load(join(dirpath_class, model_name+'.ckpt'))
            classifier.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            class_trainer = pl.Trainer(max_epochs=args.epochs, 
                             gpus=gpus, gradient_clip_val=1.0, 
                             checkpoint_callback=True, 
                             auto_lr_find=True,
                             progress_bar_refresh_rate=2,
                             callbacks=[checkpoint_classifier,],
                             )
        print(f'\n Training {model_name} Classifier')                            
        class_trainer.fit(
                model=classifier,
                train_dataloaders=classifier_train_dataloader,
                val_dataloaders=valid_dataloader,
                )

        results = class_trainer.test(classifier, dataloaders=test_dataloader)                   
        ### compute per-class metrics
        classes = classifier_train_dataset.dataset.classes
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        with torch.no_grad():
            for data in test_dataloader:
                images, labels, _ = data
                outputs = classifier(images)
                _, predictions = torch.max(outputs, 1)
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        class_predictions = {}
        for class_name, correct_count in correct_pred.items():
            acc = 100 * float(correct_count) / total_pred[class_name]
            print(f"Accuracy for class {class_name} is: {acc:.2f}%")
            class_predictions[class_name] = round(acc, 2)

        results_dict[model_name] = [results, class_predictions]
        
    else:
        new_collate_fn = lightly.data.SimCLRCollateFunction(
                            input_size=args.input_size,
                            cj_prob=0.0,
                            random_gray_scale=0.0,
                            gaussian_blur=0.
                        )
        simsiam_train_dataloader.collate_fn=new_collate_fn
        
        dirpath_simsiam = join(args.ckpt_simsiam, Path(model_name+'_nojitter'))
        checkpoint_simsiam = ModelCheckpoint(
             monitor='train_loss',
             mode='min',
             save_top_k=1,
             save_last=True,
             dirpath=dirpath_simsiam,
             filename=model_name)
        
        try:
            resume_simsiam = join(dirpath_simsiam, 'last.ckpt')
        except:
            print(f"No resume file")
            
        if os.path.exists(resume_simsiam):
            simsiam_trainer = pl.Trainer(max_epochs=args.epochs, 
                             gpus=gpus, gradient_clip_val=1.0, 
                             checkpoint_callback=True, 
                             auto_lr_find=True,
                             progress_bar_refresh_rate=2,
                             callbacks=[checkpoint_simsiam,],
                             resume_from_checkpoint=resume_simsiam)
            try:
                checkpoint = torch.load(sorted(glob(join(dirpath_simsiam, model_name+'*.ckpt')))[-2])
            except:
                checkpoint = torch.load(join(dirpath_simsiam, model_name+'.ckpt'))
            model.load_state_dict(checkpoint['state_dict'])
        else:
            simsiam_trainer = pl.Trainer(max_epochs=args.epochs, 
                             gpus=gpus, gradient_clip_val=1.0, 
                             checkpoint_callback=True, 
                             auto_lr_find=True,
                             progress_bar_refresh_rate=2,
                             callbacks=[checkpoint_simsiam,],
                             )
            
        print(f'Training {model_name} embeddings with jitter')
        simsiam_trainer.fit(
                model,
                simsiam_train_dataloader
                )
        ###################################################################
        print(f'Training {model_name} classifier')
        classifier = Classifier(model.simsiam)
        dirpath_class = join(args.ckpt_classifier, Path(model_name+'_nojitter'))
        checkpoint_classifier = ModelCheckpoint(
             monitor='val_acc',
             mode='max',
             save_top_k=1,
             save_last=True,
             dirpath=dirpath_class,
             filename=model_name)
        
        try:
            resume_classifier = join(dirpath_class, 'last.ckpt')
        except:
            print(f"No resume file")
            
        if os.path.exists(resume_classifier):
            class_trainer = pl.Trainer(max_epochs=args.epochs, 
                             gpus=gpus, gradient_clip_val=1.0, 
                             checkpoint_callback=True, 
                             auto_lr_find=True,
                             progress_bar_refresh_rate=2,
                             callbacks=[checkpoint_classifier,],
                             resume_from_checkpoint=resume_classifier)
            try:
                checkpoint = torch.load(sorted(glob(join(dirpath_class, model_name+'*.ckpt')))[-2])
            except:
                checkpoint = torch.load(join(dirpath_class, model_name+'.ckpt'))
            classifier.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            class_trainer = pl.Trainer(max_epochs=args.epochs, 
                             gpus=gpus, gradient_clip_val=1.0, 
                             checkpoint_callback=True, 
                             auto_lr_find=True,
                             progress_bar_refresh_rate=2,
                             callbacks=[checkpoint_classifier,],
                             )
        print(f'\n Training {model_name} Classifier')                            
        class_trainer.fit(
                model=classifier,
                train_dataloaders=classifier_train_dataloader,
                val_dataloaders=valid_dataloader,
                )

        results = class_trainer.test(classifier, dataloaders=test_dataloader)                   
        ### compute per-class metrics
        classes = classifier_train_dataset.dataset.classes
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        with torch.no_grad():
            for data in test_dataloader:
                images, labels, _ = data
                outputs = classifier(images)
                _, predictions = torch.max(outputs, 1)
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        class_predictions = {}
        for class_name, correct_count in correct_pred.items():
            acc = 100 * float(correct_count) / total_pred[class_name]
            print(f"Accuracy for class {class_name} is: {acc:.2f}%")
            class_predictions[class_name] = round(acc, 2)

        results_dict[model_name] = [results, class_predictions]

if args.jitter:
    results_pickle = open(args.data_dir.strip('/')+'_simsiam_jitter_results.pkl', 'wb')
else:
    results_pickle = open(args.data_dir.strip('/')+'_simsiam_nojitter_results.pkl', 'wb')
pickle.dump(results_dict, results_pickle)

