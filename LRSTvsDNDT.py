# from train import dndt_trainer
# from dataset import moons_dl
from params import parameters
from dl import get_dataloaders
from model_conf import Forest
from train_conf import Trainer
import pandas as pd
import lib
import matplotlib.pyplot as plt
import svm_tree
import torch

import sklearn.datasets
import torch
import matplotlib.pyplot as plt
import numpy as np
import utils.contour_plots
from itertools import cycle, islice
import pandas as pd

from sklearn.model_selection import train_test_split

from params import parameters


from utils.contour_plots import plot_results
prms = parameters()

# dataset_noise = .15
samples = 10000
prms.epochs = 500
n_bins = 8

sample_range = [5000,10000]
noise_range = [0.15,0.35,0.55]
lr_range = [0.1,0.03,0.01]

# for samples in sample_range:    
for noise in noise_range:
    prms.n_samples = samples
    prms.noise = noise
    trainset, testset, trainloader, testloader = get_dataloaders(prms)
    a = [[inputs, labels] for [inputs, labels] in testloader]
    X = a[0][0]
    y = a[0][1]
    # for n_bins in range(2,5):
        # for lr in lr_range:
    prms.lr = 0.1
    prms.depth = 8
    lrst = Forest(prms)
    lrst.to(prms.device) #move model to CUDA
    trainer = Trainer(prms,lrst)
    model_log = trainer.fit(trainloader,testloader)
    csv_name = f"LRSTsamples{samples}noise{int(100*noise)}bins{n_bins}epochs{prms.epochs}.csv"
    png_name = f"LRSTsamples{samples}noise{int(100*noise)}bins{n_bins}epochs{prms.epochs}.png"
    print(csv_name)
    ml = pd.DataFrame(model_log)
    ml.to_csv(csv_name,index=False)
    plot_results(X,y,lrst,image_name=png_name)
