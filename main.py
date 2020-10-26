'''
This variation of the stochastic tree uses a logistic regression located at each leaf in order to determine the binary class of the samples.
In this module the number of nodes includes the leaves.

'''
#load libraries
from params import parameters
from dl import get_dataloaders
from model_conf import Forest
from train_conf import Trainer
import pandas as pd
import lib
import matplotlib.pyplot as plt
import svm_tree
import torch
from sklearn.model_selection import train_test_split

#load default parameters (including device)
prms = parameters()

#dataloaders
trainset, testset, trainloader, testloader = get_dataloaders(prms)

X_train = trainset[:][0].numpy()
y_train = trainset[:][1].numpy()
X = testset[:][0].numpy()
y = testset[:][1].numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

def evaluate_network(prms,svm_init=True):

    #initiate model:
    net = Forest(prms)
    net.to(prms.device) #move model to CUDA

    if svm_init:
        net.svm_init(trainset)
        # net.svt.plot()

    #run\fit\whatever
    trainer = Trainer(prms,net)
    model_log = trainer.fit(trainloader,testloader)
    # trainer.fit(trainloader,testloader)
    # trainer.wavelet_validation(testloader)
    if prms.use_tree:
        if prms.wavelets == True:
            wav_acc = []
            for i in range(1,16):
                cutoff = int(2*i*prms.n_leaf/15) #arbitrary cutoff
                wav_acc.append(trainer.wavelet_validation(testloader,cutoff))
    
    return net,model_log

def plot_results(image_name):
    fig, sub = plt.subplots(1,1)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    ax = sub

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = lib.make_meshgrid(X0, X1)

    # for clf, title, ax in zip(models, titles, sub.flatten()):
    lib.plot_2d_function(ax, net, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('X0')
    ax.set_ylabel('X1')
    ax.set_xticks(())
    ax.set_yticks(())
    # ax.set_title(title)
    plt.savefig(image_name)

print('part 1 - no SVM initialization')
net,model_log = evaluate_network(prms,svm_init=False)
df1 = pd.DataFrame(model_log) 

if prms.save:
    torch.save(weights_list, f'weights_a{val_acc_list[-1]}e{prms.epochs}s{prms.n_samples}d{prms.tree_depth}svm_init_false.pt')
    df1.to_csv(f'a{val_acc_list[-1]}e{prms.epochs}s{prms.n_samples}d{prms.tree_depth}svm_init_false.csv')
    plot_results(f'a{val_acc_list[-1]}e{prms.epochs}s{prms.n_samples}d{prms.tree_depth}svm_init_false.png')


# print('part 2 - with SVM initialization')
# net,loss_list,val_acc_list,train_acc_list,weights_list = evaluate_network(prms,svm_init=True)

# df2 = pd.DataFrame(list(zip(loss_list,val_acc_list,train_acc_list)), 
#                columns =['loss_list','val_acc_list','train_acc_list']) 
# svt = svm_tree.svt(X_train,y_train,prms.tree_depth)
# svt_acc = net.svt.accuracy(X,y)
# if prms.save:
#     df2.to_csv(f'a{val_acc_list[-1]}e{prms.epochs}s{prms.n_samples}d{prms.tree_depth}svm_init_true{svt_acc}.csv')
#     torch.save(weights_list, f'weights_a{val_acc_list[-1]}e{prms.epochs}s{prms.n_samples}d{prms.tree_depth}svm_init_true{svt_acc}.pt')
#     plot_results(f'a{val_acc_list[-1]}e{prms.epochs}s{prms.n_samples}d{prms.tree_depth}svm_init_true{svt_acc}.png')



