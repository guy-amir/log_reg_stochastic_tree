import torch
import numpy
import lib

from svm_tree import svm_tree_init

# #load dataset
# X,y = lib.generate_circle(samples=500)

# weights = svm_tree_init(X,y,depth=3)

# fc = torch.nn.ModuleList([torch.nn.Linear(2, 1).float() for i in range(7)])

# def svm_weight_init()
# for i,node in enumerate(fc):
#     node.weight.data = torch.tensor([weights[i][0:2]])
#     node.bias.data = torch.tensor([weights[i][2]])

# print(f'weights {weights}')



samples = torch.normal(mean=0.5,std=1.0,generator=None, out=None)

print('hi')