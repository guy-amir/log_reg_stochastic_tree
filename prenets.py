import torch.nn as nn
import torch.nn.functional as F
import torch

class cifar_net(nn.Module):
    def __init__(self,prms=None):
        super(cifar_net, self).__init__()

        self.prms = prms

        self.conv_layer1 = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.conv_layer2 = nn.Sequential(
            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
        )
        
        self.conv_layer3 = nn.Sequential(
            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer1 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        
        # self.fc_layer2 = nn.Sequential(
        #     nn.Linear(1024, 512),
            
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(512, 10)
        # )

        self.fc_layer2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256)
        )

        # self.for_tree = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(512, 10)
        # )

        # self.softmax = nn.Sequential(
        #     nn.Softmax() #dim=1) #maybe add dim if necessarry
        # )

    def forward(self, x):

        cl1 = self.conv_layer1(x)
        cl2 = self.conv_layer2(cl1)
        cl3 = self.conv_layer3(cl2)

        # flatten
        cl3 = cl3.view(cl3.size(0), -1)
        
        # fc layer
        fc1 = self.fc_layer1(cl3)
        fc2 = self.fc_layer2(fc1)

        #softmax
        # sm = self.softmax(fc2)

        if self.prms.check_smoothness:
            return x,cl1,cl2,cl3,fc1,fc2 #option a - smoothness testing
        else:
            return fc2 #option b - no smoothness testing