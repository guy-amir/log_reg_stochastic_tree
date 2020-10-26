import torchvision
import sklearn
import torch
import torchvision.transforms as transforms
import pandas as pd
import sklearn
from torch.utils.data import DataLoader, Dataset, random_split

def get_dataloaders(prms):
    if prms.dataset == 'cifar10':
        return cifar_dl(prms)
    if prms.dataset == 'diabetes':
        return diabetes_dl(prms)
    if prms.dataset == 'red_wine':
        return red_wine_dl(prms)
    if prms.dataset == 'circle':
        return circle_dl(prms)
    if prms.dataset == 'moons':
        return moons_dl(prms)

#############################################################################################################3
def moons_dl(prms):
    
    init_dataset = moonsDataset(prms)
    params = {'batch_size': prms.n_samples,
          'shuffle': True,
          'num_workers': 6}
    
    lengths = [int(prms.n_samples*0.8), prms.n_samples-int(prms.n_samples*0.8)]
    train_dataset,valid_dataset = random_split(init_dataset, lengths)

    train_loader = DataLoader(train_dataset, **params)
    test_loader = DataLoader(valid_dataset, **params)

    return train_dataset,valid_dataset,train_loader,test_loader

class moonsDataset(Dataset):
    
    def __init__(self,prms):
        'Initialization'
        
        noisy_moons = sklearn.datasets.make_moons(n_samples=prms.n_samples, noise=prms.noise)
        X,Y = noisy_moons

        self.samples = torch.tensor(X)
        self.labels = torch.tensor(Y)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        X = self.samples[index]
        y = self.labels[index]

        return X, y

def red_wine_dl(prms):
    
    init_dataset = redWineDataset(prms)
    wine_length = 1599
    params = {'batch_size': wine_length,
          'shuffle': True,
          'num_workers': 6}
    
    lengths = [int(wine_length*0.8), wine_length-int(wine_length*0.8)]
    train_dataset,valid_dataset = random_split(init_dataset, lengths)

    train_loader = DataLoader(train_dataset, **params)
    test_loader = DataLoader(valid_dataset, **params)

    return train_dataset,valid_dataset,train_loader,test_loader

class redWineDataset(Dataset):
    
    def __init__(self,prms):
        'Initialization'
        DATA_PATH = prms.data_path
        wine = pd.read_csv(f'{DATA_PATH}/red_wine.csv', sep=';')
        
        X=wine[wine.columns[:11]]
        Y=wine['quality']-3

        self.samples = torch.tensor(X.values)
        self.labels = torch.tensor(Y.values)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        X = self.samples[index]
        y = self.labels[index]

        return X, y

#####################################################################################################################33       
def diabetes_dl(prms):
    
    init_dataset = diabetesDataset(prms)
    diab_length = 768
    params = {'batch_size': diab_length,
          'shuffle': True,
          'num_workers': 6}
    
    lengths = [int(diab_length*0.8), diab_length-int(diab_length*0.8)]
    train_dataset,valid_dataset = random_split(init_dataset, lengths)

    train_loader = DataLoader(train_dataset, **params)
    test_loader = DataLoader(valid_dataset, **params)

    return train_dataset,valid_dataset,train_loader,test_loader

class diabetesDataset(Dataset):
    
    def __init__(self,prms):
        'Initialization'
        DATA_PATH = prms.data_path
        diab = pd.read_csv(f'{DATA_PATH}/diabetes.csv')
        
        X=diab[diab.columns[:8]]
        Y=diab['Outcome']

        self.samples = torch.tensor(X.values)
        self.labels = torch.tensor(Y.values)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        X = self.samples[index]
        y = self.labels[index]

        return X, y

####################################################################################################
def cifar_dl(prms):
    DATA_PATH = prms.data_path
    train_bs = prms.train_bs
    test_bs = prms.test_bs

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs,
                                            shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs,
                                            shuffle=False, num_workers=4)
    
    return trainset, testset, trainloader, testloader

##################################################################################################


def generate_circle(samples=2000,dim=2):
    X = np.random.rand(samples,dim)
    Y = np.zeros(samples)
    Y[(X[:,0]-0.5)**2+(X[:,1]-0.5)**2<0.16]=1

    return X,Y

def circle_dl(prms):
    
    init_dataset = circleDataset(prms)
    params = {'batch_size': prms.train_bs,
          'shuffle': True,
          'num_workers': 6}
    
    lengths = [int(init_dataset.n_samples*0.8), init_dataset.n_samples-int(init_dataset.n_samples*0.8)]
    train_dataset,valid_dataset = random_split(init_dataset, lengths)

    train_loader = DataLoader(train_dataset, **params)
    test_loader = DataLoader(valid_dataset, **params)

    return train_dataset,valid_dataset,train_loader,test_loader

class circleDataset(Dataset):
    
    def __init__(self,prms):
        'Initialization'
        
        self.n_samples = prms.n_samples
        dim = 2 #prms.dim

        if prms.gaussian_generation:
            self.samples = torch.normal(0.5, 1.0, size=(self.n_samples,dim))
        else:
            self.samples = torch.rand(self.n_samples,dim)
        
        self.labels  = torch.zeros(self.n_samples)
        self.labels[(self.samples[:,0]-0.5)**2+(self.samples[:,1]-0.5)**2<0.16]=1

        # self.samples = torch.tensor(X.values)
        # self.labels = torch.tensor(Y.values)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        X = self.samples[index]
        y = self.labels[index]

        return X, y
