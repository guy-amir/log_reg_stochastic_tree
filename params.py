import torch

class parameters():
    def __init__(self):

        #General parameters
        self.output_path = './results/DL_layers/analysis'
        self.archive_path = './archive'
        self.save = True

        #Computational parameters:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")


        #Dataset parameters:
        self.gaussian_generation = False
        self.dataset = 'moons' #'cifar10' #"mnist" #'wine'
        self.data_path = '../data'
        self.train_bs = 256
        self.test_bs = 512
        if self.dataset == 'cifar10':
            self.feature_length = 256
            self.n_classes = 10
        elif self.dataset == 'diabetes':
            self.feature_length = 8
            self.n_classes = 2
        elif self.dataset == 'red_wine':
            self.feature_length = 11
            self.n_classes = 6
        elif self.dataset == 'white_wine':
            self.feature_length = 11
            self.n_classes = 6
        elif self.dataset == 'moons':
            self.feature_length = 2
            self.n_classes = 2
        elif self.dataset == 'circle':
            self.feature_length = 2
            self.n_classes = 2
        self.noise=.15
        self.n_samples = 20000


        #NN parameters:
        # self.batchnorm = True

        #Forest parameters:
        self.use_tree = True
        self.use_prenet = False
        self.classification = True
        self.use_pi = False 
        self.logistic_leaves = True

        self.n_trees = 1

        #Tree parameters:
        self.tree_depth = 5
        self.n_leaf = 2**self.tree_depth


        self.cascading = False
        self.single_level_training = True
        self.features4tree = 1
        self.logistic_regression_per_node = True
        self.feature_map = False
        self.activation = 'sigmoid'
        self.save_flag = False
        

        #Training parameters:
        self.epochs = 100
        # self.batch_size = 64
        self.learning_rate = 0.03
        self.weight_decay=1e-4
        self.momentum=0.9
        self.optimizer = 'Adam'
        self.save_every = 1

        #Wavelet parameters:
        self.wavelets = True
        self.pruned_tree = False
        self.intervals = 400

        #smoothness parameters:
        self.check_smoothness = False
