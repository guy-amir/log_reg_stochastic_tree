import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from prenets import cifar_net
from svm_tree import svm_tree_init
import wavelets_conf

class Forest(nn.Module):
    def __init__(self, prms):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        self.prms = prms
        self.y_hat_avg= []
        self.mu_list = []

        #The neural network that feeds into the trees:
        if prms.dataset == 'cifar10':
            self.prenet = cifar_net(self.prms)
        # elif prms.dataset == 'wine':


        for _ in range(self.prms.n_trees):
            tree = Tree(prms)
            self.trees.append(tree)

    def forward(self, xb,layer=None, save_flag = False):

        self.save_flag = save_flag
        self.predictions = []

        # if self.training:
        #     yb_onehot = self.vec2onehot(yb)

        if self.prms.use_prenet:
            xb = self.prenet(xb)
        

        if (self.prms.use_tree == False):
            return xb


        for tree in self.trees: 
              
            # FLT_MIN = float(np.finfo(np.float32).eps)

            pred = tree(xb)
            # print(f"prediction shape {pred.shape} xb shape {xb.shape}")
            # mu += FLT_MIN #add the smallest number possible to mu in order to avoid devision by zero - not neccessary in binary version


            #pred = torch.einsum('ij,ij->i', mu_leaves, tree.leaf_reg[:,:,0]).unsqueeze(1).cuda() #MULTIPLY LEAF_REG WITH MU_LEAVES TO GET A TENSOR OF SIZE (no. of samples)X(no. of leaves)
            # pred = torch.diag(torch.mm(mu_leaves, tree.leaf_reg.double().t()))
            # pred = pred.unsqueeze(1)
            # pred = torch.cat((pred,1-pred),dim=1)
            self.predictions.append(pred.unsqueeze(2)) 
            if self.prms.wavelets and not self.training:
            #     # self.tree.wavelets = wavelet_conf.wavelet_maker()
            #     tree.mu = mu
                wavelet_norm = tree.generate_wavelet_norm(tree.mu)
                tree.wavelet_norm_list.append(wavelet_norm)


        ##GG add averaging of trees 
        self.prediction = torch.cat(self.predictions, dim=2)
        self.prediction = torch.sum(self.prediction, dim=2)/self.prms.n_trees
        
        return self.prediction

    # def predict(self,mu,yb_onehot=None):

    #     #find the nodes that are leaves:
    #     mu_midpoint = int(mu.size(1)/2)

    #     mu_leaves = mu[:,mu_midpoint:]

    #     #create a normalizing factor for leaves:
    #     N = mu.sum(0)

    #     if self.training:
    #         if self.prms.classification:
    #             self.y_hat = yb_onehot.t() @ mu.float()/N
    #             y_hat_leaves = self.y_hat[:,mu_midpoint:]
    #             self.y_hat_batch_avg.append(self.y_hat.unsqueeze(2))
    #     ####################################################################
    #     else: 
    #         y_hat_val_avg = torch.cat(self.y_hat_avg, dim=2)
    #         y_hat_val_avg = torch.sum(y_hat_val_avg, dim=2)/y_hat_val_avg.size(2)
    #         y_hat_leaves = y_hat_val_avg[:,mu_midpoint:]
    #     ####################################################################
    #     pred = (mu_leaves @ y_hat_leaves.t())

    #     if self.prms.save_flag:
    #         self.mu_list.append(mu)
    #         # self.y_hat_val_avg = y_hat_val_avg

    #     self.predictions.append(pred.unsqueeze(1))
    
    def vec2onehot(self,yb):
        yb_onehot = torch.zeros(yb.size(0), int(yb.max()+1))
        yb = yb.view(-1,1).long()
        if yb.is_cuda:
            yb_onehot = yb_onehot.cuda()
        yb_onehot.scatter_(1, yb, 1)
        return yb_onehot
        
    def svm_init(self,dataset):
        X = dataset[:][0].numpy()
        y = dataset[:][1].numpy()
        self.svt = svm_tree_init(X,y,depth=self.prms.tree_depth)

        weights = self.svt.output_weights()

        for tree in self.trees:
            #add some randomization factor
            tree.svm_init(weights)

class Tree(nn.Module):
    def __init__(self,prms):
        super(Tree, self).__init__()
        self.depth = prms.tree_depth
        self.n_nodes = prms.n_leaf
        self.mu_cache = []
        self.prms = prms

        if self.prms.wavelets:
            self.wavelet_norm_list = []

        if prms.activation == 'relu':
            self.decision = nn.ReLU()
        elif prms.activation == 'sigmoid':
            self.decision = nn.Sigmoid()

        if prms.feature_map == True:
            self.n_features = prms.feature_length
            onehot = np.eye(prms.feature_length)
            # randomly use some neurons in the feature layer to compute decision function
            self.using_idx = np.random.choice(prms.feature_length, prms.n_leaf, replace=True)
            self.feature_mask = onehot[self.using_idx].T
            self.feature_mask = nn.parameter.Parameter(torch.from_numpy(self.feature_mask).type(torch.FloatTensor), requires_grad=False)

        if prms.logistic_regression_per_node == True:
            if self.prms.feature_map == True:
                self.fc = nn.ModuleList([nn.Linear(prms.n_leaf, 1).float() for i in range(self.n_nodes)])
            else:
                self.fc = nn.ModuleList([nn.Linear(prms.feature_length, 1).float() for i in range(2*self.n_nodes)]) #we define twice the number of nodes since we have a logistic regression in each leaf and the number of leaves is equal to the number of nodes


    def forward(self, x, save_flag = False):
        if self.prms.feature_map == True:
            if x.is_cuda and not self.feature_mask.is_cuda:
                self.feature_mask = self.feature_mask.cuda()
            feats = torch.mm(x.view(-1,self.feature_mask.size(0)).float(), self.feature_mask)
        else:
            feats = x

        self.d = [self.decision(node(feats.float())) for node in self.fc]
        
        self.d = torch.stack(self.d)

        decision = torch.cat((self.d,1-self.d),dim=2).permute(1,0,2)
        
        batch_size = x.size()[0]
        mu = x.data.new(x.size(0),1,1).fill_(1.).double()
        big_mu = x.data.new(x.size(0),2,1).fill_(1.).double()
        begin_idx = 1
        end_idx = 2
        
        for n_layer in range(0, self.depth):
            # mu stores the probability a sample is routed at certain node
            # repeat it to be multiplied for left and right routing
            mu = mu.repeat(1, 1, 2)
            # the routing probability at n_layer
            _decision = decision[:, begin_idx:end_idx, :].double() # -> [batch_size,2**n_layer,2]
            mu = mu*_decision # -> [batch_size,2**n_layer,2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (n_layer+1)
            # merge left and right nodes to the same layer
            mu = mu.view(x.size(0), -1, 1)
            big_mu = torch.cat((big_mu,mu),1)

        self.leaf_reg = decision[:, begin_idx:end_idx, 0]

        big_mu = big_mu.view(x.size(0), -1)    
        # self.mu_cache.append(big_mu)  

        if self.prms.pruned_tree:
            pruned_mu = torch.zeros(big_mu.size()).double().cuda()
            pruned_mu[:,self.cutoff_indices_with_parents] = big_mu[:,self.cutoff_indices_with_parents]
            only_child_list = self.has_siblings(self.cutoff_indices_with_parents)
            parent_list = self.get_parent_list(only_child_list)
            for i in range(self.prms.tree_depth): #propagating the parent value through all the only children
                pruned_mu[0,only_child_list]=pruned_mu[0,parent_list]
            self.mod_leaves = self.find_leaves(self.cutoff_indices_with_parents)

            #build pruned_reg_tree
            pruned_reg_tree = self.reg_tree[:,self.mod_leaves]
            #multiply pruned_leaf_reg with pruned_mu
            pred = torch.diag(torch.mm(pruned_mu[:,self.mod_leaves], pruned_reg_tree.double().t()))
            pred = pred.unsqueeze(1)
            pred = torch.cat((pred,1-pred),dim=1)
            #return prediction
            return pred

        if self.prms.wavelets and not self.training:
            # self.tree.wavelets = wavelet_conf.wavelet_maker()
            self.mu = big_mu

        mu_midpoint = int(big_mu.size(1)/2)
        big_mu = big_mu[:,mu_midpoint:]

        pred = torch.diag(torch.mm(big_mu, self.leaf_reg.double().t()))
        pred = pred.unsqueeze(1)
        pred = torch.cat((pred,1-pred),dim=1)

        return pred #-> [batch size,n_leaf]

    def find_leaves(self,node_list):
        leaf_list = []
        for node in node_list:
            if (2*node not in node_list) and (2*node+1 not in node_list):
                leaf_list.append(node)
        return leaf_list

    def has_siblings(self,cutoff_indices_with_parents):
        only_child = [] #len(cutoff_indices_with_parents)*[0]
        for i,node_number in enumerate(cutoff_indices_with_parents):
            if node_number%2: #if odd
                if (node_number-1) not in cutoff_indices_with_parents:
                    if node_number>1:
                        only_child.append(node_number)
            else: #if even
                if (node_number+1) not in cutoff_indices_with_parents:
                    only_child.append(node_number)
        return only_child
    def get_parent_list(self,node_list):
        parent_list = []
        for node in node_list:
            parent_list.append(node//2)
        return parent_list

    def svm_init(self,weights):

        for i in range(1,len(self.fc)):
            self.fc[i].weight.data = torch.tensor([weights[i-1][0:2]]).float().cuda()
            self.fc[i].bias.data = torch.tensor([weights[i-1][2]]).float().cuda()

    #wavelet stuff

    #sort
    #find all parents
    #cutoff
    #create new tree (perhaps put 0 in deleted mu values and 1 in nodes with no choice)

    def generate_wavelet_norm(self,mu):
        # mu
        # self.leaf_reg
        self.reg_tree = self.build_reg_tree()
        node_value_diff = torch.stack([self.reg_tree[:,i]-self.reg_tree[:,i//2] for i in range(self.reg_tree.size(1))]).t()
        # wavelets = mu*node_value_diff
        wavelet_norms = torch.abs(mu.float()*node_value_diff)
        wavelet_norm_means = wavelet_norms.mean(0)
        return wavelet_norm_means

    def build_reg_tree(self):
        N = self.leaf_reg.size(1)//2
        parent_level = self.leaf_reg.clone()
        reg_tree = self.leaf_reg.clone()
        while N!=0:
            #do something
            parent_level = [parent_level[:,2*i:2*i+2].sum(1)/2 for i in range(N)]
            parent_level = torch.stack(parent_level).t()
            reg_tree = torch.cat((parent_level,reg_tree),1)
            N = N//2
        reg_tree = torch.cat((torch.ones(parent_level.size()).cuda(),reg_tree),1)
        return reg_tree


def level2nodes(tree_level):
    return 2**(tree_level+1)

def level2node_delta(tree_level):
    start = level2nodes(tree_level-1)
    end = level2nodes(tree_level)
    return [start,end]