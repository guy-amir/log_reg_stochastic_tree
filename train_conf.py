from collections import OrderedDict
import torch.optim as optim
import torch.nn as nn
import torch
# from smooth import smoothness_layers

class Trainer():
    def __init__(self,prms,net):
        self.prms = prms
        self.net = net
        if prms.use_tree == True:
            self.criterion = nn.NLLLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        if prms.optimizer == 'SGD':
            self.optimizer = optim.SGD(net.parameters(), lr=prms.learning_rate, momentum=prms.momentum, weight_decay=self.prms.weight_decay)
        if prms.optimizer == 'Adam':
            self.optimizer = optim.Adam(net.parameters(), lr=prms.learning_rate, weight_decay=self.prms.weight_decay)

    def validation(self,testloader):
        self.net.train(False)
        prms = self.prms
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(prms.device), data[1].to(prms.device)
                preds = self.net(images)
                _, predicted = torch.max(preds, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                acc = 100 * correct / total

        print(f'Accuracy of the network on the validation set: {acc}')
        return acc

    def wavelet_validation(self,testloader,threshold=5):
        self.net.prms.wavelets = True
        self.net.train(False)
        prms = self.prms
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(prms.device), data[1].to(prms.device)
                preds = self.net(images)
                _, predicted = torch.max(preds, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                acc = 100 * correct / total
                #here we create wavelet_norm_list

        wavelet_norms_of_trees = []
        for tree in self.net.trees:
            wavelet_norm = torch.stack(tree.wavelet_norm_list,0)
            wavelet_norm = wavelet_norm.mean(0)
            wavelet_norms_of_trees.append(wavelet_norm)
        #fix this in the future, when relevant:
        wavelet_norms_of_trees = torch.stack(wavelet_norms_of_trees,0)
        wavelet_norms_of_trees = wavelet_norms_of_trees.mean(0)

        #def prune tree:
        sorted_wavelets, wavelet_indices = torch.sort(-wavelet_norms_of_trees)
        sorted_wavelets = -sorted_wavelets
        cutoff_wavelets = sorted_wavelets[:threshold]
        cutoff_indices = wavelet_indices[:threshold]    
        cutoff_indices_with_parents = self.add_parents(cutoff_indices)

        # self.net.pruning = True
        for tree in self.net.trees:
            tree.pruned_mu,tree.pruned_leaves = self.mod_tree(tree.mu,cutoff_indices_with_parents)
            tree.cutoff_indices_with_parents = cutoff_indices_with_parents

        
        self.net.prms.pruned_tree = True
        
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(prms.device), data[1].to(prms.device)
                preds = self.net(images)
                _, predicted = torch.max(preds, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                acc = 100 * correct / total
        # pruned_mu, pruned_log_reg = self.create_pruned_tree(mu, log_reg,cutoff_wavelets_with_parents)

        print(f'Accuracy of pruned validation set: {acc}')
        self.net.prms.pruned_tree = False
        return acc

    # # def wavelet_validation(self,testloader,cutoff):
    #     self.net.train(False)
    #     prms = self.prms
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for data in testloader:
    #             images, labels = data[0].to(prms.device), data[1].to(prms.device)
    #             preds = self.net(images, save_flag=True)

    #             _, predicted = torch.max(preds, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()
    #         #this is where the magic happens:
    #         # 1. Calcuate phi:
    #         # y = self.net.y_hat_val_avg #just create a shorthand to save typing a long name
    #         mu_list = [tree.mu for tree in self.net.trees] #just create a shorthand to save typing a long name
    #         logreg_list = [tree.leaf_reg[:,:,0] for tree in self.net.trees]
    #         # fixed_mu = [m for m in mu if m.size(0)==1024] #remove all the mus with less than 1024 samples
    #         mu = mu_list[0] #this workes for the case of 1 tree. if we want to add more trees we should change it
    #         logreg = logreg_list[0]
    #         # mu = sum(fixed_mu)/(len(fixed_mu))
    #         # mu = mu.mean(0)

    #         phi,phi_norm,sorted_nodes = self.phi_maker(logreg,mu)

    #         # 3. cutoff and add parents
    #         cutoff_nodes = sorted_nodes[:cutoff]

    #         for node in cutoff_nodes:

    #             for parent in self.find_parents(node.item()):

    #                 mask = (cutoff_nodes == parent.cpu())

    #                 if mask.sum() == 0:
    #                     cutoff_nodes = cutoff_nodes.tolist()
    #                     cutoff_nodes.append(parent.item())
    #                     cutoff_nodes = torch.LongTensor(cutoff_nodes)

    #         # 5. calculate values in new tree
    #         correct = 0
    #         total = 0
    #         for data in testloader:
    #             images, labels = data[0].to(prms.device), data[1].to(prms.device)
    #             preds = self.net.forward_wavelets(xb = images, yb = labels, cutoff_nodes=cutoff_nodes)
    #             if self.prms.check_smoothness == True:
    #                 preds = preds[-1]
    #             _, predicted = torch.max(preds, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()

    #             acc = 100 * correct / total

    #     print(f'Accuracy of the network with {cutoff} wavelets on the 10000 test images: {acc}')
    #     return acc

    def mod_tree(self,mu,cutoff_indices_with_parents):
        pruned_mu = torch.zeros(mu.size()).double().cuda()
        pruned_mu[:,cutoff_indices_with_parents] = mu[:,cutoff_indices_with_parents]

        #here we pretend to normalize mu (low priority right now)
        #new_mu = self.normalize_mu(new_mu)

        #determine whitch nodes are leaves
        pruned_leaves = self.find_leaves(cutoff_indices_with_parents)

        return pruned_mu,pruned_leaves

    def find_leaves(self,node_list):
        leaf_list = []
        for node in node_list:
            if (2*node not in node_list) and (2*node+1 not in node_list):
                leaf_list.append(node)
        return leaf_list

        # #this funciton returns a list of the modified trees leaves
        # leaf_list = []
        # for node in node_list:
        #     #check if no left-node children:
        #     if (not (node_list == 2*node+1).sum().item()) & (not (node_list == 2*node+2).sum().item()):
        #         if (node%2).item():
        #             if (node_list == node+1).sum().item():
        #                 leaf_list.append(node)
        #             else:
        #                 leaf_list.append((node-1)//2)
        #         else: 
        #             if (node_list == node-1).sum().item():
        #                 leaf_list.append(node)
        #             else:
        #                 leaf_list.append((node-1)//2)
        # return torch.unique(torch.FloatTensor(leaf_list))

    def add_parents(self,cutoff_indices):
        ## this function adds the parent nodes to the list of nodes in tree
        # parents = [self.find_parents(node_number.item()) for node_number in cutoff_indices]
        cutoff_indices_with_parents = cutoff_indices.tolist()
        for node_number in cutoff_indices:
            cutoff_indices_with_parents = cutoff_indices_with_parents+self.find_parents(node_number.item())
        cutoff_indices_with_parents = list(set(cutoff_indices_with_parents)) #retain only unique values
        return cutoff_indices_with_parents

    def phi_maker(self,logreg,mu):
        phi = torch.zeros(logreg.size())
        phi_norm = torch.zeros(logreg.size(1))
        #calculate the phis and the norms:
        for i in range(2,logreg.size(1)):
            p = self.find_parents(i)[0]
            phi[:,i] = mu[i]*(logreg[:,i]-logreg[:,p])
            phi_norm[i] = phi[:,i].norm(2)
        #Order phis from large to small:
        _,sorted_nodes = torch.sort(-phi_norm)
        return phi,phi_norm,sorted_nodes

    def find_parents(self,N):
        parent_list = []
        current_parent = N//2
        while(current_parent is not 0):
            parent_list.append(current_parent)
            current_parent = current_parent//2
        return parent_list
    
    def fit(self,trainloader,testloader):
        self.net.train(True)
        prms = self.prms
        self.net.y_hat_avg = []

        self.loss_list = []
        self.val_acc_list = []
        self.train_acc_list = []
        self.wav_acc_list = []
        self.smooth_list = []
        self.cutoff_list = []
        self.weights_list = []

        for epoch in range(prms.epochs):  # loop over the dataset multiple times


            self.net.train(True)
            #add if for tree:
            # print(f'epoch {epoch}')
            self.net.y_hat_batch_avg = []

            total = 0
            correct = 0
            running_loss = 0.0
            long_running_loss = 0.0
            for i, data in enumerate(trainloader, 0):

                # get the x; data is a list of [x, y]
                xb, yb = data[0].to(prms.device), data[1].to(prms.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                
                preds = self.net(xb)
                if prms.use_tree==True:
                    loss = self.criterion(torch.log(preds), yb.long())
                else:
                    loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                long_running_loss  += loss.item()
                # if i % 50 == 49:    # print every 50 mini-batches

                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss}')
                running_loss = 0.0

                # self.wavelet_validation(testloader)

            
            _, predicted = torch.max(preds, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
            train_acc = 100 * correct / total

            if prms.check_smoothness:
                preds_list = self.net.pred_list
                smooth_layers = smoothness_layers(preds_list,yb)


            # if prms.use_tree:
            #     if prms.wavelets == True:
            #         wav_acc = []
            #         for i in range(1,6):
            #             cutoff = int(2*i*prms.n_leaf/5) #arbitrary cutoff
            #             wav_acc.append(self.wavelet_validation(testloader,cutoff))

            #convert weights to list:
            w_list_raw = [p.tolist()[0] for p in list(self.net.parameters())]
            if self.prms.use_pi:
                w_list = [[w_list_raw[2*i]]+w_list_raw[2*i+1] for i in range(int(len(w_list_raw)/2))]
            else:
                w_list = [w_list_raw[2*i]+[w_list_raw[2*i+1]] for i in range(int(len(w_list_raw)/2))]
            self.weights_list.append(w_list)


            self.loss_list.append(long_running_loss)
            val_acc = self.validation(testloader)
            self.val_acc_list.append(val_acc)
            self.train_acc_list.append(train_acc)
            # if prms.use_tree and prms.wavelets:
            #     self.wav_acc_list.append(wav_acc)
            self.cutoff_list = [int(i*prms.n_leaf/5) for i in range(1,6)]
            if prms.check_smoothness == True:
                self.smooth_list.append(smooth_layers)

            if epoch % prms.save_every == 0:			
                checkpoint_path = f"{prms.output_path}/weights.{epoch}.h5"
                model_state_dict = self.net.state_dict()
                state_dict = OrderedDict()
                state_dict["epoch"] = epoch
                state_dict["checkpoint"] = model_state_dict
                state_dict["train_acc"] = train_acc
                state_dict["valid_acc"] = val_acc
                torch.save(state_dict, checkpoint_path)
            
        #weights_list is a 3d tensor of trained weights of all epochs
        #its shape is (number of epochs)x(number of nodes)x(number of weights+bias)
        self.weights_list = torch.tensor(self.weights_list)

        model_log = {'loss_list':self.loss_list,'val_acc':self.val_acc_list,'train_acc':self.train_acc_list}#,self.weights_list,self.wav_acc_list,self.cutoff_list,self.smooth_list}
        return model_log
        
