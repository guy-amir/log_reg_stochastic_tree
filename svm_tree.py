from svm_tree_lib import *
import numpy as np
from sklearn.metrics import accuracy_score

def svm_tree_init(X,y,depth=3):

    nt = svt(X,y,depth)
    

    return nt

class svt:
    def __init__(self,X,y,depth=3):
        self.X = X
        self.y = y
        self.depth=depth
        self.nodes = []
        self.root = node(self,None,X,y,1,0)
        
        self.sort()
        
    def sort(self):
        a = [node.number for node in self.nodes]
        sortkey = list(np.argsort(a))
        self.nodes = [self.nodes[int(i)] for i in sortkey]
        
    def plot(self):
        
        fig, sub = plt.subplots(1,1)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        X = self.X
        y = self.y
        ax = sub
        
        for node in self.nodes:
            
            w = node.w

            X0, X1 = X[:,0], X[:,1]
            xx, yy = lib.make_meshgrid(X0, X1)

            # # ###########################################################################

            a = -w[0] / w[1]
            XX = np.linspace(-5, 5)
            YY = a * XX - w[2] / w[1]
            ax.plot(XX, YY, "-")
            # # ###########################################################################
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())

        plt.show()
        
    def output_weights(self):
        
        weights = [node.w for node in self.nodes]

        
        return weights
        
    def predict(self,X):
        weights = self.output_weights()
        splits = []
        
        for w in weights:
            splits.append(split_testset(X,w))
        a = np.array(splits)
        y_pred = np.zeros(np.shape(a)[1])
        for ii in range(np.shape(a)[1]):
            b = a[:,ii]
            j=1
            for kk in range(self.depth):
                if b[j-1]:
                    j=2*j+1
                else:
                    j=2*j
            y_pred[ii] = self.nodes[j-1].value
#             print(f'{ii+1}th sample reaches leaf {j} and has a prediction value of {y_pred[ii]}')
        return y_pred

    def accuracy(self,X,y):
        y_pred = self.predict(X)
        y_pred2 = (y_pred>0.5)
        y_pred2.astype(int)
        return accuracy_score(y, y_pred2)
        
class node:
    def __init__(self,tree,parent,Xn,yn,node_number,level):
        
        if parent==None:
            self.root = True
        else:
            self.root = False
            
        self.X = Xn
        self.y = yn
        self.number = node_number
        self.level = level
        self.tree = tree
        self.tree.nodes.append(self)
        self.parent = parent
        
        #define node value
        self.value = np.mean(yn)
        if np.isnan(self.value):
            self.value = parent.value

        #parameterize svm
        if len(yn) and np.std(yn):
            self.w = params_to_coef(Xn,yn)
        else:
            self.w = parent.w
        #calculate SVM over children
        if level<tree.depth:
            X_left, y_left,  X_right,  y_right = split_dataset(Xn,yn,self.w)
            self.l = node(tree,self,X_left, y_left,2*node_number,level+1)
            self.r = node(tree,self,X_right, y_right,2*node_number+1,level+1)
            self.leaf=False
        else:
            self.leaf=True
            
#     def predict(self,X):
#         if leaf==False:
#             X_left, y_left,  X_right,  y_right = split_testset(X,y,self.w)
#         else:
            