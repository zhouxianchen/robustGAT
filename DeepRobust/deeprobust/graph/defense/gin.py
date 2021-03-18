import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import GINConv
from dgl import DGLGraph

class GIN(Module):
    """
    The module of GIN, implmented by DGL
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.5 , lr=0.01, weight_decay=5e-4, device=None):

        super(GIN, self).__init__()

        self.dropout = dropout
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = nhid
        self.nclass= nclass
        self.gc1 = GINConv(apply_func=None, aggregator_type='sum',init_eps=0.03)
        self.gc2 = nn.Linear(nfeat,nhid,bias=True)
        self.gc3 = nn.Linear(nhid, nclass, bias=True)
        self.lr = lr
        self.weight_decay =weight_decay

    def forward(self, graph, feat):
        '''
                    graph normalized adjacency matrix of dglgraph
        '''
        x = self.gc1(graph, feat)
        x = self.gc2(x)
        x = F.relu(x)
        x = self.gc3(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    def fit(self, features, g, labels, idx_train, train_iters=200, initialize=True, verbose=False):
        """
        train the GIN
        """
        # if initialize:
        #     self.initialize()

        self.g = g
        self.features = features
        self.labels = labels

        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.g, self.features)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
        self.eval()
        output = self.forward(self.g, self.features)


    def test(self, idx_test):
        self.eval()
        output = self.forward(g, features)
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test
    #
    # def initialize(self):
    #     self.gc1.reset_parameters()
    #     self.gc2.reset_parameters()
    #     self.gc3.reset_parameters()


from dgl.data import citation_graph as citegrh


def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    g = DGLGraph(data.graph)
    return g, features, labels, train_mask, test_mask


if __name__=="__main__":
    g, features, labels, train_mask, test_mask = load_cora_data()
    n,k = features.shape
    print(k)
    my_gin = GIN(nfeat=k, nhid=64, nclass=7)
    my_gin.fit(features,g, labels,  train_mask, train_iters=200, verbose=True)
    my_gin.test(test_mask)

