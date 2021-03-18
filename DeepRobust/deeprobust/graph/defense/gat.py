
"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from DeepRobust.deeprobust.graph.defense.MyGATConv import GATConv
from dgl.data import register_data_args, load_data
import dgl
from DeepRobust.deeprobust.graph.defense.utils import EarlyStopping
import torch.nn.functional as F
import argparse
import numpy as np
import time

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,cuda=True):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # self.cuda = cuda
        # input projection (no residual)
        self.gat_layers.append(GATConv(in_dim, num_hidden, heads[0], feat_drop, attn_drop, negative_slope, False, self.activation, cuda=self.cuda))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(num_hidden * heads[l-1], num_hidden, heads[l], feat_drop, attn_drop, negative_slope, residual, self.activation,cuda=self.cuda))
        # output projection
        self.gat_layers.append(GATConv(num_hidden * heads[-2], num_classes, heads[-1], feat_drop, attn_drop, negative_slope, residual, None,cuda=self.cuda))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits

    def evaluate(self, g, features, labels, mask):
        self.eval()
        with torch.no_grad():
            logits = self.forward(g, features)
            logits = logits[mask]
            labels = labels[mask]
            return accuracy(logits, labels)

    def fit(self, features, g, labels, train_mask, val_mask, cuda=True, iters=200, early_stop=True, fastmode=False, lr=0.05, weight_decay=1e-3):
        self.features = features
        self.labels =labels
        if early_stop:
            stopper = EarlyStopping(patience=100)
        if cuda:
            self.cuda()
        loss_fcn = torch.nn.CrossEntropyLoss()

        # use optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)

        # initialize graph
        dur = []
        n_edges = g.number_of_edges()
        for epoch in range(iters):
            self.train()
            if epoch >= 3:
                t0 = time.time()
            # forward
            logits = self.forward(g, features)

            loss = loss_fcn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            train_acc = accuracy(logits[train_mask], labels[train_mask])

            if fastmode:
                val_acc = accuracy(logits[val_mask], labels[val_mask])
            else:
                val_acc = self.evaluate(g, features, labels, val_mask)
                if early_stop:
                    if stopper.step(val_acc, self):
                        break

            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
                  " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
                  format(epoch, np.mean(dur), loss.item(), train_acc,
                         val_acc, n_edges / np.mean(dur) / 1000))


    def test(self,g, test_mask, early_stop=False):
        print()
        if early_stop:
            self.load_state_dict(torch.load('es_checkpoint.pt'))
        acc = self.evaluate(g, self.features, self.labels, test_mask)
        print("Test Accuracy {:.4f}".format(acc))


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def generate_data(args):
    data = load_data(args)
    labels = torch.LongTensor(data.labels)
    features = torch.FloatTensor(data.features)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)

    num_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d 
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g).to('cuda:0')
    g.add_edges(g.nodes(), g.nodes())
    netg = nx.from_numpy_matrix(g.adjacency_matrix().to_dense().numpy(), create_using=nx.DiGraph)
    print(netg)
    g= dgl.from_networkx(netg, edge_attrs=['weight']).to("cuda:0")
    n_edges = g.number_of_edges()
    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    print("train_mask-shape",train_mask)
    return g,num_feats,n_classes, heads, cuda, features,labels,train_mask,val_mask,test_mask




from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import networkx as nx


def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    g = DGLGraph(data.graph)

    return g, features, labels, train_mask, test_mask



def  main(args):
    load_cora_data()
    g, num_feats, n_classes, heads, cuda, features, labels, train_mask, val_mask, test_mask= generate_data(args)
    model = GAT(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)
    model.fit(features, g, labels, train_mask, val_mask, cuda=cuda,iters=args.epochs,fastmode=args.fastmode, early_stop=args.early_stop)

    model.test(g, test_mask)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    args = parser.parse_args()
    args.dataset='cora'
    print(args)

    main(args)