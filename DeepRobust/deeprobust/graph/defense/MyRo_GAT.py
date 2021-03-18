"""
This model is using the GAT with revised A,X，（my method).
"""

import torch
import torch.nn as nn
import scipy.sparse as sp
import torchvision
import numpy as np
import time
import torch.optim as optim
import torch.nn.functional as F
from DeepRobust.deeprobust.graph.utils import accuracy
from DeepRobust.deeprobust.graph.defense.pgd import PGD, prox_operators
from copy import deepcopy
import dgl
import logging
from dgl import DGLGraph
import networkx as nx
print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def sparse_diag(input_tensor):
    """
    Returns the identity matrix as a sparse matrix
    """
    size = input_tensor.shape[0]
    indices = torch.arange(0, size).long().unsqueeze(0).expand(2, size).cuda()
    return torch.sparse.FloatTensor(indices, input_tensor, torch.Size([size, size]))

class Pre_GAT:
    def __init__(self, model, args, device):

        '''
        Compute structure and adjaceny iteratively.
        model: The backbone GNN model in ProGNN

        For Pre_GCN, args.gamma=0 args.lambda=0
        '''
        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.best_feature = None
        self.weights = None
        self.estimator = None
        self.estimator2 = None
        self.model = model.to(device)

    def fit(self, features, adj, labels, idx_train, idx_val):
        args = self.args
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if self.args.dataset == "pubmed":
            estimator = EstimateAdj(adj.to_dense(), symmetric=args.symmetric).to(self.device)
            estimator2 = EstimateFeature(features.to_dense()).to(self.device)
        else:
            estimator = EstimateAdj(adj, symmetric=args.symmetric).to(self.device)
            estimator2 = EstimateFeature(features).to(self.device)

        self.estimator2 = estimator2
        self.estimator = estimator
        self.optimizer_adj = optim.SGD(estimator.parameters(),
                                       momentum=0.9, lr=args.lr_adj)

        self.optimizer_feat = optim.SGD(estimator2.parameters(),
                                        momentum=0.9, lr=args.lr_adj)
        self.optimizer_l1 = PGD(estimator.parameters(),
                                proxs=[prox_operators.prox_l1],
                                lr=args.lr_adj, alphas=[args.alpha])

        if args.dataset == "pubmed":
            self.optimizer_nuclear = PGD(estimator.parameters(),
                                         proxs=[prox_operators.prox_nuclear_cuda],
                                         lr=args.lr_adj, alphas=[args.beta])
        else:
            self.optimizer_nuclear = PGD(estimator.parameters(),
                                         proxs=[prox_operators.prox_nuclear_cuda],
                                         lr=args.lr_adj, alphas=[args.beta])

        t_total = time.time()
        adj_list = []
        adj_list.append(estimator.estimated_adj.clone().detach().cpu().numpy())
        if args.dataset=='pubmed':
            features = features.to_dense()
            adj = adj.to_dense()
        for epoch in range(args.epochs):
            if args.only_gcn:
                self.train_gcn(epoch, features, estimator.estimated_adj, labels, idx_train, idx_val)
            else:
                for i in range(int(args.outer_steps)):
                    self.train_adj(epoch, features, adj, labels, idx_train, idx_val)
                    adj_list.append(estimator.estimated_adj.clone().detach().cpu().numpy())
                    torch.cuda.empty_cache()
                for i in range(int(args.outer_steps)):
                    self.train_feat(epoch, features, adj, labels, idx_train, idx_val)
                    torch.cuda.empty_cache()
                for i in range(int(args.inner_steps)):
                    self.train_gcn(epoch,
                                   labels, idx_train, idx_val)
                    torch.cuda.empty_cache()

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(args)

        # Testing
        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)
        return adj_list

    def train_adj(self, epoch, features, adj, labels, idx_train, idx_val):
        """

        :param epoch:
        :param features:
        :param adj:
        :param labels:
        :param idx_train:
        :param idx_val:
        :return:
        """
        estimator = self.estimator
        estimator2 = self.estimator2
        args = self.args
        if args.debug:
            print("\n=== train_adj ===")
        t = time.time()
        estimator.train()
        self.optimizer_adj.zero_grad()
        loss_fro = torch.norm(estimator.estimated_adj - adj, p='fro')
        g = self.generate_g(estimator.estimated_adj)
        if args.lambda_:
            loss_smooth_feat = self.feature_smoothing(estimator.estimated_adj, features)
        else:
            loss_smooth_feat = 0
        print(features.dtype)
        output = self.model(g, features)
        loss_fcn = torch.nn.CrossEntropyLoss()
        loss_gcn = loss_fcn(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_diffiential = loss_fro + args.gamma * loss_gcn + args.lambda_ * loss_smooth_feat
        loss_diffiential.backward()
        self.optimizer_adj.step()
 
        estimator.estimated_adj.data.copy_(torch.clamp(
            estimator.estimated_adj.data, min=0, max=1))

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        #以下两行要注释

        g = self.generate_g(estimator.estimated_adj)
        with torch.no_grad():
            output = self.model(g, features)
        loss_fcn = torch.nn.CrossEntropyLoss()
        loss_val = loss_fcn(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = g
            self.best_feature = estimator2.estimated_feature.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = g
            self.best_feature = estimator2.estimated_feature.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())



    def train_feat(self, epoch, features, adj, labels, idx_train, idx_val):

        args = self.args
        if args.debug:
            print("\n === This the train_feature===")
        t = time.time()
        self.estimator2.train()
        self.optimizer_feat.zero_grad()
        loss_fro = torch.norm(self.estimator2.estimated_feature - features, p='fro')
        g = self.generate_g(adj)
        output = self.model(g, self.estimator2.estimated_feature)
        loss_fcn = torch.nn.CrossEntropyLoss()
        loss_gcn = loss_fcn(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        if args.method == "smooth":
             loss_feat = self.feature_smoothing(adj, self.estimator2.estimated_feature)
        loss_diffiential = loss_fro + args.gamma * loss_gcn + args.lambda_ * loss_feat

        loss_diffiential.backward()
        self.optimizer_feat.step()

        total_loss = loss_fro \
                 + args.gamma * loss_gcn \
                  + args.lambda_ * loss_feat
        # estimator2.estimated_feature.data.copy(estimator2.estimated_feature.data)

        del output,adj
        self.model.eval()
        with torch.no_grad():
            output = self.model(g, self.estimator2.estimated_feature)

            loss_val = loss_fcn(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))
        torch.cuda.empty_cache()
        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = g
            self.best_feature = self.estimator2.estimated_feature.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = g
            self.best_feature = self.estimator2.estimated_feature.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())
        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_fro: {:.4f}'.format(loss_fro.item()),
                      'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                      'loss_feat: {:.4f}'.format(loss_feat.item()),
                      'loss_total: {:.4f}'.format(total_loss.item())
                      )

    def train_gcn(self, epoch, labels, idx_train, idx_val):
        args = self.args
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        g = self.generate_g(self.estimator.estimated_adj)
        output = self.model(g, self.estimator2.estimated_feature)
        loss_fcn = torch.nn.CrossEntropyLoss()
        loss_train = loss_fcn(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        self.optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        with torch.no_grad():
            output = self.model(g, self.estimator2.estimated_feature)

            loss_val = loss_fcn(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])


        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = g
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print('\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = g
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())
        del g,output
        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      'time: {:.4f}s'.format(time.time() - t))

        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.4f}'.format(loss_train.item()), 'acc_train: {:.4f}'.format(acc_train.item()), 'loss_val: {:.4f}'.format(loss_val.item()), 'acc_val: {:.4f}'.format(acc_val.item()),  'time: {:.4f}s'.format(time.time() - t))

    def test(self, features, labels, idx_test):
        print("\t=== testing ===")
        self.model.eval()
        g = self.best_graph
        features = self.best_feature
        args = self.args

        if self.best_graph is None:
            adj = self.estimator.estimated_adj
            features = self.estimator2.estimated_feature
            g = self.generate_g(adj)
        with torch.no_grad():
            output = self.model(g, features)
            loss_fcn = torch.nn.CrossEntropyLoss()
            loss_test = loss_fcn(output[idx_test], labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
        print("\tTest set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        logging.info("Accuracy："+str(acc_test.data))
    def feature_filter(self, adj, X):
        """
        Compute the loss of filter,the sum of rank 0 of \|U^TX\|_0
        """
        return X
    
    def feature_smoothing(self, adj, X):
        args = self.args
        adj = (adj.t() + adj) / 2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj
        r_inv = r_inv + 1e-3
        r_inv = r_inv.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        L = r_mat_inv @ L @ r_mat_inv
        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)

        return loss_smooth_feat
    


    def generate_g(self, estimated_adj):
        args = self.args
        if args.symmetric:
            adj = (estimated_adj+estimated_adj.t())/2
        else:
            adj = estimated_adj
        a = (adj.cpu() + torch.eye(adj.shape[0])).detach().cpu().numpy()
        b = sp.coo_matrix(a)

        g = dgl.from_scipy(b, 'weight').to(device)

        return g


class EstimateAdj(nn.Module):

    def __init__(self, adj, symmetric=False):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)
        self.symmetric = symmetric

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):

        if self.symmetric:
            adj = (self.estimated_adj + self.estimated_adj.t())
        else:
            adj = self.estimated_adj

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).cuda())
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx



class EstimateFeature(nn.Module):

    def __init__(self, feature):
        super(EstimateFeature, self).__init__()
        n, k = feature.size()
        self.estimated_feature = nn.Parameter(torch.FloatTensor(n, k))
        self._init_estimation(feature)

    def _init_estimation(self, feature):
        with torch.no_grad():
            n, k = feature.size()
            self.estimated_feature.data.copy_(feature)

    def forward(self):
        return self.estimated_feature

 

