#! /opt/conda/bin/python3.8
import time
import os
import sys
import argparse

sys.path.append(os.path.dirname(sys.path[0]))
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.abspath(os.path.dirname(curPath)+os.path.sep+".")
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)


import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
import networkx as nx
# import wandb
import scipy.sparse as sp
import logging
from DeepRobust.deeprobust.graph.defense import GCN, ProGNN, Pre_GCN,GAT,generate_data, Ori_GAT, Pre_GAT
from DeepRobust.deeprobust.graph.data import Dataset, PrePtbDataset
from DeepRobust.deeprobust.graph.utils import preprocess

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT)

from dgl.data import register_data_args, load_data
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--only_gcn', action='store_true',
        default=False, help='test the performance of gcn without other components')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units. 16')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
        choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--attack', type=str, default='meta',
        choices=['no', 'meta', 'random', 'nettack'])
parser.add_argument('--ptb_rate', type=float, default=0, help="noise ptb_rate")
parser.add_argument('--epochs', type=int,  default=10, help='Number of epochs to train.')
parser.add_argument('--alpha', type=float, default=1, help='weight of l1 norm 5')
parser.add_argument('--beta', type=float, default=0, help='weight of nuclear norm  1.5')
parser.add_argument('--gamma', type=float, default=1, help='weight of l2 norm   1')
# parser.add_argument('--gamma', type=float, default=1, help='weight of l2 norm')
parser.add_argument('--lambda_', type=float, default=1, help='weight of feature smoothing  1')
parser.add_argument('--phi', type=float, default=0, help='weight of symmetric loss')
parser.add_argument('--inner_steps', type=int, default=10, help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=3, help='steps for outer optimization')
parser.add_argument('--lr_adj', type=float, default=0.01, help='lr for training adj')
parser.add_argument('--method', type=str, default="smooth", help= "THe revised method for feature",
                   choices=['smooth','filter'])
parser.add_argument('--symmetric', action='store_true', default=False,
            help='whether use symmetric matrix')



# parser = argparse.ArgumentParser(description='GAT')
parser.add_argument("--gpu", type=int, default=0,
                    help="which GPU to use. Set -1 to use CPU.")
parser.add_argument("--num-heads", type=int, default=4,
                    help="number of hidden attention heads default 8")
parser.add_argument("--num-out-heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--num-layers", type=int, default=1,
                    help="number of hidden layers")
parser.add_argument("--num-hidden", type=int, default=4,
                    help="number of hidden units 8")
parser.add_argument("--residual", action="store_true", default=False,
                    help="use residual connection")
parser.add_argument("--in-drop", type=float, default=.6,
                    help="input feature dropout")
parser.add_argument("--attn-drop", type=float, default=.6,
                    help="attention dropout")
parser.add_argument('--weight-decay', type=float, default=5e-4,
                    help="weight decay")
parser.add_argument('--negative-slope', type=float, default=0.2,
                    help="the negative slope of leaky relu")
parser.add_argument('--early-stop', action='store_true', default=False,
                    help="indicates whether to use early stop or not")
parser.add_argument('--fastmode', action="store_true", default=False,
                    help="skip re-evaluate the validation set")
args = parser.parse_args()
# wandb.init(project="RoGAT-project", name="Final_result_multi_"+args.attack, config=args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if args.cuda:
    torch.cuda.manual_seed(args.seed)
if args.ptb_rate == 0:
    args.attack = "no"


np.random.seed(15) # Here the random seed is to split the train/val/test data, we need to set the random seed to be the same as that when you generate the perturbed graph


def generate_gat_data(args, data):

    adj, features, labels = data.adj, data.features, data.labels
    features = torch.FloatTensor(data.features.todense())
    labels = torch.LongTensor(data.labels)
    train_mask = data.idx_train
    val_mask = data.idx_val
    test_mask = data.idx_test
    num_feats = features.shape[1]
    n_classes = labels.max().item() + 1
    print("""----Data statistics------'
          #Classes %d
          #Train samples %d
          #Val samples %d
          #Test samples %d""" %
          (n_classes,
           len(train_mask),
           len(val_mask),
           len(test_mask)))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    return adj, num_feats, n_classes, heads, cuda, features, labels, train_mask, val_mask, test_mask


def mask_to_tensor_mask(mask,total_length):
    tt = torch.BoolTensor(total_length)
    tt[mask]=1
    return 1

if __name__ == "__main__":

    print(args)
    data = Dataset(root='../dataset/', name=args.dataset, setting='nettack')
    adj, num_feats, n_classes, heads, cuda, features, labels, train_mask, val_mask, test_mask = generate_gat_data(args, data)

    if args.attack == 'no':
        perturbed_adj = adj
    if args.attack == 'random':
        from DeepRobust.deeprobust.graph.global_attack import Random

        attacker = Random()
        n_perturbations = int(args.ptb_rate * (adj.sum() // 2))
        perturbed_adj = attacker.attack(adj, n_perturbations, type='add')

    if args.attack == 'meta' or args.attack == 'nettack':
        perturbed_data = PrePtbDataset(root='../dataset/',
                                       name=args.dataset,
                                       attack_method=args.attack,
                                       ptb_rate=args.ptb_rate)
        perturbed_adj = perturbed_data.adj

    g = nx.from_scipy_sparse_matrix(perturbed_adj, create_using=nx.DiGraph())
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = dgl.from_networkx(g)
    g.add_edges(g.nodes(), g.nodes())
    b = sp.coo_matrix(g.adjacency_matrix().to_dense().numpy())
    g = dgl.from_scipy(b, 'weight').to(device)
    torch.cuda.empty_cache()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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


features = data.features

if args.dataset == "pubmed":
    perturbed_adj, features, labels = preprocess(perturbed_adj, features, labels, preprocess_adj=False, device=device,sparse=True)
else:
    perturbed_adj, features, labels = preprocess(perturbed_adj, features, labels, preprocess_adj=False, device=device,
                                                 sparse=False)
del data

pregat = Pre_GAT(model, args, device)
pregat.fit(features, perturbed_adj, labels, train_mask, val_mask)

import json


def load_json(file):
    with open(file, 'r') as load_f:
        load_dict = json.load(load_f)
    return load_dict["attacked_test_nodes"]

if args.attack == "nettack":
    test_mask = load_json("../dataset/" + args.dataset+"_nettacked_nodes.json")

logging.info("dataset:"+args.dataset+" attack:"+args.attack+" ptb_rate:"+str(args.ptb_rate)+" lambda_:"+str(args.lambda_)+" gamma:"+str(args.gamma))
pregat.test(features,labels, test_mask)
