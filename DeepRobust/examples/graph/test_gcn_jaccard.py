import torch
import numpy as np
import torch.nn.functional as F
from deeprobust.graph.defense import GCNJaccard
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import PtbDataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# make sure you use the same data splits as you generated attacks
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# load original dataset (to get clean features and labels)
data = Dataset(root='/tmp/', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

# load pre-attacked graph
perturbed_data = PtbDataset(root='/tmp/', name=args.dataset)
perturbed_adj = perturbed_data.adj


# Setup Defense Model
model = GCNJaccard(nfeat=features.shape[1], nclass=labels.max()+1,
                nhid=16, device=device)

model = model.to(device)

print('=== testing GCN-Jaccard on perturbed graph ===')
model.fit(features, perturbed_adj, labels, idx_train, idx_val, threshold=0.01)
model.eval()
output = model.test(idx_test)

