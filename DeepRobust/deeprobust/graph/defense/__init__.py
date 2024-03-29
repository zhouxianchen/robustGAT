from .gcn import GCN
from .gcn_preprocess import GCNSVD, GCNJaccard
from .r_gcn import RGCN
from .prognn import ProGNN
from .gat import GAT, generate_data
from .Ro_GAT import Ori_GAT
from .MyGATConv import GATConv
from .MyRo_GAT import Pre_GAT
__all__ = ['GCN', 'GCNSVD', 'GCNJaccard', 'RGCN', 'ProGNN',"GAT", "generate_data", "Ori_GAT", "GATConv", "Pre_GAT"]
