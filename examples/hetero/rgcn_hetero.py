import argparse
import os.path as osp

import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import HeteroConv, GCNConv, FastRGCNConv, to_hetero
from torch_geometric.data import HeteroData
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    choices=['AIFB', 'MUTAG', 'BGS', 'AM'])
args = parser.parse_args()

# # Trade memory consumption for faster computation.
# if args.dataset in ['AIFB', 'MUTAG']:
#     RGCNConv = FastRGCNConv

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Entities')
dataset = Entities(path, args.dataset)
data = dataset[0]

print(data)

data['v'].x = torch.ones(data.num_nodes, 1)

# BGS and AM graphs are too big to process them in a full-batch fashion.
# Since our model does only make use of a rather small receptive field, we
# filter the graph to only contain the nodes that are at most 2-hop neighbors
# away from any training/test node.

data.train_y_dict = data.train_y_dict['v']

class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(data.num_nodes, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu') if args.dataset == 'AM' else device
model = GNN()
model = to_hetero(model, data.metadata(), aggr='sum').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

# y = torch.ones(data.test_idx.shape[0], dtype=torch.long).to(device)

dur = []
def train():
    t0 = time.time()
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    # loss = F.cross_entropy(out['node_type1'][:data.test_idx.shape[0]], y)
    # loss = F.nll_loss(out['node_type1'], data.train_y_dict)
    # loss.backward()
    # optimizer.step()
    dur.append(time.time()-t0)
    return #loss.item()


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.edge_index, data.edge_type).argmax(dim=-1)
    train_acc = pred[data.train_idx].eq(data.train_y).to(torch.float).mean()
    test_acc = pred[data.test_idx].eq(data.test_y).to(torch.float).mean()
    return train_acc.item(), test_acc.item()


for epoch in range(1, 51):
    # print("Epoch", epoch)
    train()
    print(f'Epoch: {epoch:02d}:  Time: {np.average(dur):.4f} ')
            # f'Time: {np.average(dur):.4f}')
    # train_acc, test_acc = test()
    # print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f} '
    #       f'Test: {test_acc:.4f}')
