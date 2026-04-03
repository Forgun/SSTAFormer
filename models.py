# -*- coding:utf-8 -*-
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from SMTA import *

print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats)
        self.conv2 = SAGEConv(h_feats, h_feats)
        self.conv3 = SAGEConv(h_feats, out_feats)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = self.conv3(x, edge_index)
        else:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = self.conv3(x, edge_index)
        return x


# ######################################################################


class SSTAFormer(nn.Module):
    def __init__(self, args):
        super(SSTAFormer, self).__init__()
        self.args = args
        self.out_feats =64#
        self.sage = GraphSAGE(in_feats=args.seq_len, h_feats=128, out_feats=self.out_feats)
        OutputChannelList = [64]
        n_heads = 4
        self.tci = SMTA(input_features_num=args.input_size, input_len=args.seq_len, output_len=128,
                          tcn_OutputChannelList=OutputChannelList, tcn_KernelSize=2,
                          tcn_Dropout=0.1, n_heads=n_heads)
        self.fcs = nn.ModuleList()
        for k in range(args.input_size):
            self.fcs.append(nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, args.output_size)
            ))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        batch_size = torch.max(batch).item() + 1
        x = self.sage(x, edge_index)
        batch_list = batch.cpu().numpy()
        xs = [[] for k in range(batch_size)]
        ys = [[] for k in range(batch_size)]
        for k in range(x.shape[0]):
            xs[batch_list[k]].append(x[k, :])
            ys[batch_list[k]].append(data.y[k, :])
        xs = [torch.stack(x, dim=0) for x in xs]
        ys = [torch.stack(x, dim=0) for x in ys]
        x = torch.stack(xs, dim=0)
        y = torch.stack(ys, dim=0)
        x = x.permute(0, 2, 1)
        x = self.tci(x)
        preds = []
        for fc in self.fcs:
            preds.append(fc(x))
        pred = torch.stack(preds, dim=0)
        return pred, y

# ######################################################################
