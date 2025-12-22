# model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from torch_geometric.nn import SAGEConv
class CompositeGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=15):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin_out = torch.nn.Linear(hidden_channels, out_channels)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in [self.conv1, self.conv2, self.conv3]:
            x = torch.relu(conv(x, edge_index))
        return self.lin_out(x)


#class CompositeGNN(torch.nn.Module):
#    def __init__(self, in_channels, hidden_channels, out_channels=15):
#        super().__init__()
#        self.conv1 = GCNConv(in_channels, hidden_channels)
#        self.conv2 = GCNConv(hidden_channels, hidden_channels)
#        self.conv3 = GCNConv(hidden_channels, hidden_channels)
#        self.lin_out = torch.nn.Linear(hidden_channels, out_channels)
#
#    def forward(self, data):
#        x, edge_index, batch = data.x, data.edge_index, data.batch
#        
#        x = self.conv1(x, edge_index)
#        x = F.relu(x)
#        x = self.conv2(x, edge_index)
#        x = F.relu(x)
#        x = self.conv3(x, edge_index)
#        
#        # Predict
#        out = self.lin_out(x)
#        return out
