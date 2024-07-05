import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool


class GCNNet(nn.Module):
    def __init__(self, input_dim, model_args, device, output_dim = 2):
        super(GCNNet, self).__init__()
        self.output_dim = output_dim
        self.latent_dim = model_args.latent_dim
        self.mlp_hidden = model_args.mlp_hidden
        self.emb_normlize = model_args.emb_normlize
        self.device = device
        self.num_gnn_layers = len(self.latent_dim)
        self.readout_layers = self.get_readout_layers(model_args.readout)

        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GCNConv(input_dim, self.latent_dim[0]))
        for i in range(1, self.num_gnn_layers):
            self.gnn_layers.append(
                GCNConv(self.latent_dim[i - 1], self.latent_dim[i]))
        self.gnn_non_linear = nn.ReLU()


    def get_readout_layers(self, readout):
        readout_func_dict = {
            "mean": global_mean_pool,
            "sum": global_add_pool,
            "max": global_max_pool
        }
        readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
        ret_readout = []
        for k, v in readout_func_dict.items():
            if k in readout.lower():
                ret_readout.append(v)
        return ret_readout


    def forward(self, x, edge_index, batch, edge_weight = None):

        for i in range(self.num_gnn_layers):
            if edge_weight == None:
                x = self.gnn_layers[i](x, edge_index)
            else:
                x = self.gnn_layers[i](x, edge_index, edge_weight)
            if self.emb_normlize:
                x = F.normalize(x, p=2, dim=-1)
            x = self.gnn_non_linear(x)

        node_emb = x
        pooled = []
        batch = batch.reshape(-1)
        for readout in self.readout_layers:
            pooled.append(readout(x, batch))
        graph_emb = torch.cat(pooled, dim=-1)


        return graph_emb, node_emb
