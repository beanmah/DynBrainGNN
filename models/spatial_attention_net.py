import torch
import torch.nn as nn


class Spatial_Attention_Net(torch.nn.Module):

    def __init__(self, model, model_args, device):
        super(Spatial_Attention_Net, self).__init__()

        self.model = model
        self.device = device
        self.reg_coefs = (0.001, 0.001)
        self.expl_embedding = model_args.latent_dim[-1]*2
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.mlp = nn.Sequential(
            nn.Linear(model_args.latent_dim[-1], 16),
            nn.Linear(16,2)
        )


    def _create_explainer_input(self, pair, embeds):
        
        rows = pair[0]
        cols = pair[1]
        row_embeds = embeds[rows]
        col_embeds = embeds[cols]

        input_expl = torch.cat([row_embeds, col_embeds], 1)
        return input_expl


    def sampling(self, sampling_weights, temperature=1.0, bias=0.0):

        bias = bias + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1-bias)) * torch.rand(sampling_weights.size()) + (1-bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs= gate_inputs.to(self.device)
        gate_inputs = (gate_inputs + sampling_weights) / temperature
        out = torch.sigmoid(gate_inputs)

        return out


    def forward(self, node_feature, edge_index, batch):

        graph_emb, node_emb = self.model(node_feature, edge_index, batch)
        input_expl = self._create_explainer_input(edge_index, node_emb).unsqueeze(dim=0)
        sampling_weights = self.explainer_model(input_expl).squeeze()
        edge_mask = self.sampling(sampling_weights, temperature = 1).squeeze().reshape(-1,1)
        graph_emb, node_emb = self.model(node_feature, edge_index, batch, edge_weight = edge_mask)

        return graph_emb
