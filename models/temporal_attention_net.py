import torch
import torch.nn as nn
import torch.nn.functional as F


class Temporal_Attention_Net(nn.Module):
    def __init__(self, Tem_attNet_input_size, lstm_hidden_size, lstm_num_layers, device, model_args):
        super(Temporal_Attention_Net, self).__init__()
        self.lstm_num_layers = lstm_num_layers
        self.device = device

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(input_size=Tem_attNet_input_size, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(lstm_hidden_size, Tem_attNet_input_size)
        self.fc2 = nn.Linear(Tem_attNet_input_size, 1)


        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_size, 16),
            nn.Linear(16,2)
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(model_args.dropout)


    def sampling(self, sampling_weights, temperature=1.0, bias=0.0):

        bias = bias + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1-bias)) * torch.rand(sampling_weights.size()) + (1-bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs= gate_inputs.to(self.device)
        gate_inputs = (gate_inputs + sampling_weights) / temperature
        out = torch.sigmoid(gate_inputs)

        return out


    def forward(self, lstm_input, oneperson_batch_list):


        lstm_input_list = []
        for i in range(len(oneperson_batch_list)):
            start = sum(oneperson_batch_list[0:i])
            end = start + oneperson_batch_list[i]
            lstm_input_list.append(lstm_input[start:end , :])

        lstm_input_list = torch.nn.utils.rnn.pad_sequence(lstm_input_list, batch_first=True)
        lstm_input_list = torch.nn.utils.rnn.pack_padded_sequence(lstm_input_list, lengths = oneperson_batch_list.cpu(), enforce_sorted = False, batch_first = True)
        lstm_output, hidden = self.lstm(lstm_input_list)
        lstm_output, lstm_output_lens = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        lstm_final_output = torch.empty(self.lstm_hidden_size).to(self.device)
        for i in range(len(lstm_output_lens)):
            k = lstm_output_lens[i]
            tep_padded_output = lstm_output[i,0:k,:]
            lstm_final_output = torch.vstack((lstm_final_output, tep_padded_output))
        lstm_final_output = lstm_final_output[1:,:]
        
        passed_Z = self.fc1(lstm_final_output)

        tem_att_matrix = self.fc2(passed_Z).squeeze()
        tem_att_matrix = self.sampling(tem_att_matrix, temperature=1).reshape(-1,1)

        tep_att_representation = torch.mul(lstm_final_output, tem_att_matrix)
        final_representation = torch.empty(0).to(self.device)


        for i in range(len(oneperson_batch_list)):
            start = sum(oneperson_batch_list[0:i])
            end = start + oneperson_batch_list[i]

            tep = tep_att_representation[start:end , :].sum(dim=0)
            tep = self.dropout(tep)
            tep = self.mlp(tep)
            tep = tep.unsqueeze(0)

            final_representation = torch.cat((final_representation, tep), dim = 0)  


        return final_representation, passed_Z
