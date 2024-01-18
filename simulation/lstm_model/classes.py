import torch
import torch.nn as nn

from simulation.lstm_model.config import config


class AttributeDict(dict):
    """This class allows the keys of in a dictionary to be retrieved as if they were attributes.
    dict.key syntax instead of dict['key']"""
    __slots__ = ()
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

config = AttributeDict(config)

# DEFINE LSTM MODEL CLASS
class LSTM_model_wandb(nn.Module):
    def __init__(self, n_features, n_output, seq_len):
        super(LSTM_model_wandb, self).__init__()

        self.n_features = n_features
        self.n_output = n_output  # number of output
        self.seq_len = seq_len  # lookback value
        self.n_hidden = config.hidden_size  # number of hidden states
        self.n_layers = config.num_layer  # number of LSTM layers (stacked)

        self.l_lstm = nn.LSTM(input_size=self.n_features,
                              hidden_size=self.n_hidden,
                              num_layers=self.n_layers,
                              batch_first=True)
        self.dropout = torch.nn.Dropout(config.dropout)
        # LSTM Outputs: output, (h_n, c_n)
        # according to pytorch docs LSTM output is
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = nn.Linear(self.n_hidden, self.n_output)

    def init_hidden(self, batch_size, device):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        hidden_state = hidden_state.to(device)
        cell_state = cell_state.to(device)
        hidden = (hidden_state, cell_state)  # HIDDEN is defined as a TUPLE
        return hidden

    def forward(self, input_tensor, hidden_cell_tuple):
        batch_size, seq_len, _ = input_tensor.size()
        lstm_out, hidden_cell_tuple = self.l_lstm(input_tensor, hidden_cell_tuple)
        lstm_out = self.dropout(lstm_out)  # Applying dropout
        # out_numpy = lstm_out.detach().numpy()
        out = lstm_out[:, -1, :]  # many to one, I take only the last output vector, for each Batch
        out_linear = self.l_linear(out)
        return out_linear, hidden_cell_tuple



# LSTM FOR OPTUNA
class LSTM_model_optuna(nn.Module):
    def __init__(self, n_features, n_output, drop_prob, seq_len, num_hidden, num_layers):
        super(LSTM_model_optuna, self).__init__()

        self.n_features = n_features  # number of inputs variable
        self.n_output = n_output  # number of output
        self.seq_len = seq_len  # lookback value
        self.n_hidden = num_hidden  # number of hidden states
        self.n_layers = num_layers  # number of LSTM layers (stacked)

        self.l_lstm = nn.LSTM(input_size=self.n_features,
                              hidden_size=self.n_hidden,
                              num_layers=self.n_layers,
                              batch_first=True)
        self.dropout = torch.nn.Dropout(drop_prob)
        # LSTM Outputs: output, (h_n, c_n)
        # according to pytorch docs LSTM output is
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = nn.Linear(self.n_hidden, n_output)

    def init_hidden(self, batch_size, device):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        hidden_state = hidden_state.to(device)
        cell_state = cell_state.to(device)
        hidden = (hidden_state, cell_state)  # HIDDEN is defined as a TUPLE
        return hidden

    def forward(self, input_tensor, hidden_cell_tuple):
        batch_size, seq_len, _ = input_tensor.size()
        lstm_out, hidden_cell_tuple = self.l_lstm(input_tensor, hidden_cell_tuple)
        lstm_out = self.dropout(lstm_out)  # Applying dropout
        # out_numpy = lstm_out.detach().numpy()
        out = lstm_out[:, -1, :]  # I take only the last output vector, for each Batch
        out_linear = self.l_linear(out)
        return out_linear, hidden_cell_tuple


import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attn_weights = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        attn_scores = nn.functional.softmax(self.attn_weights(lstm_out), dim=1)
        attn_output = torch.sum(attn_scores * lstm_out, dim=1)
        return attn_output

# DEFINE LSTM MODEL CLASS
class LSTM_attention(nn.Module):
    def __init__(self, n_features, n_output, drop_prob, seq_len, num_hidden, num_layers):
        super(LSTM_attention, self).__init__()

        self.n_features = n_features  # number of inputs variable
        self.n_output = n_output  # number of output
        self.seq_len = seq_len  # lookback value
        self.n_hidden = num_hidden  # number of hidden states
        self.n_layers = num_layers  # number of LSTM layers (stacked)

        self.l_lstm = nn.LSTM(input_size=self.n_features,
                              hidden_size=self.n_hidden,
                              num_layers=self.n_layers,
                              batch_first=True)
        self.dropout = torch.nn.Dropout(drop_prob)
        # LSTM Outputs: output, (h_n, c_n)
        # according to pytorch docs LSTM output is
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = nn.Linear(self.n_hidden, n_output)
        self.attention_layer = AttentionLayer(self.n_hidden)


    def init_hidden(self, batch_size, device):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        hidden_state = hidden_state.to(device)
        cell_state = cell_state.to(device)
        hidden = (hidden_state, cell_state)  # HIDDEN is defined as a TUPLE
        return hidden

    def forward(self, input_tensor, hidden_cell_tuple):
        batch_size, seq_len, _ = input_tensor.size()
        lstm_out, hidden_cell_tuple = self.l_lstm(input_tensor, hidden_cell_tuple)
        lstm_out = self.dropout(lstm_out)

        # Apply attention
        attn_output = self.attention_layer(lstm_out)

        # out = attn_output[:, -1, :]  # I take only the last output vector, for each Batch

        out_linear = self.l_linear(attn_output)

        return out_linear, hidden_cell_tuple
