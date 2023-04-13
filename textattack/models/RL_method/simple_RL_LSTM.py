import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)


class SimpleRLLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_size, num_hidden_layers=1) -> None:
        super(SimpleRLLSTM, self).__init__()
        # NOTE: skip the embedding here since we're already passing in gloVe embeddings
        self.embedding_dim = embedding_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, num_layers = num_hidden_layers)

        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, input):
        lstm_out, _ = self.lstm(input.view(len(input), 1, -1))
        out = self.fc(lstm_out.view(len(input), -1))
        return out