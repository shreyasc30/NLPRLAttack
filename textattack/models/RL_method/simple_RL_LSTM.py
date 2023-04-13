import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)


class SimpleRLLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_size, num_hidden_layers=1, sequence_length=50) -> None:
        super(SimpleRLLSTM, self).__init__()
        # NOTE: skip the embedding here since we're already passing in gloVe embeddings
        self.embedding_dim = embedding_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, num_layers = num_hidden_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim * sequence_length, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # Each datapoint in batch is flattened (dims corresponding to sequence and embedding)
        out = self.fc(torch.reshape(lstm_out, (x.size()[0], -1)))
        return out