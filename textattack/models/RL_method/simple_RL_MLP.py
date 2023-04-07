import torch
import torch.nn as nn
import torch.nn.functional as F

# me: tokenization and padding with wrapper
class SimpleRLMLP(nn.Module):

    # input size is all tokens length: const. w/pad
    # output size is (# replaceable tokens): const for all samples
    def __init__(self, input_size, output_size) -> None:
        super(SimpleRLMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size * 2)
        self.fc2 = nn.Linear(input_size * 2, input_size * 2)
        self.fc3 = nn.Linear(input_size * 2, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out