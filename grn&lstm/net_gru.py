from torch import nn
import torch


class gruRNN(nn.Module):

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=2):
        super().__init__()

        # (input_size, output_size, layer)
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.mlp = nn.Linear(hidden_size, output_size)

    def forward(self, _x):

        out, _ = self.gru(_x)  # (seq_len, batch, input_size)
        out = self.mlp(out)
        return out


if __name__ == '__main__':

    # (input_size, hidden_size, output_size, num_layers)
    net1 = gruRNN(1, 16, 1, 5)
    x = torch.randn(30, 5, 1)  # (seq_len, batch, input_size)
    y = net1(x)
    print(x.shape)
    print(y.shape)
