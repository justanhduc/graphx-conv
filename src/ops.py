import torch as T
import torch.nn as nn


class GraphXConv(nn.Module):
    def __init__(self, in_features, out_features, in_instances, out_instances=None, bias=True, activation=None):
        super().__init__()
        self.activation = activation
        self.conv_l = nn.Linear(in_instances, out_instances if out_instances else in_instances, bias=bias)
        self.conv_r = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        assert len(input.shape) == 3, 'Input dimension must be (b, n, d)'

        output = self.conv_l(input.transpose(2, 1)).transpose(2, 1)
        output = self.conv_r(output)
        return output if self.activation is None else self.activation(output)
