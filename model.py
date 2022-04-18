import torch
from torchsummary import summary
from torch import nn
import torch.nn.functional as F
from activation import *
from block import *


class FullConnected_DNN(nn.Module):
    def __init__(
            self,
            in_dim=2,
            out_dim=1,
            hidden_dim=10,
            num_blks=5,
            skip=True,
            act=ReLU_k(),
    ) -> None:
        super(FullConnected_DNN, self).__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        blks = [
            ResBlock(
                io_dim=hidden_dim,
                hidden_dim=hidden_dim,
                skip=skip,
                act=act,
            ) for _ in range(num_blks)
        ]
        self.backbone = nn.Sequential(*blks)
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.backbone(x)
        x = self.fc_out(x)

        return x


def initialize_weights(m):
    if isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)

    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    model = FullConnected_DNN(hidden_dim=10, num_blks=3)
    model.apply(initialize_weights)
    # x = torch.zeros(size=(2, 2))
    # out = model(x)
    # print(out.shape)
    # print(out)
    # print(model)

    summary(model, (2, ))
