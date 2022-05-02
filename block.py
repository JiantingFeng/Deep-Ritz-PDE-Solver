import torch
from torch import nn
import torch.nn.functional as F
import activation


class PowerReLU(nn.Module):
    """
    Implements simga(x)^(power)
    Applies a power of the rectified linear unit element-wise.
    NOTE: inplace may not be working.
    Can set inplace for inplace operation if desired.
    BUT I don't think it is working now.
    INPUT:
        x -- size (N,*) tensor where * is any number of additional
             dimensions
    OUTPUT:
        y -- size (N,*)
    """

    def __init__(self, inplace=False, power=3):
        super(PowerReLU, self).__init__()
        self.inplace = inplace
        self.power = power

    def forward(self, input):
        y = F.relu(input, inplace=self.inplace)
        return torch.pow(y, self.power)


class ResBlock(nn.Module):
    def __init__(self, io_dim, hidden_dim, skip=True, act=activation.ReLU_k()) -> None:
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(io_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, io_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.bn2 = nn.BatchNorm1d(io_dim)
        self.act = act
        self.skip = skip

    def forward(self, x):
        # out = self.dropout(self.act(self.bn1(self.fc1(x))))
        # out = self.dropout(self.act(self.bn2(self.fc2(out))))
        # out = self.bn1(self.act(self.fc1(x)))
        # out = self.bn2(self.act(self.fc2(out)))
        out = self.act(self.fc1(x))
        out = self.act(self.fc2(out))
        if self.skip:
            return out + x
        return out


if __name__ == "__main__":
    model = ResBlock(2, 5, act=activation.Leaky_ReLU_k())
    x = torch.zeros(size=(2, 2))
    out = model(x)
    print(out.shape)
    print(out)
