from cProfile import label
from genericpath import exists
from math import ceil
import os
import random
import numpy as np
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, autograd
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from utils import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=30000)
parser.add_argument("-lr", "--learningrate", type=float, default=3e-3)
parser.add_argument("-i", "--indim", type=int, default=2)
parser.add_argument("-o", "--outdim", type=int, default=1)
parser.add_argument("-hd", "--hiddim", type=int, default=10)
parser.add_argument("-hn", "--hidnum", type=int, default=5)
parser.add_argument("-p", "--pretrained", type=bool, default=False)
parser.add_argument("-s", "--seed", type=int, default=2022)
parser.add_argument("-sp", "--sample", type=int, default=100)
parser.add_argument("-skip", "--skip", type=bool, default=True)

args = parser.parse_args()

EPOCHS = args.epochs
LR = args.learningrate
IN_DIM = args.indim
OUT_DIM = args.outdim
HIDDEN_DIM = args.hiddim
NUM_HIDDENS = args.hidnum
PRETRAINED = args.pretrained
SEED = args.seed
SAMPLE = args.sample
SKIP = args.skip

sns.set_style("white")


def u_real(x):
    return (1 - x[:, 0] ** 2) * (1 - x[:, 1] ** 2)


x1 = torch.linspace(-1, 1, 1001)
x2 = torch.linspace(-1, 1, 1001)
X, Y = torch.meshgrid(x1, x2)
Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)
real = u_real(Z).reshape(1001, 1001)
plt.figure()
fig, ax = plt.subplots()
h = ax.imshow(
    real,
    Interpolation="nearest",
    cmap="rainbow",
    extent=[-1, 1, -1, 1],
    origin="lower",
    aspect="auto",
)
# ax.set_title("Deep Neural Network Solution")
ax.set_xlabel(r"x")
ax.set_ylabel(r"y")
plt.colorbar(h)
plt.savefig("real.pdf")

