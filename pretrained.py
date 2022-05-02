#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, autograd

from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from utils import *
from model import *


class CONFIG:
    PATH = "./results/GELU/ckpt.bin"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SEED = 42


sns.set_style("whitegrid")

if __name__ == "__main__":
    print(f"Training Device: {CONFIG.DEVICE}")
    seed_everything(CONFIG.SEED)
    model = FullConnected_DNN(
        in_dim=2,
        out_dim=1,
        hidden_dim=10,
        num_blks=5,
        skip=True,
        act=nn.GELU(),
    ).to(CONFIG.DEVICE)

    model.eval()
    model.load_state_dict(torch.load(CONFIG.PATH))
    t0 = time.time()
    print(f"Model loaded in {time.time() - t0:.2f} seconds")
    print("Load model successfully!")

    with torch.no_grad():
        x1 = torch.linspace(-1, 1, 1001)
        x2 = torch.linspace(-1, 1, 1001)
        X, Y = torch.meshgrid(x1, x2)
        Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)
        Z = Z.to(CONFIG.DEVICE)
        pred = model(Z)

    pred = pred.cpu().numpy()
    pred = pred.reshape(1001, 1001)
    print(f"Prediction finished in {time.time() - t0:.2f} seconds!")
    plot_result_and_save(pred, "./results/GELU/")

    print(f"Result saved in ./results/GELU/")
