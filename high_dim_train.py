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
parser.add_argument("-e", "--epochs", type=int, default=50000)
parser.add_argument("-lr", "--learningrate", type=float, default=3e-3)
parser.add_argument("-i", "--indim", type=int, default=10)
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

exp_name = "HighDim_SiLU"
path = os.path.join("./results/", exp_name)
os.makedirs(path, exist_ok=True)


def u(x):
    ans = 0
    for i in range(5):
        u += x[:, 2 * i] * x[:, 2 * i + 1]
    return ans


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Device: {device}")
    seed_everything(SEED)
    print(f"Random Seed: {SEED}")
    model = FullConnected_DNN(
        in_dim=IN_DIM,
        out_dim=OUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blks=NUM_HIDDENS,
        skip=SKIP,
        act=nn.SiLU(),
    ).to(device)
    losses = []
    losses_r = []
    losses_b = []
    model.apply(initialize_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=500, T_mult=2, last_epoch=-1
    )
    best_loss, best_b, best_r, best_epoch = 0x3F3F3F, 0x3F3F3F, 0x3F3F3F, 0
    bar = tqdm(range(EPOCHS))
    model.train()
    t0 = time.time()
    for epoch in bar:
        bar.set_description("Training Epoch " + str(epoch))
        # generate the data set
        xr = get_interior_points(N=1000, d=10)
        xb = get_boundary_points_high(N=100)

        xr = xr.to(device)
        xb = xb.to(device)

        xr.requires_grad_()
        output_r = model(xr)
        output_b = model(xb)
        # print(output_r.shape, output_b.shape)
        grads = autograd.grad(
            outputs=output_r,
            inputs=xr,
            grad_outputs=torch.ones_like(output_r),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grads_sum = torch.sum(torch.pow(grads, 2), dim=1)
        u1 = torch.mean(0.5 * grads_sum)
        u2 = torch.mean(torch.pow(output_b - u(xb), 2))
        # loss = 4 * loss_r + 9 * 500 * loss_b
        # loss = loss_r + 4 * 500 * loss_b
        loss = u1 + 20 * 500 * u2
        bar.set_postfix(
            {"Loss": "{:.4f}".format(abs(loss)),}
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_t = loss.detach().numpy()
        losses.append(abs(loss_t))
        if epoch > int(4 * EPOCHS / 5):
            if torch.abs(loss) < best_loss:
                best_loss = torch.abs(loss).item()
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(path, "ckpt.bin"))

    print(
        "Best epoch:", best_epoch, "Best loss:", best_loss,
    )
    elapse = time.time() - t0
    print(f"Training time: {elapse}")
    print(f"# of parameters: {get_param_num(model)}")
    # plot figure
    model.eval()
    model.load_state_dict(torch.load(os.path.join(path, "ckpt.bin")))
    print("Load weights from checkpoint!")
    with torch.no_grad():
        x = torch.rand(100000, 10)
        u_exact = u(x)
        x = x.to(device)
        u_pred = model(x)
    err_l2 = torch.sqrt(torch.mean(torch.pow(u_pred - u_exact, 2))) / torch.sqrt(
        torch.mean(torch.pow(u_exact, 2))
    )
    print("L^2 relative error:", err_l2)

    plot_loss_and_save_high(EPOCHS=EPOCHS, SAMPLE=SAMPLE, losses=losses, path=path)
    print("Output figure saved!")
