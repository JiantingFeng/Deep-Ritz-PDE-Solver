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

exp_name = "ReLU_3_CLIP_1"
path = os.path.join("./results/", exp_name)
os.makedirs(path, exist_ok=True)


def u_real(x):
    norm = torch.norm(x, dim=1).unsqueeze(1)
    theta = torch.atan2(x[:, 1], x[:, 0]).unsqueeze(1)
    sin = torch.sin(theta / 2)
    return torch.einsum("bi,bi->b", norm, sin).unsqueeze(1)


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
        act=PowerReLU(),
    ).to(device)
    losses = []
    losses_r = []
    losses_b = []
    model.apply(initialize_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=500, T_mult=2, last_epoch=-1
    )
    best_loss, best_b, best_r, best_epoch = 1e4, 1e4, 1e4, 0
    bar = tqdm(range(EPOCHS))
    model.train()
    t0 = time.time()
    for epoch in bar:
        bar.set_description("Training Epoch " + str(epoch))
        # generate the data set
        xr = get_interior_points()
        xb = get_boundary_points()

        xr = xr.to(device)
        xb = xb.to(device)

        xr.requires_grad_()
        output_r = model(xr)
        output_b = model(xb)
        # print(output_r.shape, output_b.shape)
        print(f"output_r:{output_r}" "output_b:{output_b}")
        grads = autograd.grad(
            outputs=output_r,
            inputs=xr,
            grad_outputs=torch.ones_like(output_r),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        # print(torch.sum(torch.pow(grads, 2), dim=1).shape, output_r.shape)
        loss_r = 0.5 * torch.sum(torch.pow(grads, 2), dim=1) - output_r
        # print(loss_r, loss_b)
        loss_r = torch.mean(loss_r)
        loss_b = torch.mean(torch.pow(output_b, 2))
        # loss = 4 * loss_r + 9 * 500 * loss_b
        # loss = loss_r + 4 * 500 * loss_b
        loss = loss_r + 500 * loss_b
        bar.set_postfix(
            {
                "Tol Loss": "{:.4f}".format(abs(loss)),
                "Var Loss": "{:.4f}".format(abs(loss_r)),
                "Bnd Loss": "{:.4f}".format(abs(loss_b)),
            }
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()
        loss_t = loss.detach().numpy()
        losses.append(abs(loss_t))
        loss_r = loss_r.detach().numpy()
        losses_r.append(abs(loss_r))
        loss_b = loss_b.detach().numpy()
        losses_b.append(abs(loss_b))
        saved = False
        if epoch > int(4 * EPOCHS / 5):
            if torch.abs(loss) < best_loss:
                best_loss = torch.abs(loss).item()
                best_b = loss_b
                best_r = loss_r
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(path, "ckpt.bin"))
        if not saved:
            torch.save(model.state_dict(), os.path.join(path, "ckpt.bin"))
    print(
        "Best epoch:",
        best_epoch,
        "Best loss:",
        best_loss,
        "Best Var loss:",
        best_r,
        "Best Boundary loss",
        best_b,
    )
    elapse = time.time() - t0
    print(f"Training time: {elapse}")
    print(f"# of parameters: {get_param_num(model)}")
    # plot figure
    model.eval()
    model.load_state_dict(torch.load(os.path.join(path, "ckpt.bin")))
    print("Load weights from checkpoint!")
    with torch.no_grad():
        x1 = torch.linspace(-1, 1, 1001)
        x2 = torch.linspace(-1, 1, 1001)
        X, Y = torch.meshgrid(x1, x2)
        Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)
        Z = Z.to(device)
        pred = model(Z)

    point_wise_loss = (
        torch.abs(pred.reshape(1001, 1001) - u_real(Z).reshape(1001, 1001))
        .cpu()
        .numpy()
    )
    plot_result_and_save(point_wise_loss, path, NAME="loss")
    pred = pred.cpu().numpy()
    pred = pred.reshape(1001, 1001)
    plot_result_and_save(pred, path)

    print("Output figure saved!")
