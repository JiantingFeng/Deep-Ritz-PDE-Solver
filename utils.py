from configparser import Interpolation
import torch
import numpy as np
from math import *
import os
import random
import matplotlib.pyplot as plt


def seed_everything(seed=42):

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_param_num(model):
    return sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())


def get_interior_points(N=128, d=2):
    """
    randomly sample N points from interior of [-1,1]^d
    """
    return torch.rand(N, d) * 2 - 1


def get_boundary_points(N=33):
    index = torch.rand(N, 1)
    index1 = torch.rand(N, 1) * 2 - 1
    xb1 = torch.cat((index, torch.zeros_like(index)), dim=1)
    xb2 = torch.cat((index1, torch.ones_like(index1)), dim=1)
    xb3 = torch.cat((index1, torch.full_like(index1, -1)), dim=1)
    xb4 = torch.cat((torch.ones_like(index1), index1), dim=1)
    xb5 = torch.cat((torch.full_like(index1, -1), index1), dim=1)
    xb = torch.cat((xb1, xb2, xb3, xb4, xb5), dim=0)

    return xb


def get_boundary_points_high(N=100):
    xb = torch.rand(2 * 10 * N, 10)
    for i in range(10):
        xb[2 * i * N : (2 * i + 1) * N, i] = 0.0
        xb[(2 * i + 1) * N : (2 * i + 2) * N, i] = 1.0

    return xb


def plot_loss_and_save(EPOCHS, SAMPLE, losses, losses_r, losses_b, path="."):
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(
        range(1, EPOCHS, ceil(EPOCHS / SAMPLE)),
        np.log(losses[:: ceil(EPOCHS / SAMPLE)]),
        label="Total Loss",
    )
    ax.plot(
        range(1, EPOCHS, ceil(EPOCHS / SAMPLE)),
        np.log(losses_r[:: ceil(EPOCHS / SAMPLE)]),
        label="Variational Loss",
    )
    ax.plot(
        range(1, EPOCHS, ceil(EPOCHS / SAMPLE)),
        np.log(losses_b[:: ceil(EPOCHS / SAMPLE)]),
        label="Boundary Loss",
    )
    ax.set_title(r"$Log$ Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\log$ Loss")
    ax.legend()
    plt.savefig(os.path.join(path, "training_loss.pdf"))


def plot_result_and_save(pred, path=".", NAME="dnn_solution.pdf"):
    plt.figure()
    fig, ax = plt.subplots()
    h = ax.imshow(
        pred,
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
    plt.savefig(os.path.join(path, NAME))


def plot_loss_and_save_high(EPOCHS, SAMPLE, losses, path="."):
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(
        range(1, EPOCHS, ceil(EPOCHS / SAMPLE)),
        np.log(losses[:: ceil(EPOCHS / SAMPLE)]),
        label="Loss",
    )
    ax.set_title(r"$Log$ Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\log$ Loss")
    ax.legend()
    plt.savefig(os.path.join(path, "training_loss.pdf"))
