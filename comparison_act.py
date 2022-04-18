#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_theme(style="whitegrid")

# Different activation functions
def ReLU(x):
    return np.maximum(0, x)


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


def Tanh(x):
    return np.tanh(x)


def ELU(x):
    return np.where(x > 0, x, np.exp(x) - 1)


def GELU(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def SiLU(x):
    return x * Sigmoid(x)


# Draw the activation functions in the same graph
def draw_activation_functions(activation_functions):
    x = np.arange(-3, 3, 0.1)
    for activation_function in activation_functions:
        plt.plot(
            x, activation_function(x), label=activation_function.__name__, linewidth=1
        )
    plt.legend()
    plt.show()
    plt.savefig("activation_functions.pdf")


if __name__ == "__main__":
    activation_functions = [ReLU, Sigmoid, Tanh, ELU, GELU, SiLU]
    draw_activation_functions(activation_functions)
