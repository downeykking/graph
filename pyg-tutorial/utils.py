import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import random
import numpy as np
import torch


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def visualize(h, color):
    z = TSNE().fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color.cpu().numpy(), cmap="Set2")
    plt.show()
    plt.savefig()