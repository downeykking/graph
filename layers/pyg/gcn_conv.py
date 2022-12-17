import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import sort_edge_index
from torch_geometric import seed_everything
from typing import Optional, Tuple, Union

from torch_geometric.nn import MessagePassing