from dataclasses import dataclass
import torch

from topo_utils import Topo

@dataclass
class Equation:
    F_prefix: str
    G_prefix: str
    F_infix: str
    G_infix: str
    F_infix_c: str
    G_infix_c: str
    
    dimension: int
    id: str


@dataclass
class FitParams:
    word2id: dict
    id2word: dict
    total_variables: list


@dataclass
class TestEquation:
    node_state: list
    neighbour_state: list
    info: torch.tensor
    y: torch.tensor
    F_token: torch.tensor
    G_token: torch.tensor
    F_expr: list
    G_expr: list
    F_ske: list
    G_ske: list
    dimension: list
    topo: Topo


@dataclass
class DataInfo:
    Nodes_state: torch.tensor
    Nodes_neighbours: torch.tensor
    Nodes_neighbours_info: torch.tensor
    Y: torch.tensor
    F_token: torch.tensor
    G_token: torch.tensor
    F_equation: list
    G_equation: list
    F_ske: list
    G_ske: list
    Dimension: list
    Topo: Topo
    Index_sample_time: list
    Index_sample_node: list
    Origin_data: dict