#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# Author:  Jun Jin, jjin5@ualberta.ca
# ==========================================================
"""
graph modules
"""

import numpy as np
import torch
import numpy as np
import torch.nn as nn
from deep_geometry.tools import *


class GeometrySkillBasis(nn.Module):
    """
    parent class of all geometric skill basis
    1 input key point heatmaps
    2 do multiple step using Message Passing
    3 do final output using GraphReadout, readout function
    """

    def __init__(self, node_num, layer_num, msg_dim, h_dim, output_dim, msg_aggrgt='AVG'):
        super(GeometrySkillBasis, self).__init__()
        self.dim_h, self.msg_aggrgt, self.msg_dim, self.node_num = h_dim, msg_aggrgt, msg_dim, node_num
        self.graph_layer_num = layer_num
        self.msg_passing = MessagePassing(h_dim, msg_dim, msg_aggrgt)
        self.readout = GraphReadOut(h_dim, node_num, output_dim)
        self.A = None  # adjacent matrix

    def forward(self, h_init):
        """
        input: node features X, and adjacent matrix
        inputs: n graph instances : nodes : node vector
        1 project x to H_initial
        2 for 0 - graph_layers_num (time step): do MessagePassing, get Ht
        3 feed final Hn to readout function
        :param h_init:  k key_points (each is a h_dim vector)
        :return: graph read out vector
        """
        for t in range(self.graph_layer_num):
            h_init = self.msg_passing(h_init, self.A)
        return self.readout(h_init)


class PPGraphBasis(GeometrySkillBasis):
    def __init__(self, layer_num, msg_dim, h_dim, output_dim):
        super().__init__(2, layer_num, msg_dim, h_dim, output_dim)
        self.A = np.array([[0, 1], [1, 0]])


class PLGraphBasis(GeometrySkillBasis):
    def __init__(self, layer_num, msg_dim, h_dim, output_dim):
        super().__init__(3, layer_num, msg_dim, h_dim, output_dim)
        self.A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])


class LLGraphBasis(GeometrySkillBasis):
    def __init__(self, layer_num, msg_dim, h_dim, output_dim):
        super().__init__(4, layer_num, msg_dim, h_dim, output_dim)
        self.A = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]])


