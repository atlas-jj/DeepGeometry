#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# Author:  Jun Jin, jjin5@ualberta.ca
# ==========================================================
"""
graph modules
"""
import math

import numpy as np
import torch
import numpy as np
import torch.nn as nn
from deep_geometry.tools import *
from geometer import Point, Line


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

    def local_geometric_err(self, point_coors):
        pass

    def set_control_rectify_scale(self):
        """
        rectify the errors by a scale
        :return:
        """
        pass


class PPGraphBasis(GeometrySkillBasis):
    def __init__(self, layer_num, msg_dim, h_dim, output_dim):
        super().__init__(2, layer_num, msg_dim, h_dim, output_dim)
        self.A = np.array([[0, 1], [1, 0]])

    def local_geometric_err(self, point_coors):
        """
        point to point errors.
        :param point_coors: N * 2 [x, y],  x1, y1: current, x2, y2: target, img coors normalized by [-1,1]
        all point_coors should be scaled back to the original image coors
        :return: N * [dx, dy] s - s*
        """
        errs = point_coors[:, 0, :] - point_coors[:, 1, :]
        return errs


class PLGraphBasis(GeometrySkillBasis):
    def __init__(self, layer_num, msg_dim, h_dim, output_dim):
        super().__init__(3, layer_num, msg_dim, h_dim, output_dim)
        self.A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])

    def local_geometric_err(self, point_coors):
        """
        point to line errors
        :param point_coors: N * [x1, y1, x2, y2, x3, y3], x1, y1: the point, x2, y2, x3, y3: the line.
        all point_coors should be scaled back to the original image coors
        :return:
        """
        # calculate line function from two points
        N = point_coors.shape[0]
        p2l_errs = []
        for i in range(N):
            l = Line(Point(point_coors[i,1,0].item(), point_coors[i,1,1].item()), Point(point_coors[i,2,0].item(), point_coors[i,2,1].item()))
            l_coors = l.array
            err = point_coors[i,0,0].item()*l_coors[0] + point_coors[i,0,1].item()*l_coors[1] + l_coors[2]
            p2l_errs.append(err)
        return torch.unsqueeze(torch.tensor(p2l_errs),1).to(point_coors.device)  # N * 1


class LLGraphBasis(GeometrySkillBasis):
    def __init__(self, layer_num, msg_dim, h_dim, output_dim):
        super().__init__(4, layer_num, msg_dim, h_dim, output_dim)
        self.A = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]])

    def local_geometric_err(self, point_coors):
        """
        line to line errors
        :param point_coors: N * 4 * [x,y], x1 y1, x2 y2: line 1; x3 y3, x4 y4: line 2
        line 1 is the current line, line2 is the target line
        :return:
        """
        N = point_coors.shape[0]
        l2l_errs = []
        for i in range(N):
            l1 = Line(Point(point_coors[i,0,0].item(), point_coors[i,0,1].item()),
                     Point(point_coors[i,1,0].item(), point_coors[i,1,1].item()))
            l2 = Line(Point(point_coors[i,2,0].item(), point_coors[i,2,1].item()),
                      Point(point_coors[i,3,0].item(), point_coors[i,3,1].item()))
            l1_coors = l1.array
            l2_coors = l2.array
            d_rho = -l1.array[2] + l2.array[2]  # x*cos theta + y * sin theta - rho
            d_theta = math.atan2(l1.array[1], l1.array[0]) - math.atan2(l2.array[1], l2.array[0])
            if d_theta < -math.pi:
                d_theta += 2*math.pi
            if d_theta > math.pi:
                d_theta = d_theta - 2*math.pi
            l2l_errs.append([d_rho, d_theta])
        return torch.tensor(l2l_errs).to(point_coors.device)  # N * 1


