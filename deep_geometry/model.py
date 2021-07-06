#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# Author:  Jun Jin, jjin5@ualberta.ca
# ==========================================================
"""
Class for deep geometric feature set models
"""

import numpy as np
import copy
import math
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.nn.parameter import Parameter
from deep_geometry.skill_graph_basis import *
import copy
import collections


class LinearExpandingLayer(nn.Module):
    """
    linear expanding layer to learn weights for each basis
    return the same dim as the input but with weight ratios
    """

    def __init__(self, _basis_vector_dim, _basis_vector_num):
        super(LinearExpandingLayer, self).__init__()
        self.basis_vector_dim = _basis_vector_dim
        self.weight = Parameter(torch.Tensor(_basis_vector_num))
        self.register_parameter('bias', None)  # without bias term
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        repeated_weight = self.weight.repeat_interleave(self.basis_vector_dim)
        return x * repeated_weight


class DeepGeometricSet(nn.Module):

    def __init__(self, _conv_encoder=None, _encoder_out_channels=32, _vi_key_pointer=None, key_point_num=20,
                 _gnn_params=None, _debug_tool=None, _debug_frequency=None, _vi_mode_on=True):
        """
        deep geometric feature set as state representation
        :param _conv_encoder:
        :param _encoder_out_channels:
        :param _vi_key_pointer:
        :param key_point_num: default 20
        :param _gnn_params: {'layer_nums': [3, 3, 3, 4, 4, 5, 5], 'msg_dim':256, 'h_dim':512, 'output_dim':512, 'share_basis_graphs':True}
        """
        super(DeepGeometricSet, self).__init__()
        self.encoder = _conv_encoder
        self.k_heatmaps_layer = nn.Sequential(
            nn.Conv2d(_encoder_out_channels, key_point_num, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(key_point_num),
            nn.ReLU())
        self.vi_key_pointer = _vi_key_pointer
        # _gnn_params
        _gnn_params = dict(_gnn_params)
        gnn_layers_nums, gnn_msg_dim, gnn_h_dim, gnn_output_dim, share_basis_graphs = list(_gnn_params.values())
        pp_graph_layer_num, pl_graph_layer_num, ll_graph_layer_num = gnn_layers_nums
        if share_basis_graphs:
            self.pp_basis = PPGraphBasis(pp_graph_layer_num, msg_dim=gnn_msg_dim, h_dim=gnn_h_dim, output_dim=gnn_output_dim)
            self.pl_basis = PLGraphBasis(pl_graph_layer_num, msg_dim=gnn_msg_dim, h_dim=gnn_h_dim, output_dim=gnn_output_dim)
            self.ll_basis = LLGraphBasis(ll_graph_layer_num, msg_dim=gnn_msg_dim, h_dim=gnn_h_dim, output_dim=gnn_output_dim)
            self.graph_sets = {'pp1': self.pp_basis, 'pp2': self.pp_basis, 'pp3': self.pp_basis,
                               'pl1': self.pl_basis, 'pl2': self.pl_basis,
                               'll1': self.ll_basis, 'll2': self.ll_basis}
        else:
            self.pp1 = PPGraphBasis(pp_graph_layer_num, msg_dim=gnn_msg_dim, h_dim=gnn_h_dim, output_dim=gnn_output_dim)
            self.pp2 = PPGraphBasis(pp_graph_layer_num, msg_dim=gnn_msg_dim, h_dim=gnn_h_dim, output_dim=gnn_output_dim)
            self.pp3 = PPGraphBasis(pp_graph_layer_num, msg_dim=gnn_msg_dim, h_dim=gnn_h_dim, output_dim=gnn_output_dim)
            self.pl1 = PLGraphBasis(pl_graph_layer_num, msg_dim=gnn_msg_dim, h_dim=gnn_h_dim, output_dim=gnn_output_dim)
            self.pl2 = PLGraphBasis(pl_graph_layer_num, msg_dim=gnn_msg_dim, h_dim=gnn_h_dim, output_dim=gnn_output_dim)
            self.ll1 = LLGraphBasis(ll_graph_layer_num, msg_dim=gnn_msg_dim, h_dim=gnn_h_dim, output_dim=gnn_output_dim)
            self.ll2 = LLGraphBasis(ll_graph_layer_num, msg_dim=gnn_msg_dim, h_dim=gnn_h_dim, output_dim=gnn_output_dim)
            self.graph_sets = {'pp1': self.pp1, 'pp2': self.pp2, 'pp3': self.pp3,
                               'pl1': self.pl1, 'pl2': self.pl2,
                               'll1': self.ll1, 'll2': self.ll2}
        self.basis_weight_layer = LinearExpandingLayer(gnn_output_dim, len(self.graph_sets))
        self.debug_tool = _debug_tool
        self.debug_frequency = _debug_frequency
        self.it_count = 0
        self.vi_mode_on = _vi_mode_on

    def forward(self, batch_image_tensors):
        self.it_count += 1  # this is only for debug purpose
        x = self.encoder(batch_image_tensors)
        x = self.k_heatmaps_layer(x)  # x: k=20 heat maps
        # print('conv heatmap size')
        # print(x.shape)
        if self.debug_tool is not None and self.it_count % self.debug_frequency == 0:  # if debug mode, output more info
            print('debug')
            conv_heatmaps = copy.deepcopy(x.detach().to('cpu'))
            x, gauss_mu, gauss_maps, vi_key_point_heatmaps = self.vi_key_pointer(x, self.vi_mode_on)
            self.debug_tool.vis_debugger(batch_image_tensors.detach().to('cpu'),
                                         conv_heatmaps,
                                         gauss_maps.detach().to('cpu'),
                                         vi_key_point_heatmaps.detach().to('cpu'),
                                         gauss_mu[:, :, 0].detach().to('cpu'),
                                         gauss_mu[:, :, 1].detach().to('cpu'),
                                         show_graphs=True,
                                         local=True
                                         )

        else:
            x, _, _, _ = self.vi_key_pointer(x)  # x: k=20 key points with visual features

        geometry_basis_encodings = torch.cat(
            [self.graph_sets['pp1'](x[:, 0:2, :]),  # N, C, h, w
             self.graph_sets['pp2'](x[:, 2:4, :]),
             self.graph_sets['pp3'](x[:, 4:6, :]),
             self.graph_sets['pl1'](x[:, 6:9, :]),
             self.graph_sets['pl2'](x[:, 9:12, :]),
             self.graph_sets['ll1'](x[:, 12:16, :]),
             self.graph_sets['ll2'](x[:, 16:20, :])
             ], 1)  # an array of _gnn_output_dim vectors
        # print('geometry basis encodings shape')
        # print(geometry_basis_encodings.shape)   # 10 * 3584(7*_gnn_output_dim)
        return self.basis_weight_layer(geometry_basis_encodings)


if __name__ == "__main__":
    # test
    from tools import *
    from vi_key_point_bottleneck import *
    conv_layer_params = {'filter num': [16, 16, 32, 32], 'kernel sizes': [7, 3, 3, 3], 'strides': [1, 1, 2, 1]}
    conv_encoder = ImageEncoder(input_channel_num=3, layer_params=conv_layer_params)

    encoder_out_channel_num = conv_layer_params['filter num'][-1]
    gnn_params = {'layer_nums': [3, 4, 5], 'msg_dim': 128, 'h_dim': 128, 'output_dim': 128, 'share_basis_graphs': True}

    vi_key_pointer = LocalViKeyPointBottleneck(_key_point_num=20, _gauss_std=0.1,
                                          _output_dim=gnn_params['h_dim'], _flatten='conv2d')

    from debug_tools import *
    debug_tool = DeepGeometrySetVisualizer(_save_dir='../raw', _name_prefix='test', _verbose=True)

    deep_geometry_set = DeepGeometricSet(_conv_encoder=conv_encoder,
                                         _encoder_out_channels=encoder_out_channel_num,
                                         _vi_key_pointer=vi_key_pointer,
                                         key_point_num=20, _gnn_params=gnn_params, _debug_tool=debug_tool, _debug_frequency=1
                                         )
    deep_geometry_set.apply(init_weights)

    # prepare an image tensor
    device = 'cuda:0'
    deep_geometry_set = deep_geometry_set.to(device)
    images = np.load('../raw/sample_img.npy')
    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    images = transform(images).to(device).unsqueeze(0)
    # images = torch.randn((10,3,128,128)).to(device)

    basis_weighted_layer = deep_geometry_set(images)  # dim N * (7*_gnn_output_dim)
    # print(basis_weighted_layer)
    # from pytorch_model_summary import summary
    # print(summary(deep_geometry_set, torch.zeros(1, 3, 128, 128).to(device), show_input=False, show_hierarchical=True))

    # import matplotlib.pyplot as plt
    # MEAN = torch.tensor([0.485, 0.456, 0.406])
    # STD = torch.tensor([0.229, 0.224, 0.225])
    # images = images.to('cpu') * STD[:, None, None] + MEAN[:, None, None]  # de-normalize for vis
    # # plt.imshow(images[0].permute(1, 2, 0))
    # # if pure color
    # plt.imshow(np.zeros((128, 128, 3)))
    # plt.scatter(x=[30, 40], y=[50, 60], c='r', s=30)  # scatter points
    # plt.plot([30, 40], [50, 60], color="b")  # plt.plot(x1, y1, x2, y2, color="white", linewidth=3)
    # # plt.axis('off')
    # conv_encoder = conv_encoder.to(device)
    # conv_layer_params = {'filter num': [16, 16, 32, 32], 'kernel sizes': [7, 3, 3, 3], 'strides': [1, 1, 2, 1]}
    # conv_encoder2 = ImageEncoder(input_channel_num=3, layer_params=conv_layer_params)
    # import torch.optim as optim
    # # optimizerG = optim.Adam(conv_encoder.parameters(), lr=1e-4)
    # # optimizerG2 = optim.Adam(conv_encoder2.parameters(), lr=1e-4)
    # # y1 = conv_encoder(images)
    # # y2 = conv_encoder2(images)
    # deep_geometry_set2 = DeepGeometricSet(_conv_encoder=conv_encoder2,
    #                                      _encoder_out_channels=encoder_out_channel_num,
    #                                      _vi_key_pointer=vi_key_pointer,
    #                                      key_point_num=20, _gnn_params=gnn_params, _debug_tool=debug_tool,
    #                                      _debug_frequency=10
    #                                      )
    # deep_geometry_set2.apply(init_weights)
    # deep_geometry_set2 = deep_geometry_set.to(device)
    # optg1 = optim.Adam(deep_geometry_set.parameters(), lr=1e-4)
    # optg2 = optim.Adam(deep_geometry_set2.parameters(), lr=1e-4)
    # y1 = deep_geometry_set(images)
    # y2 = deep_geometry_set2(images)




