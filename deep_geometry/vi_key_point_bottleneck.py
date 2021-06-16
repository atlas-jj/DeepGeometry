import numpy as np
import torch
import numpy as np
import torch.nn as nn
from deep_geometry.tools import *


class ViKeyPointBottleneck(nn.Module):
    """
    # visual key point bottleneck
    convert k heatmaps to k vi key point bottleneck heat maps by soft max operator and Gaussian operator
    basically, it's multiply the original k heat map with a skip connection, just like the ResNet
    """

    def __init__(self, _key_point_num, _gauss_std, _output_dim, _flatten='conv2d'):
        super(ViKeyPointBottleneck, self).__init__()
        self.key_point_num = _key_point_num
        self.gauss_std = torch.tensor(_gauss_std)
        self.soft_max = nn.Softmax(dim=2)
        self.flatten = _flatten
        if _flatten == 'conv2d':
            self.key_point_heatmap_encoder = KeyPointHeatmapEncoder(layer_params={'filter num': [8, 8, 32],
                                                                                  'operator': ['conv2d','max_pool','conv2d'],
                                                                                  'kernel sizes': [3, 3, 3],
                                                                                  'strides': [1, 2, 2]}
                                                                    )
        self.to_hidden_node = nn.Sequential(nn.Linear(169, _output_dim),  # 1458
                                            nn.ReLU()
                                            )

    def forward(self, k_heat_maps, vi_mode_on=True):
        """
        [N, C, H, W].
        takes in k channel heat maps and return k key point heatmaps via visual key point bottleneck
        :param k_heat_maps: N * k * w * h heatmaps directly from just normal conv layers
        :return: N * k * h * w heatmaps by visual keypoint bottleneck.
        """
        gauss_mu = self._soft_max_key_points(k_heat_maps)  # centers
        map_size = k_heat_maps.shape[2:4]
        gauss_maps = self._get_gaussian_maps(gauss_mu, map_size)  # gauss normalized heatmaps
        if vi_mode_on:
            # print('vi_mode ON')
            vi_key_point_heatmaps = k_heat_maps * gauss_maps  # do element wise multiplication
        else:
            # print('vi_mode OFF')
            vi_key_point_heatmaps = gauss_maps
        N, k = vi_key_point_heatmaps.shape[0:2]
        # N * k * h * w input --> N*k *1 * h * w input --> N * k * output_dim
        # print('vi_key_point_heatmaps shape: ')
        # print(vi_key_point_heatmaps.shape)  # 57 * 57
        #if self.flatten == 'conv2d':  # use conv2d, then flatten to desired output dimension
        h = vi_key_point_heatmaps.flatten(start_dim=0, end_dim=1).unsqueeze(1)
        h = self.key_point_heatmap_encoder(h)  # N*k * output dim
        h = self.to_hidden_node(h.reshape(N, k, -1))  # # N * k * output dim
        return h, gauss_mu, gauss_maps, vi_key_point_heatmaps

    ########################
    # private functions #
    ########################
    def _soft_max_key_points(self, k_heat_maps):
        """

        :param k_heat_maps: N0 * k1 * hy2 * wx3
        :return:
        """
        gauss_x = self._get_coord(k_heat_maps, 3)  # each image with 20 key points selected
        gauss_y = self._get_coord(k_heat_maps, 2)  # each image with 20 key points selected

        # print(gauss_y)  # each image with 20 key points selected
        gauss_mu = torch.stack([gauss_x, gauss_y], axis=2)  # N * k * 2
        return gauss_mu

    def _get_coord(self, k_heat_maps, axis):
        """
        get coordinates from k heatmaps
        :param k_heat_maps:
        :param axis:
        :return:
        """
        other_axis = 2 if axis == 3 else 3
        axis_size = k_heat_maps.shape[axis]

        g_c_prob = torch.mean(k_heat_maps, dim=other_axis)  # sum up each axis elements
        g_c_prob = self.soft_max(g_c_prob)

        # Linear combination of the interval [-1, 1] using the normalized weights to
        # give a single coordinate in the same interval [-1, 1]
        scale = torch.linspace(-1.0, 1.0, axis_size).type(torch.FloatTensor)
        scale = torch.reshape(scale, [1, 1, axis_size]).to(k_heat_maps.device)
        coordinate = torch.sum(g_c_prob * scale, dim=2)
        return coordinate

    def _get_gaussian_maps(self, mu, map_size, power=2):
        """Transforms the keypoint center points to a gaussian masks."""
        # N * k * hy * wx
        mu_x, mu_y = mu[:, :, 0:1], mu[:, :, 1:2]

        x = torch.linspace(-1.0, 1.0, map_size[1]).type(torch.FloatTensor)  # change range [-1,1] to [0, 1]
        y = torch.linspace(-1.0, 1.0, map_size[0]).type(torch.FloatTensor)

        mu_x, mu_y = torch.unsqueeze(mu_x, 3), torch.unsqueeze(mu_y, 3)  # N * k * 1 * 1

        x = torch.reshape(x, [1, 1, 1, map_size[1]]).to(mu.device)
        y = torch.reshape(y, [1, 1, map_size[0], 1]).to(mu.device)

        g_y = torch.pow(y - mu_y, power)
        g_x = torch.pow(x - mu_x, power)
        dist = (g_x + g_y) * torch.pow(1.0/self.gauss_std, power)
        g_yx = torch.exp(-dist)  # # hy * wx
        return g_yx
