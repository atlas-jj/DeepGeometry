import numpy as np
import torch
import numpy as np
import torch.nn as nn
from deep_geometry.tools import *
import abc


class KeyPointBottleneck(nn.Module):
    """
    # visual key point bottleneck
    convert k heatmaps to k vi key point bottleneck heat maps by soft max operator and Gaussian operator
    basically, it's multiply the original k heat map with a skip connection, just like the ResNet
    """
    def __init__(self, input_conv_channels, _key_point_num, _gauss_std, _output_dim, _flatten='conv2d'):
        super(KeyPointBottleneck, self).__init__()
        self.key_point_num = _key_point_num
        self.gauss_std = torch.tensor(_gauss_std)
        self.soft_max = nn.Softmax(dim=2)
        self.flatten = _flatten
        self.key_point_detector = nn.Sequential(
            nn.ConvTranspose2d(input_conv_channels, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, _key_point_num, 3, 2),
            nn.BatchNorm2d(_key_point_num),
            nn.ReLU(True)
        )

    @abc.abstractmethod
    def forward(self, k_heat_maps, vi_mode_on=True):
        pass

    ########################
    # private functions #
    ########################
    def _soft_max_key_points(self, k_heat_maps):
        """

        :param k_heat_maps: N0 * k1 * hy2 * wx3
        :return:
        """
        print('k heatmaps')
        print(k_heat_maps.shape)
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

        x = torch.linspace(-1.0, 1.0, map_size[1]).type(torch.FloatTensor)
        y = torch.linspace(-1.0, 1.0, map_size[0]).type(torch.FloatTensor)

        mu_x, mu_y = torch.unsqueeze(mu_x, 3), torch.unsqueeze(mu_y, 3)  # N * k * 1 * 1

        x = torch.reshape(x, [1, 1, 1, map_size[1]]).to(mu.device)
        y = torch.reshape(y, [1, 1, map_size[0], 1]).to(mu.device)

        g_y = torch.pow(y - mu_y, power)
        g_x = torch.pow(x - mu_x, power)
        dist = (g_x + g_y) * torch.pow(1.0/self.gauss_std, power)
        g_yx = torch.exp(-dist)  # # hy * wx
        return g_yx

# class ViKeyPointBottleneck(KeyPointBottleneck):
#     """
#     # visual key point bottleneck
#     convert k heatmaps to k vi key point bottleneck heat maps by soft max operator and Gaussian operator
#     basically, it's multiply the original k heat map with a skip connection, just like the ResNet
#     """
#     def __init__(self, _key_point_num, _gauss_std, _output_dim, _flatten='conv2d'):
#         super().__init__(_key_point_num, _gauss_std, _output_dim, _flatten=_flatten)
#         if _flatten == 'conv2d':
#             self.key_point_heatmap_encoder = KeyPointHeatmapEncoder(layer_params={'filter num': [8, 8, 32],
#                                                                                   'operator': ['conv2d', 'max_pool',
#                                                                                                'conv2d'],
#                                                                                   'kernel sizes': [3, 3, 3],
#                                                                                   'strides': [1, 2, 2]}
#                                                                     )
#         self.to_hidden_node = nn.Sequential(nn.Linear(169, _output_dim),  # 1458
#                                             nn.ReLU()
#                                             )
#
#     def forward(self, k_heat_maps, vi_mode_on=True):
#         """
#         [N, C, H, W].
#         takes in k channel heat maps and return k key point heatmaps via visual key point bottleneck
#         :param k_heat_maps: N * k * h * w heatmaps directly from just normal conv layers
#         :return: N * k * h * w heatmaps by visual keypoint bottleneck.
#         """
#         # print(k_heat_maps.shape)  # [N, C, H, W].
#         gauss_mu = self._soft_max_key_points(k_heat_maps)*10  # centers
#         # print(gauss_mu)
#         map_size = k_heat_maps.shape[2:4]
#         gauss_maps = self._get_gaussian_maps(gauss_mu, map_size)  # gauss normalized heatmaps
#         # print(gauss_maps.shape)
#         if vi_mode_on:
#             # print('vi_mode ON')
#             vi_key_point_heatmaps = k_heat_maps * gauss_maps  # do element wise multiplication
#         else:
#             # print('vi_mode OFF')
#             vi_key_point_heatmaps = gauss_maps
#         N, k = vi_key_point_heatmaps.shape[0:2]
#         # print(vi_key_point_heatmaps.shape)  # [N, C, H, W].
#         # N * k * h * w input --> N*k *1 * h * w input --> N * k * output_dim
#         # print('vi_key_point_heatmaps shape: ')
#         # print(vi_key_point_heatmaps.shape)  # 57 * 57
#         #if self.flatten == 'conv2d':  # use conv2d, then flatten to desired output dimension
#         # h = vi_key_point_heatmaps.flatten(start_dim=0, end_dim=1).unsqueeze(1)
#         # print(vi_key_point_heatmaps.shape)
#         h = self.key_point_heatmap_encoder(vi_key_point_heatmaps)  # N*k * output dim
#         h = self.to_hidden_node(h.reshape(N, k, -1))  # # N * k * output dim
#         return h, gauss_mu, gauss_maps, vi_key_point_heatmaps

class LocalViKeyPointBottleneck(KeyPointBottleneck):
    """
    # visual key point bottleneck
    convert k heatmaps to k vi key point bottleneck heat maps by soft max operator and Gaussian operator
    basically, it's multiply the original k heat map with a skip connection, just like the ResNet
    """
    def __init__(self, input_conv_channels, _key_point_num, _gauss_std, _output_dim, _flatten='conv2d', crop_size=20):
        super().__init__(input_conv_channels, _key_point_num, _gauss_std, _output_dim, _flatten=_flatten)
        self.crop_size = crop_size
        if _flatten == 'conv2d':
            self.key_point_heatmap_encoder = KeyPointHeatmapEncoder(layer_params={'filter num': [input_conv_channels, 128, 32],
                                                                                  'operator': ['conv2d', 'conv2d'],
                                                                                  'kernel sizes': [3, 3, 3],
                                                                                  'strides': [2, 2, 1]}
                                                                    )
            in_take_dim = 1024
        else:
            self.key_point_heatmap_encoder = None
            in_take_dim = crop_size*crop_size

        self.to_hidden_node = nn.Sequential(nn.Linear(in_take_dim, _output_dim),  # 144
                                            nn.ReLU()
                                            )

    def forward(self, x, vi_mode_on=True):
        """
        [N, C, H, W].
        takes in k channel heat maps and return k key point heatmaps via visual key point bottleneck
        :param k_heat_maps: N * k * w * h heatmaps directly from just normal conv layers
        :return: N * k * 3 * h * w heatmaps by visual keypoint bottleneck.
        """
        k_heat_maps = self.key_point_detector(x)
        gauss_mu = self._soft_max_key_points(k_heat_maps*2)  # centers
        print('gaussian mu')
        print(gauss_mu)
        map_size = x.shape[2:4] # k_heat_maps.shape[2:4]
        # print('map size')
        # print(map_size)
        gauss_maps = self._get_gaussian_maps(gauss_mu, map_size)  # gauss normalized heatmaps, N * K * H * W
        # torch.save(k_heat_maps, 'k_heatmaps_conv.pt')
        # torch.save(gauss_maps, 'gauss_maps.pt')
        # upsampling the gaussian map

        if vi_mode_on:
            # print('vi_mode ON')
            vi_key_point_heatmaps = []
            for i in range(k_heat_maps.shape[1]):  # for each keypoint, extract from original image
                key_point_channels = []
                for c in range(x.shape[1]):  # for each channel of the image
                    key_point_channels.append(x[:,c,:,:] * gauss_maps[:,i,:,:])
                key_point_channels = torch.stack(key_point_channels, dim=1)  # N * c * H * W
                vi_key_point_heatmaps.append(key_point_channels)
            vi_key_point_heatmaps = torch.stack(vi_key_point_heatmaps, dim=1)  # N * K * C * H * W
        else:
            # print('vi_mode OFF')
            vi_key_point_heatmaps = gauss_maps.unsqueeze(2)
        # torch.save(vi_key_point_heatmaps, 'vi_keypoints.pt')
        # center at the mu_x, mu_y point, crop to a size
        # print(vi_key_point_heatmaps.shape)  # [N, C, H, W].
        N, k = vi_key_point_heatmaps.shape[0:2]
        # crop it.
        # print('vi keypoint heatmap size')
        # print(vi_key_point_heatmaps.shape)
        centered_heatmaps = self._get_centered_key_point_heatmaps(vi_key_point_heatmaps, gauss_mu, map_size)  # this is the simply crop function
        # print(centered_heatmaps.shape)

        # print(centered_heatmaps)
        # N * k * h * w input --> N*k *1 * h * w input --> N * k * output_dim
        # print('vi_key_point_heatmaps shape: ')
        # print(vi_key_point_heatmaps.shape)  # 57 * 57
        #if self.flatten == 'conv2d':  # use conv2d, then flatten to desired output dimension
        # h = centered_heatmaps.flatten(start_dim=0, end_dim=1).unsqueeze(1)
        # print(h.shape)
        # print('center heathamps shape')
        # print(centered_heatmaps.shape)
        if self.key_point_heatmap_encoder is not None:
            h = []
            for i in range(centered_heatmaps.shape[0]):  # for each sample...
                h.append(self.key_point_heatmap_encoder(centered_heatmaps[i]).reshape(k, -1))  # N*k * output dim
            h = torch.stack(h, dim=0)
        else:
            h = centered_heatmaps.reshape(N, k, -1)
        print('h shape')
        print(h.shape)
        h = self.to_hidden_node(h)  # # N * k * output dim
        # print(h.shape)
        # print(gauss_mu)
        return h, gauss_mu, gauss_maps, centered_heatmaps

    def _get_centered_key_point_heatmaps(self, heatmap, mu, map_size):
        """Transforms the keypoint center points to a gaussian masks."""
        """heatmap: N * K * C * H * W"""
        # return N * K * c * croppedH * cropped W
        # empty heatmap with cropsize
        centered_heatmap = torch.zeros(heatmap.shape[0], heatmap.shape[1], heatmap.shape[2], self.crop_size, self.crop_size).to(heatmap.device)
        # N * k * hy * wx
        mu_x, mu_y = mu[:, :, 0:1], mu[:, :, 1:2]
        # print('mu x 3 {}, mu y 3 {}'.format(mu_x[0, 3].item(), mu_y[0,3].item()))
        # print(heatmap[0, 3])

        # makes x, y centered at mu_x, mu_y
        # x = torch.linspace(-1.0, 1.0, map_size[1]).type(torch.FloatTensor)
        # y = torch.linspace(-1.0, 1.0, map_size[0]).type(torch.FloatTensor)
        # locate cx, cy
        cx = torch.round((mu_x + 1)*(map_size[1])/2) -1  # W
        cy = torch.round((mu_y + 1) * (map_size[0]) / 2) -1 # h
        # print('cx {}, cy {}, mapsize[1] {}, mapsize[0] {}'.format(cx[0,3], cy[0,3], map_size[1], map_size[0]))

        for i in range(heatmap.shape[0]):  # N
            for j in range(heatmap.shape[1]):  # K maps
                cx_left = int(cx[i, j].item() - int(self.crop_size/2)) if cx[i, j].item() - int(self.crop_size/2)>0 else 0
                ccx_left = int(int(self.crop_size/2) - cx[i, j].item()) if cx[i, j].item() - int(self.crop_size/2)<0 else 0
                cx_right = int(cx[i, j].item() + int(self.crop_size/2)) if cx[i, j].item() + int(self.crop_size/2)<map_size[1] else map_size[1]

                cy_left = int(cy[i, j].item() - int(self.crop_size / 2)) if cy[i, j].item() - int(self.crop_size / 2) > 0 else 0
                ccy_left = int(int(self.crop_size/2) - cy[i, j].item()) if cy[i, j].item() - int(self.crop_size/2) <0 else 0
                cy_right = int(cy[i, j].item() + int(self.crop_size / 2)) if cy[i, j].item() + int(self.crop_size / 2) < map_size[0] else map_size[0]
                # print('cx left {}, ccx left {}, cx right {}, cy left {}, ccy left{}, cy right {}'.format(cx_left, ccx_left, cx_right, cy_left, ccy_left, cy_right))
                centered_heatmap[i, j, :, ccy_left:ccy_left+cy_right-cy_left,ccx_left:ccx_left+cx_right-cx_left] = heatmap[i, j, :, cy_left:cy_right, cx_left:cx_right]
        # print(centered_heatmap[0,3])
        #
        # print(heatmap[0, 3, int(cy[0,3].item())-2:int(cy[0,3].item())+2, int(cx[0,3].item())-2:int(cx[0,3].item())+2])
        # print(centered_heatmap[0,3,6:10, 6:10])
        return centered_heatmap

