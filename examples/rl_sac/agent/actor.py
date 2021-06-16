import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
import examples.rl_sac.utils as utils


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(obs_dim, hidden_dim, 2 * action_dim,
                               hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


class DiagGaussianActorWithEncoder(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds):
        super().__init__()
        self.image_encoder = None
        # self.before_mlp_layers = nn.Sequential(
        #     nn.Linear(obs_dim, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 128),
        #     nn.ReLU(inplace=True)
        # )
        self.mlp_actor = DiagGaussianActor(obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds)

    def set_image_encoder(self, _encoder):
        self.image_encoder = _encoder

    def forward(self, obs):
        obs = self.image_encoder(obs)
        # obs = self.before_mlp_layers(obs)
        return self.mlp_actor(obs)

    def log(self, logger, step):
        self.mlp_actor.log(logger, step)



# class DeepGeometryGaussianActor(DiagGaussianActorWithEncoder):
#     def __init__(self, image_input_channel_num, conv_layer_params, gnn_params, key_point_num, key_pointer_gauss_std,
#                  key_pointer_flatten, debug_save_dir, debug_frequency,
#                  obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds):
#         # initialize deep geometry image encoder
#         conv_encoder = gtools.ImageEncoder(input_channel_num=image_input_channel_num, layer_params=conv_layer_params)
#         encoder_out_channel_num = conv_layer_params['filter num'][-1]
#         vi_key_pointer = ViKeyPointBottleneck(_key_point_num=key_point_num, _gauss_std=key_pointer_gauss_std,
#                                               _output_dim=gnn_params['h_dim'], _flatten=key_pointer_flatten)
#         debug_tool = DeepGeometrySetVisualizer(_save_dir=debug_save_dir, _verbose=True)
#         deep_geometry_set = DeepGeometricSet(_conv_encoder=conv_encoder,
#                                              _encoder_out_channels=encoder_out_channel_num,
#                                              _vi_key_pointer=vi_key_pointer,
#                                              key_point_num=20, _gnn_params=gnn_params,
#                                              _debug_tool=debug_tool, _debug_frequency=debug_frequency
#                                              )
#         deep_geometry_set.apply(gtools.init_weights)
#         # obs_dim = 7 * gnn_params['output_dim']  # 7 is the number of basis  # 896
#         super().__init__(deep_geometry_set, obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds)
#
#
# class ConvEncoderGaussianActor(DiagGaussianActorWithEncoder):
#     def __init__(self, image_input_channel_num, conv_layer_params,
#                  obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds):
#         conv_encoder = nn.Sequential(
#             gtools.ImageEncoder(input_channel_num=image_input_channel_num, layer_params=conv_layer_params),
#             gtools.View(-1, ),
#             nn.Linear(169, obs_dim),  # for a fair comparison 896 is exactly the dim of geometry set output dim
#             nn.ReLU())
#         super().__init__(conv_encoder, obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds)
