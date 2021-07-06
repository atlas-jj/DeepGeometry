import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import examples.rl_sac.utils as utils


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, layer_norm=False):
        super().__init__()

        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth, layer_norm=layer_norm)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth, layer_norm=layer_norm)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)


class DoubleQCriticWithEncoder(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, has_proprio_input, proprio_dim, proprio_expanded_dim):
        super().__init__()
        self.image_encoder = None
        self.has_proprio_input = has_proprio_input
        if has_proprio_input:
            self.proprio_dim = proprio_dim  # joint, joint_vel, joint_torque
            self.proprio_layer = nn.Sequential(
                nn.Linear(self.proprio_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, proprio_expanded_dim),
                nn.ReLU(inplace=True)
            )
            self.mlp_critic = DoubleQCritic(obs_dim+proprio_expanded_dim, action_dim, hidden_dim, hidden_depth, layer_norm=True)
        else:
            self.mlp_critic = DoubleQCritic(obs_dim, action_dim, hidden_dim, hidden_depth, layer_norm=False)

    def set_image_encoder(self, _encoder):
        self.image_encoder = _encoder

    def forward(self, obs, action):
        if self.has_proprio_input:
            obs_img = self.image_encoder(obs['image'])
            proprio_signals = self.proprio_layer(obs['proprios'])
            obs = torch.hstack((obs_img, proprio_signals))
        else:
            obs = self.image_encoder(obs)
        return self.mlp_critic(obs, action)

    def log(self, logger, step):
        self.mlp_critic.log(logger, step)

