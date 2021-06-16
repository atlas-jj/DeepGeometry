import hydra
import yaml
from examples.rl_sac.encoders.make_encoders import *
from examples.rl_sac.agent.actor import *
import torch
import torchvision.transforms as T

action_range = [
    -0.5,  # 0.13
    0.5
]

transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

with open("rl_sac/config/test.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)

actor_params = cfg['actor']['params']
actor = DiagGaussianActorWithEncoder(actor_params['obs_dim'],
                                     actor_params['action_dim'],
                                     actor_params['hidden_dim'],
                                     actor_params['hidden_depth'],
                                     actor_params['log_std_bounds'])
if cfg['actor']['need_encoder']:  # plugin an encoder
    enc = cfg['encoder']
    encoder = make_encoders(enc)
    # enc['image_input_channel_num'], enc['conv_layer_params'], enc['gnn_params'], enc['key_point_num'] , enc['key_pointer_gauss_std'], enc['key_pointer_flatten'], enc['debug_save_dir'], enc['debug_frequency'], enc['output_dim'])
    # tie up the encoder of actor and critic
    actor.set_image_encoder(encoder)


def act(obs, sample=False):
    obs = torch.FloatTensor(obs)
    obs = obs.unsqueeze(0)
    dist = actor(obs)
    action = dist.sample() if sample else dist.mean
    action = action.clamp(*action_range)
    assert action.ndim == 2 and action.shape[0] == 1
    return utils.to_np(action[0])


# load parameters
actor.mlp_actor.load_state_dict(torch.load('0_mlp.pth'))
actor.image_encoder.load_state_dict(torch.load('0_encoder.pth'))

actions = act(torch.ones(3, 128, 128))
