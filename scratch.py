# import numpy as np
# import copy
# import math
# import torch
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter
# from torch.nn.modules.module import Module
# import abc
#
#
# class LinearExpandingLayer(nn.Module):
#     """
#     linear expanding layer to learn weights for each basis
#     return the same dim as the input but with weight ratios
#     """
#
#     def __init__(self, _basis_vector_dim, _basis_vector_num):
#         super(LinearExpandingLayer, self).__init__()
#         self.basis_vector_dim = _basis_vector_dim
#         self.weight = Parameter(torch.Tensor(_basis_vector_num))
#         self.register_parameter('bias', None)  # without bias term
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(0))
#         self.weight.data.uniform_(-stdv, stdv)
#
#     def forward(self, x):
#         repeated_weight = self.weight.repeat_interleave(self.basis_vector_dim)
#         return x * repeated_weight
#
# layer = LinearExpandingLayer(512, 7)
# # x = torch.randn(10, 512*7)
# x = torch.cat([torch.zeros(10,512), torch.ones(10,512), torch.ones(10,512)*2, torch.ones(10,512)*3, torch.ones(10,512)*4, torch.ones(10,512)*5, 6*torch.ones(10,512)],1)
# y = layer(x)
# out = torch.sum(y)
# out.backward()
# params = next(layer.parameters())
# print(x)
# print(params.grad)  # will be 4 times x because of the repeated weights. Autograd has taken the repetition into account
#
#
import metaworld
import random
#
# print(metaworld.ML45.ENV_NAMES)  # Check out the available environments
#
# ml1 = metaworld.ML1('pick-place-v2') # Construct the benchmark, sampling tasks
#
# env = ml1.train_classes['pick-place-v2']()  # Create an environment with task `pick_place`
# task = random.choice(ml1.train_tasks)
# env.set_task(task)  # Set task
#
# obs = env.reset()  # Reset environment
# a = env.action_space.sample()  # Sample an action
# obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN
from gym.wrappers import Monitor
import numpy as np
# from gym.envs.classic_control import rendering
import operator
import copy
# viewer = rendering.SimpleImageViewer()

tasks={
    # shor-tem task
    1: 'button-press-v2',
    2: 'soccer-v2',
    3: 'reach-v2',
    # long-term task: two sub skills
    4: 'box-close-v2',
    5: 'hammer-v2',
    6: 'assembly-v2',
    # vision + force tasks
    7: 'door-open-v2',
    8: 'window-open-v2',
    9: 'faucet-open-v2'
}

camera_views={
    1: 'corner',
    2: 'corner',
    3: 'corner',
    4: 'corner',
    5: 'corner',
    6: 'corner',
    7: 'corner',
    8: 'corner',
    9: 'corner'
}



# for task_id in range(1,10):
task_id = 2
cam_view = 'corner'
seed = 6
def save_image(img_array):
    img = Image.fromarray(img_array)
    img.save(str(task_id) + '-' + tasks[task_id]+'-'+cam_view + '_obs.jpg')

def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

if seed is not None:
    st0 = np.random.get_state()
    np.random.seed(seed)
    np.random.set_state(st0)


env = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[tasks[task_id] + "-goal-hidden"]()
env._rgb_array_res = (230, 230)

# env = Monitor(env, './video', force=True)
obs_original = env.reset()  # Reset environment
action = env.action_space.sample()  # Sample an action
env.seed(0)
for i in range(200):  # env.max_path_length
    # a[3] = 0.1  # positive, finger close gradually; negative finger open gradually.
    action = np.array([-0.023, -0.011, -0.076, -0.073])

    obs, reward, done, info = env.step(copy.deepcopy(action))  # Step the environoment with the sampled random action

    # obs[0:4]: end effector x, y, z, gripper close or open distance.
    robot_state = env.get_env_state()  # robot_state[0].qpos, qvel
    # obs (39, 1)
    # obs1 = env.render(mode='human')
    obs2 = env.render(mode='rgb_array', camera_name=cam_view)  # # 'topview', 'corner', 'corner2' reverse, 'behindGripper', 'gripperPOV'
    obs2 = cropND(obs2, [128, 128])
    # viewer.imshow(obs2)
    print('step {}'.format(i+1))
    print('action')
    print(action)
    print(obs[0:4])
# save_image(obs2)
# self.sim.render(mode='window', camera_name='first-person', width=16, height=16, depth=False)
# img = self.sim.render(mode='offscreen', camera_name='first-person', width=16, height=16, depth=False)



# import robosuite as suite
# from robosuite.wrappers import GymWrapper
# from gym.envs.classic_control import rendering
#
# viewer = rendering.SimpleImageViewer()
#
# if __name__ == "__main__":
#
#     # Notice how the environment is wrapped by the wrapper
#     env = GymWrapper(
#         suite.make(
#             "Lift",
#             robots="Panda",                # use Sawyer robot
#             has_renderer=False,  # no on-screen renderer
#             has_offscreen_renderer=True,  # off-screen renderer is required for camera observations
#             ignore_done=True,  # (optional) never terminates episode
#             use_camera_obs=True,  # use camera observations
#             camera_heights=128,  # set camera height
#             camera_widths=128,  # set camera width
#             camera_names="agentview",  # use "agentview" camera
#             use_object_obs=False,  # no object feature when training on pixels
#             reward_shaping=True,  # (optional) using a shaping reward
#         )
#     )
#
#     for i_episode in range(20):
#         observation = env.reset()
#         for t in range(500):
#             # env.render() if using image observation, comment this line
#             action = env.action_space.sample()
#             observation, reward, done, info = env.step(action)
#             print(action)
#             viewer.imshow(observation["agentview_image"])
#             print('reward {}'.format(reward))
#             if done:
#                 print("Episode finished after {} timesteps".format(t + 1))
#                 break
#
#
