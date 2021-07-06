import numpy as np
import torch
import torchvision.transforms as T
from collections import namedtuple
import copy
import random

class ClumsyReplayBuffer(object):
    """Buffer to store environment transitions."""
    """ This is quite a bad implementation from the authors"""
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False


    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)

        # obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        # actions = torch.as_tensor(self.actions[idxs], device=self.device)
        # rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        # next_obses = torch.as_tensor(self.next_obses[idxs],
        #                              device=self.device).float()
        # not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        # not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
        #                                    device=self.device)
        return self.obses[idxs], self.actions[idxs], self.rewards[idxs], self.next_obses[idxs], self.not_dones[idxs], self.not_dones_no_max[idxs]
        # print(self.obses[idxs].shape)
        # obses = self.transform(self.obses[idxs]).float().to(self.device)
        # actions = self.transform(self.actions[idxs]).float().to(self.device)
        # rewards = self.transform(self.rewards[idxs]).float().to(self.device)
        # next_obses = self.transform(self.next_obses[idxs]).float().to(self.device)
        # not_dones = self.transform(self.not_dones[idxs]).float().to(self.device)
        # not_dones_no_max = self.transform(self.not_dones_no_max[idxs]).float().to(self.device)
        # return obses, actions, rewards, next_obses, not_dones, not_dones_no_max


class ReplayBuffer(object):
    def __init__(self, capacity, seed=None):
        assert capacity > 0, "Capacity must be a positive integer"
        self.capacity = capacity
        self.data = {}

        self.write_index = -1
        self.buffer_size = 0
        self.seed = seed
        # self.transform = T.Compose([
        #     T.ToPILImage(),
        #     T.ToTensor(),
        #     T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # ])
        if self.seed is not None:
            np.random.seed(seed)

    def add(self, transition):
        self.write_index += 1
        if self.write_index >= self.capacity:
            self.write_index = 0
        if self.buffer_size < self.capacity:
            self.buffer_size += 1
        else:
            self.buffer_size = self.capacity

        self.data[self.write_index] = transition

    def sample(self, batchsize, replacement=True):
        indices = np.random.choice(self.buffer_size, batchsize, replace=replacement)
        len_transition = len(self.data[0])
        batch = []
        for j in range(len_transition):
            batch.append([self.data[idx][j] for idx in indices])
        # batch = [self.data[idx] for idx in indices]
        return copy.deepcopy(batch)

    def __len__(self):
        return self.buffer_size

    def __iter__(self):
        self.iter_idx = -1
        return self

    def __next__(self):
        self.iter_idx += 1
        if self.iter_idx < self.buffer_size:
            return self.data[self.iter_idx]
        else:
            raise StopIteration()

    def next(self):
        return self.__next__()


#
#
# # Git from
# # https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb
# # modified for ssw-batman project, Jun Jin, June 24, 14:49:00 MDT 2019
# # robot_state, observation, action, reward, mask, next_robot_state, next_observation
# Transition = namedtuple(
#     'Transition', ('robot_state', 'observation', 'action', 'mask', 'reward', 'next_robot_state', 'next_observation'))
#
# class ReplayMemory(object):
#
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0
#         self.transform = T.Compose([
#             T.ToPILImage(),
#             T.ToTensor(),
#             T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#         ])
#
#     def push(self, *args):
#         """Saves a transition."""
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = Transition(*args)
#         self.position = (self.position + 1) % self.capacity
#
#     def sample(self, batch_size):
#         indices = np.random.choice(self.buffer_size, batchsize, replace=replacement)
#         batch = [self.data[idx] for idx in indices]
#
#         return indices, batch
#         return random.sample(self.memory, batch_size)
#
#     def __len__(self):
#         return len(self.memory)
