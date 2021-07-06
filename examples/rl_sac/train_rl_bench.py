#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils
import random

import hydra
from copy import deepcopy
from scipy.io import savemat
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode, GripperActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
import numpy as np
import operator

"""
observation types
obs.front_depth                 obs.gripper_joint_positions     obs.joint_forces                obs.left_shoulder_point_cloud   obs.overhead_point_cloud        obs.right_shoulder_rgb          obs.wrist_rgb
obs.front_mask                  obs.gripper_matrix              obs.joint_positions             obs.left_shoulder_rgb           obs.overhead_rgb                obs.task_low_dim_state          
obs.front_point_cloud           obs.gripper_open                obs.joint_velocities            obs.misc                        obs.right_shoulder_depth        obs.wrist_depth                 
obs.front_rgb                   obs.gripper_pose                obs.left_shoulder_depth         obs.overhead_depth              obs.right_shoulder_mask         obs.wrist_mask                  
obs.get_low_dim_data(           obs.gripper_touch_forces        obs.left_shoulder_mask          obs.overhead_mask               obs.right_shoulder_point_cloud  obs.wrist_point_cloud  
"""

tasks = {
    '1': ReachTarget,
    '2': RemoveCups,
    '3': OpenBox,
    '4': MoveHanger
}

obs_config = ObservationConfig()
obs_config.front_camera.set_all(True)
observation_space = [128, 128, 3]
# obs_type_name = 'front_rgb'

gripper_params = {'control': False, 'always_open':True}
action_dim = 8 if gripper_params['control'] else 7
action_range = [
    -0.5,  # 0.13
    0.5
]


rets, rets_ste, num_steps, eval_step_index, eval_returns = [], [], [], [], []

WORK_DIR = None

def sample_random_action():
    if gripper_params['control']:
        arm = np.random.normal(action_range[0], action_range[1], size=(action_dim - 1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)
    else:
        return np.random.normal(action_range[0], action_range[1], size=(action_dim,))

def wrap_action_with_gripper(action):
    if not gripper_params['control']:
        gripper = [1.0] if gripper_params['always_open'] else 0.0
        return np.concatenate([action, gripper], axis=-1)

def parse_obs(obs, obs_type_name):
    return operator.attrgetter(obs_type_name)(obs)

def save_eval_mat(log_name):
    savemat(
        WORK_DIR + '/' + log_name + ".mat",
        {
            "test_returns": rets,
            "test_returns_std_err": rets_ste,
            'num_steps': num_steps,
            'eval_step_index': eval_step_index,
            'eval_returns': eval_returns
        }
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # tf.set_random_seed(seed)
    torch.manual_seed(seed)
    print("Using seed {}".format(seed))


def make_env(cfg):
    """Helper function to create dm_control environment"""
    set_seed(cfg.seed)

    action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
    env = Environment(
        action_mode, obs_config=obs_config, headless=cfg.headless)
    env.launch()
    task = env.get_task(tasks[str(cfg.taskid)])
    print(task)
    print('action dim: ')
    print(env.action_size)
    return task, env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        global WORK_DIR
        WORK_DIR = self.work_dir
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             save_frequency=cfg.save_frequency)
                             # agent=cfg.agent.name)
        training_seed = cfg.seed
        print('env {}, seed {}'.format(cfg.taskid, cfg.seed))
        set_seed(training_seed)
        self.device = torch.device(cfg.device)
        self.env, _ = make_env(cfg)
        # use the same training env seed
        # self.env.seed(1234)
        self.eval_env = self.env  # make_env(cfg, _headless=True)  # libcoppeliaSim does not support multiple threads
        # use the same evaluation env seed
        # self.eval_env.seed(0)

        action_space = (action_dim, )

        # cfg.agent.params.obs_dim = observation_space  # this is the final layer's obs_dim, set inside the agent.yaml
        cfg.agent.params.action_dim = action_dim
        cfg.agent.params.action_range = action_range
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(observation_space,
                                          action_space,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        test_rets = []
        steps_in_this_episode = []
        for episode in range(self.cfg.num_eval_episodes):
            _, obs = self.eval_env.reset()
            obs = parse_obs(obs, self.cfg.obs_key)
            # self.agent.reset()
            # self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            i_step = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done = self.eval_env.step(wrap_action_with_gripper(action))
                done = True if episode_step + 1 == self.cfg.max_episode_steps else False
                print('Eval step {}, episode {}'.format(i_step, episode))
                print(action)
                obs = parse_obs(obs, self.cfg.obs_key)
                i_step += 1
                # self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            test_rets.append(episode_reward)
            steps_in_this_episode.append(i_step)
            # self.video_recorder.save(f'{self.step}.mp4')

        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/episode_reward_std', np.std(test_rets) / np.sqrt(5),
                        self.step)
        self.logger.dump(self.step, ty='eval')
        eval_returns.append(test_rets)
        rets.append(np.mean(test_rets))
        rets_ste.append(np.std(test_rets) / np.sqrt(5))
        num_steps.append(steps_in_this_episode)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        print('num of train steps {}'.format(self.cfg.num_train_steps))
        while self.step < self.cfg.num_train_steps:
            if done:
                print('done true')
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps), ty='train')

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                self.logger.save_check_point(self.agent.actor, self.step)

                _, obs = self.env.reset()
                obs = parse_obs(obs, self.cfg.obs_key)
                # self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # evaluate agent periodically
            if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                self.logger.log('eval/episode', episode, self.step)
                self.evaluate()
                eval_step_index.append(self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = sample_random_action()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)
            print('train step {}'.format(self.step))
            print(action)
            next_obs, reward, done = self.env.step(wrap_action_with_gripper(action))
            done = True if episode_step + 1 == self.cfg.max_episode_steps else False
            next_obs = parse_obs(next_obs, self.cfg.obs_key)
            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.cfg.max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1

        save_eval_mat(self.cfg.experiment)


@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
