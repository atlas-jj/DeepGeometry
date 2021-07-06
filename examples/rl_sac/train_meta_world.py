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
from PIL import Image
import metaworld
import random
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN
from gym.wrappers import Monitor
import numpy as np
# from gym.envs.classic_control import rendering
from colorama import Fore, Back, Style

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

rets, rets_ste, num_steps, eval_step_index, eval_returns = [], [], [], [], []

WORK_DIR = None

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
    env = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[tasks[cfg.taskid] + "-goal-hidden"]()
    env._rgb_array_res = (230, 230)
    return env

def make_observation(robot_states, env, task_id, viewer=None):
    img = env.render(mode='rgb_array', camera_name=camera_views[task_id])
    img = utils.cropND(img, [128, 128])
    if viewer is not None:
        viewer.imshow(img)
    # proprio_inputs end effector x, y, z, gripper close or open distance.
    return {'proprios': robot_states[0: 4], 'image': img}

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
        self.env = make_env(cfg)
        # use the same training env seed
        self.env.seed(1234)
        self.eval_env = make_env(cfg)  # make_env(cfg, _headless=True)  # libcoppeliaSim does not support multiple threads
        # use the same evaluation env seed
        self.eval_env.seed(0)
        action_dim = self.env.action_space.shape[0]
        # cfg.agent.params.obs_dim = observation_space  # this is the final layer's obs_dim, set inside the agent.yaml
        cfg.agent.params.action_dim = action_dim
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(int(cfg.replay_buffer_capacity))

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0
        if not cfg.headless:
            self.viewer = rendering.SimpleImageViewer()
        else:
            self.viewer = None

    def evaluate(self):
        average_episode_reward = 0
        test_rets = []
        steps_in_this_episode = []
        for episode in range(self.cfg.num_eval_episodes):
            true_state = self.eval_env.reset()
            obs = make_observation(true_state, self.eval_env, self.cfg.taskid, self.viewer)
            # self.agent.reset()
            # self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            i_step = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                true_state, reward, done, info = self.eval_env.step(action)
                obs = make_observation(true_state, self.eval_env, self.cfg.taskid, self.viewer)
                done = True if i_step + 1 == self.cfg.max_episode_steps else done
                i_step += 1
                # self.video_recorder.record(self.env)
                episode_reward += reward
                print(Fore.RED + 'EVAL' + Style.RESET_ALL, end='')
                print(' step {:6d}, episode: {:1d}/{:1d}, '.format(self.step, episode+1, self.cfg.num_eval_episodes,), end='')
                print('i_step {:3d}/{:3d}, action=['.format(i_step, self.cfg.max_episode_steps), end='')
                for a in action:
                    print('{:+1.3f} '.format(a), end='')
                print('], state=[{:+1.3f},{:+1.3f},{:+1.3f},{:+1.3f}]'.format(*obs['proprios']), end='')
                print(', reward {:+1.3f}, done {:1.0f}'.format(reward, done))
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

                true_state = self.env.reset()
                obs = make_observation(true_state, self.env, self.cfg.taskid, self.viewer)
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
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(copy.deepcopy(obs), sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps and self.step % self.cfg.agent_update_frequency == 0:
                self.agent.update(self.replay_buffer, self.logger, self.step)
            next_true_state, reward, done, info = self.env.step(action)

            next_obs = make_observation(next_true_state, self.env, self.cfg.taskid, self.viewer)
            done = True if episode_step + 1 == self.cfg.max_episode_steps else done
            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.cfg.max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add([obs, action, reward, next_obs, done, done_no_max])
            obs = next_obs
            episode_step += 1
            self.step += 1
            print(Fore.GREEN + 'TRAIN' + Style.RESET_ALL, end='')
            print(' step {:6d}, i_step {:3d}/{:3d}, action=['.format(self.step, episode_step, self.cfg.max_episode_steps), end='')
            for a in action:
                print('{:+1.3f} '.format(a), end='')
            print('], state=[{:+1.3f},{:+1.3f},{:+1.3f},{:+1.3f}]'.format(*obs['proprios']), end='')
            print(', reward {:+1.3f}, done {:1.0f}'.format(reward, done))
        save_eval_mat(self.cfg.experiment)


@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
