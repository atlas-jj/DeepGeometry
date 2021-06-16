from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
import numpy as np
import tools.seeding as seeding
import torchvision.transforms as T
transform = T.Compose([
# T.ToPILImage(),
T.ToTensor(),
# T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

tasks = {
    '1': ReachTarget,
    '2': RemoveCups,
    '3': OpenBox
}
obs_type_name='front_rgb'
def parse_obs(obs):
    return operator.attrgetter(obs_type_name)(obs)

class Agent(object):

    def __init__(self, action_size):
        self.action_size = action_size

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_size - 1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)


obs_config = ObservationConfig()
obs_config.front_camera.set_all(True)
# obs_config.set_all(True)

action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
env = Environment(
    action_mode, obs_config=obs_config, headless=True)
env.launch()

task = env.get_task(ReachTarget)
task.np_random = seeding.np_random(1)
agent = Agent(env.action_size)

training_steps = 20
episode_length = 400
obs = None
descriptions, obs = task.reset()
# for i in range(training_steps):
#     if i % episode_length == 0:
#         print('Reset Episode')
#         descriptions, obs = task.reset()
#         print(descriptions)
#     action = agent.act(obs)
#     print(action)
#     obs, reward, terminate = task.step(action)
#     print(reward)
#
# print('Done')
# # env.shutdown()
