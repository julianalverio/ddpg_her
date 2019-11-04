from mujoco_py import GlfwContext
GlfwContext(offscreen=True) # Create a window to init GLFW.

import sys
# sys.path.insert(0, '/storage/jalverio/gym/')
import gym
from common.cmd_util import make_vec_env
import os
import argparse
import torch
import shutil
from her.rollout import RolloutWorker
from mpi4py import MPI
import numpy as np
import random
from her.ddpg import DDPG
import GPUtil
from her.her_sampler import make_sample_her_transitions
from torch.utils.tensorboard import SummaryWriter
from cv2 import VideoWriter



PARAMS = {
    'lr': 0.001,
    'buffer_size': int(1E6),
    'polyak': 0.95,
    'clip_obs': 200.,
    'n_cycles': 10,
    'n_batches': 40,
    'batch_size': 256,
    'n_test_rollouts': 10,
    'random_eps': 0.3,
    'noise_eps': 0.2,
    'norm_eps': 0.01,
    'norm_clip': 5,
    'T': 50,
    'gamma': 0.98,
    'clip_return': 50.,
}


def set_seed(seed):
    if not seed:
        files = os.listdir('runs/')
        if not files:
            seed = 0
        else:
            seed = max([int(f.split('seed=')[1][0]) for f in files]) + 1
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed


# automatically choose the GPU
def choose_gpu(threshold=0.99):
    gpus = GPUtil.getGPUs()
    gpus = [gpu for gpu in gpus if gpu.load < threshold and gpu.memoryUtil < threshold]
    # gpus = sorted(gpus, key=lambda x: x.load)
    # gpu = gpus[0].id
    gpu = random.choice(gpus).id
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--env', help='environment ID', type=str, choices=['FetchReach-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1'])
    argparse.add_argument('--model_path', type=str)
    arg_parser.add_argument('--seed', type=int, default=None)
    args = arg_parser.parse_args()
    PARAMS['model_path'] = args.model_path
    return args


def get_dims(env):
    env_name = env.spec.id
    tmp_env = gym.make(env_name)
    tmp_env.reset()
    obs, _, _, info = tmp_env.step(env.action_space.sample())
    dims = {
        'o': obs['observation'].shape[0],
        'u': env.action_space.shape[0],
        'g': obs['desired_goal'].shape[0],
        'info_is_success': 1,
    }


    distance_threshold = tmp_env.env.distance_threshold
    PARAMS['distance_threshold'] = distance_threshold
    PARAMS['dims'] = dims

def write_video(frames, location):
    FPS = 5.0
    height, width = frames[0].shape[:2]
    video = VideoWriter('%s/fetch.avi' % location, 0, FPS, (width, height))
    for frame in frames:
        video.write(frame)
    video.release()


def evaluate(policy, env_name):
    env = gym.make(env_name)
    obs_dict = env.reset()
    import pdb; pdb.set_trace()
    obs = obs_dict['observation']
    goal = obs_dict['goal']
    env.render(mode='human')
    images = []
    images.append(env.render(mode='rgb_array'))
    for _ in range(50):
        action = policy.get_actions(obs, goal, 0, 0)
        obs_dict, _, _, _ = env.step(action)
        obs = obs_dict['observation']
        env.render(mode='human')
        images.append(env.render(mode='rgb_array'))
    import pdb; pdb.set_trace()


def main():
    import pdb; pdb.set_trace()
    choose_gpu()
    args = parse_args()
    seed = set_seed(args.seed)
    # env = make_vec_env(args.env, 'robotics', 1, seed=seed, reward_scale=1.0, flatten_dict_observations=False)
    env = gym.make(args.env)
    env.render()
    get_dims(env)
    PARAMS['sample_her_transitions'] = make_sample_her_transitions(PARAMS['distance_threshold'])

    policy = DDPG(PARAMS)
    evaluate(policy)




if __name__ == '__main__':
    main()