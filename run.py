# from mujoco_py import GlfwContext
# GlfwContext(offscreen=True) # Create a window to init GLFW.

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
import cv2


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
    arg_parser.add_argument('--model_dir', type=str, default='')
    arg_parser.add_argument('--env', help='environment ID', type=str, choices=['FetchReach-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1'])
    arg_parser.add_argument('--num_timesteps', type=float, default=1e7),
    arg_parser.add_argument('--num_envs', default=25, type=int)
    arg_parser.add_argument('--seed', type=int, default=None)
    args = arg_parser.parse_args()
    PARAMS['num_envs'] = args.num_envs
    PARAMS['num_timesteps'] = args.num_timesteps
    PARAMS['model_dir'] = args.model_dir
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


def train(policy, rollout_worker, evaluator, writer):
    n_epochs = int(PARAMS['num_timesteps'] // PARAMS['n_cycles'] // PARAMS['T'] // PARAMS['num_envs'])
    for epoch in range(n_epochs):
        print('epoch:', epoch, 'of', n_epochs)
        for _ in range(PARAMS['n_cycles']):
            episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode)
            for _ in range(PARAMS['n_batches']):
                policy.torch_train()
            policy.torch_update_target_net()

        # test
        test_scores = []
        for _ in range(PARAMS['n_test_rollouts']):
            evaluator.generate_rollouts()
            test_scores.append(evaluator.mean_success)
        writer.add_scalar('score', np.mean(test_scores), epoch)
        print('epoch %s: %s' % (epoch, np.mean(test_scores)))

        # make sure that different threads have different seeds
        MPI.COMM_WORLD.Bcast(np.random.uniform(size=(1,)), root=0)

        if epoch % 50 == 0:
            policy.main.save(epoch, np.mean(test_scores))

def write_video(frames):
    FPS = 5.0
    height, width = frames[0].shape[:2]
    # video = VideoWriter(location, 0, FPS, (width, height))
    video = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (500, 500))
    for frame in frames:
        video.write(frame)
    video.release()

def make_videos(model_dir, policy, env_name):
    policy.main.load_models(model_dir)
    env = gym.make(env_name)

    obs_dict = env.reset()
    obs = obs_dict['observation']
    obs = np.expand_dims(obs, axis=0)
    goal = obs_dict['desired_goal']
    goal = np.expand_dims(goal, axis=0)
    frames = []
    for _ in range(50):
        actions = np.squeeze(policy.get_actions(obs, goal, 0, 0).cpu().detach().numpy())
        obs_dict_new, _, done, info = env.step(actions)
        obs = obs_dict_new['observation']
        obs = np.expand_dims(obs, axis=0)
        frames.append(env.render(mode='rgb_array'))
    write_video(frames)
    print('GO BACK AND RENAME THE VIDEO!! D:<')
    import pdb; pdb.set_trace()
    import sys; sys.exit()


def main():
    choose_gpu()
    args = parse_args()
    seed = set_seed(args.seed)
    env = make_vec_env(args.env, 'robotics', args.num_envs, seed=seed, reward_scale=1.0, flatten_dict_observations=False)
    env.get_images()
    seed = set_seed(args.seed)
    get_dims(env)
    PARAMS['sample_her_transitions'] = make_sample_her_transitions(PARAMS['distance_threshold'])
    PARAMS['log_dir'] = 'runs/env=%s_seed=%s' % (args.env, seed)
    shutil.rmtree(PARAMS['log_dir'], ignore_errors=True)
    print('logging to:', PARAMS['log_dir'])
    writer = SummaryWriter(PARAMS['log_dir'])

    policy = DDPG(PARAMS)
    if PARAMS['model_dir']:
        make_videos(PARAMS['model_dir'], policy, args.env)
    else:
        rollout_worker = RolloutWorker(env, policy, PARAMS)
        evaluator = RolloutWorker(env, policy, PARAMS, evaluate=True)
        train(policy, rollout_worker, evaluator, writer)



if __name__ == '__main__':
    main()
