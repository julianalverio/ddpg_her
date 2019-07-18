import sys
import argparse
from baselines.common.cmd_util import make_vec_env
from baselines.her.her import learn


def build_env(args):
    seed = 0
    return make_vec_env(args.env, 'robotics', args.num_env, seed, reward_scale=1.0, flatten_dict_observations=False)


def main(args):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    arg_parser.add_argument('--num_timesteps', type=float, default=1e6),
    arg_parser.add_argument('--num_env', default=None, type=int)
    args, _ = arg_parser.parse_known_args(args)
    env = build_env(args)
    learn(env=env, total_timesteps=int(args.num_timesteps))


if __name__ == '__main__':
    main(sys.argv)