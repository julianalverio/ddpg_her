from collections import OrderedDict
import numpy as np
from her.normalizer import Normalizer
from her.replay_buffer import ReplayBuffer
from common.mpi_adam_torch import MpiAdam as MpiAdamTorch
from her.actor_critic import ActorCritic

import torch
import copy
import torch.optim as optim


class DDPG(object):
    def __init__(self, params):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
        """

        self.input_dims = params['dims']
        self.buffer_size = params['buffer_size']
        self.polyak = params['polyak']
        self.batch_size = params['batch_size']
        self.Q_lr = params['lr']
        self.pi_lr = params['lr']
        self.norm_eps = params['norm_eps']
        self.norm_clip = params['norm_clip']
        self.clip_obs = params['clip_obs']
        self.T = params['T']
        self.rollout_batch_size = params['num_envs']
        self.clip_return = params['clip_return']
        self.sample_transitions = params['sample_her_transitions']
        self.gamma = params['gamma']

        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, self.input_dims[key])
        stage_shapes['o_2'] = stage_shapes['o']
        stage_shapes['r'] = (None,)
        self.stage_shapes = stage_shapes

        self.torch_create_network()

        # Configure the replay buffer.
        buffer_shapes = {key: (self.T-1 if key != 'o' else self.T, self.input_dims[key])
                         for key, val in self.input_dims.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T, self.dimg)

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)

    def torch_random_action(self, n):
        return torch.tensor(np.random.uniform(low=-1., high=1., size=(n, self.dimu)).astype(np.float32))

    def get_actions(self, o, g, noise_eps=0., random_eps=0.):
        actions = self.main.get_action(o, g)

        noise = (noise_eps * np.random.randn(actions.shape[0], 4)).astype(np.float32)
        actions += torch.tensor(noise)

        actions = torch.clamp(actions, -1., 1.)
        eps_greedy_noise = np.random.binomial(1, random_eps, actions.shape[0]).reshape(-1, 1)
        random_action = self.torch_random_action(actions.shape[0])
        actions += torch.tensor(eps_greedy_noise.astype(np.float32)) * (
                    random_action - actions)  # eps-greedy
        return actions


    def store_episode(self, episode_batch):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """
        self.buffer.store_episode(episode_batch)

        # add transitions to normalizer
        episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
        episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
        shape = episode_batch['u'].shape
        num_normalizing_transitions = shape[0] * shape[1]  # num_rollouts * (rollout_horizon - 1) --> total steps per cycle
        transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

        self.o_stats.update(transitions['o'])
        self.g_stats.update(transitions['g'])

        self.o_stats.recompute_stats()
        self.g_stats.recompute_stats()

    def sample_batch(self):
        transitions = self.buffer.sample(self.batch_size)
        return [transitions[key] for key in self.stage_shapes.keys()]

    def torch_train(self):
        batch = self.sample_batch()
        batch_dict = OrderedDict([(key, batch[i].astype(np.float32).copy())
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_dict['r'] = np.reshape(batch_dict['r'], [-1, 1])

        main_batch = batch_dict
        target_batch = batch_dict.copy()
        target_batch['o'] = batch_dict['o_2']

        self.main.compute_all(main_batch['o'], main_batch['g'],
                              main_batch['u'])
        self.target.compute_all(target_batch['o'], target_batch['g'],
                                target_batch['u'])

        # Q function loss
        rewards = torch.tensor(main_batch['r'].astype(np.float32))
        discounted_reward = self.gamma * self.target.q_pi_tf
        target_tf = torch.clamp(rewards + discounted_reward, -self.clip_return, 0.)
        q_loss_tf = torch.nn.MSELoss()(target_tf.detach(), self.main.q_tf)

        # STEP MPI ADAM OPTIMIZER FOR CRITIC
        self.critic_optimizer_util.zero_grad()
        q_loss_tf.backward()
        self.critic_optimizer.update()

        # policy iteration loss
        pi_loss_tf = -self.main.q_pi_tf.mean()
        pi_loss_tf += (self.main.pi_tf ** 2).mean()

        # STEP MPI ADAM OPTIMIZER FOR ACTOR
        self.actor_optimizer_util.zero_grad()
        pi_loss_tf.backward()
        self.actor_optimizer.update()

    # TODO: use load_state_dict here
    def torch_update_target_net(self):
        beta = 1. - self.polyak

        # actor
        self.target.actor.linear1.weight = torch.nn.Parameter(beta * self.main.actor.linear1.weight + self.polyak * self.target.actor.linear1.weight)
        self.target.actor.linear1.bias = torch.nn.Parameter(beta * self.main.actor.linear1.bias + self.polyak * self.target.actor.linear1.bias)
        self.target.actor.linear2.weight = torch.nn.Parameter(beta * self.main.actor.linear2.weight + self.polyak * self.target.actor.linear2.weight)
        self.target.actor.linear2.bias = torch.nn.Parameter(beta * self.main.actor.linear2.bias + self.polyak * self.target.actor.linear2.bias)
        self.target.actor.linear3.weight = torch.nn.Parameter(beta * self.main.actor.linear3.weight + self.polyak * self.target.actor.linear3.weight)
        self.target.actor.linear3.bias = torch.nn.Parameter(beta * self.main.actor.linear3.bias + self.polyak * self.target.actor.linear3.bias)
        self.target.actor.linear4.weight = torch.nn.Parameter(beta * self.main.actor.linear4.weight + self.polyak * self.target.actor.linear4.weight)
        self.target.actor.linear4.bias = torch.nn.Parameter(beta * self.main.actor.linear4.bias + self.polyak * self.target.actor.linear4.bias)

        # critic
        self.target.critic.linear1.weight = torch.nn.Parameter(beta * self.main.critic.linear1.weight + self.polyak * self.target.critic.linear1.weight)
        self.target.critic.linear1.bias = torch.nn.Parameter(beta * self.main.critic.linear1.bias + self.polyak * self.target.critic.linear1.bias)
        self.target.critic.linear2.weight = torch.nn.Parameter(beta * self.main.critic.linear2.weight + self.polyak * self.target.critic.linear2.weight)
        self.target.critic.linear2.bias = torch.nn.Parameter(beta * self.main.critic.linear2.bias + self.polyak * self.target.critic.linear2.bias)
        self.target.critic.linear3.weight = torch.nn.Parameter(beta * self.main.critic.linear3.weight + self.polyak * self.target.critic.linear3.weight)
        self.target.critic.linear3.bias = torch.nn.Parameter(beta * self.main.critic.linear3.bias + self.polyak * self.target.critic.linear3.bias)
        self.target.critic.linear4.weight = torch.nn.Parameter(beta * self.main.critic.linear4.weight + self.polyak * self.target.critic.linear4.weight)
        self.target.critic.linear4.bias = torch.nn.Parameter(beta * self.main.critic.linear4.bias + self.polyak * self.target.critic.linear4.bias)

    def torch_create_network(self):
        # for actor network
        self.o_stats = Normalizer(size=self.dimo, eps=self.norm_eps, default_clip_range=self.norm_clip)
        self.g_stats = Normalizer(size=self.dimg, eps=self.norm_eps, default_clip_range=self.norm_clip)

        self.main = ActorCritic(self.o_stats, self.g_stats)
        self.target = ActorCritic(self.o_stats, self.g_stats)
        self.target.actor = copy.deepcopy(self.main.actor)
        self.target.critic = copy.deepcopy(self.main.critic)

        # use MPI_ADAM instead
        self.actor_optimizer_util = optim.Adam(self.main.actor.parameters(), lr=self.pi_lr)
        self.critic_optimizer_util = optim.Adam(self.main.critic.parameters(), lr=self.Q_lr)
        self.actor_optimizer = MpiAdamTorch(self.main.actor, self.pi_lr)
        self.critic_optimizer = MpiAdamTorch(self.main.critic, self.Q_lr)
        self.actor_optimizer.sync()
        self.critic_optimizer.sync()
