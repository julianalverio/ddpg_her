import numpy as np
from mpi4py import MPI
from her.rollout import RolloutWorker
import gym
from her.her_sampler import make_sample_her_transitions
from her.ddpg import DDPG


def train(policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches):
    for epoch in range(n_epochs):
        print('epoch:', epoch+1, 'of', n_epochs)
        for _ in range(n_cycles):
            episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode)
            for _ in range(n_batches):
                policy.torch_train()
            policy.torch_update_target_net()

        # test
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # make sure that different threads have different seeds
        MPI.COMM_WORLD.Bcast(np.random.uniform(size=(1,)), root=0)


def learn(env, total_timesteps):
    #ddpg
    Q_lr=0.001
    pi_lr = 0.001
    buffer_size = int(1E6)
    polyak=0.95
    clip_obs=200.
    #training
    n_cycles = 10
    n_batches = 40
    batch_size = 256
    n_test_rollouts = 10
    #exploration
    random_eps = 0.3
    noise_eps = 0.2
    #normalization
    norm_eps = 0.01
    norm_clip = 5
    # others
    env_name = env.spec.id
    tmp_env = gym.make(env_name)
    T = 50  # this is hard-coded to 50 in all fetch envs
    gamma = 0.98
    rollout_batch_size = env.num_envs
    clip_return = 50.

    # get dimensions
    tmp_env.reset()
    distance_threshold = tmp_env.env.distance_threshold
    obs, _, _, info = tmp_env.step(env.action_space.sample())

    dims = {
        'o': obs['observation'].shape[0],
        'u': env.action_space.shape[0],
        'g': obs['desired_goal'].shape[0],
        'info_is_success': 1
    }

    sample_her_transitions = make_sample_her_transitions(distance_threshold)


    policy=DDPG(input_dims=dims, buffer_size=buffer_size, polyak=polyak, batch_size=batch_size, Q_lr=Q_lr,
                pi_lr=pi_lr, norm_eps=norm_eps, norm_clip=norm_clip, clip_obs=clip_obs, T=T,
                rollout_batch_size=rollout_batch_size, clip_return=clip_return,
                sample_transitions=sample_her_transitions, gamma=gamma)

    rollout_worker = RolloutWorker(venv=env, policy=policy, dims=dims, rollout_batch_size=rollout_batch_size,
                                    noise_eps=noise_eps, random_eps=random_eps, T=T, clip_obs=clip_obs)
    evaluator = RolloutWorker(venv=env, policy=policy, dims=dims, rollout_batch_size=rollout_batch_size,
                              noise_eps=0., random_eps=0., T=T, clip_obs=clip_obs)

    n_epochs = total_timesteps // n_cycles // rollout_worker.T // rollout_worker.rollout_batch_size
    train(policy=policy, rollout_worker=rollout_worker,
          evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=n_test_rollouts,
          n_cycles=n_cycles, n_batches=n_batches)
