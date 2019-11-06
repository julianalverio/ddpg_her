import torch
import torch.nn as nn
import pickle
import os
import copy


class ActorCritic(nn.Module):
    def __init__(self, o_stats, g_stats, dims):
        super(ActorCritic, self).__init__()
        self.actor = Actor(dims)
        self.critic = Critic(dims)

        self.o_stats = o_stats
        self.g_stats = g_stats
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_all(self, obs, goal, actions):
        obs = self.o_stats.normalize(obs)
        goal = self.g_stats.normalize(goal)
        obs = torch.tensor(obs).to(self.device)
        goal = torch.tensor(goal).to(self.device)
        policy_input = torch.cat([obs, goal], dim=1)
        policy_output = self.actor(policy_input)
        # temporary
        self.pi_tf = policy_output
        critic_input = torch.cat([obs, goal, policy_output], dim=1)
        self.q_pi_tf = self.critic(critic_input)
        actions = torch.tensor(actions).to(self.device)
        critic_input = torch.cat([obs, goal, actions], dim=1)
        self.q_tf = self.critic(critic_input)

    def get_action(self, obs, goals):
        obs = self.o_stats.normalize(obs)
        goals = self.g_stats.normalize(goals)
        obs = torch.tensor(obs).to(self.device)
        goals = torch.tensor(goals).to(self.device)
        policy_input = torch.cat([obs, goals], dim=1)
        return self.actor(policy_input)

    def compute_q_values(self, obs, goals, actions):
        obs = self.o_stats.normalize(obs)
        goals = self.g_stats.normalize(goals)
        obs = torch.tensor(obs).to(self.device)
        goals = torch.tensor(goals).to(self.device)
        input_tensor = torch.cat([obs, goals, actions], dim=1)
        return self.critic(input_tensor)

    def load_models(self, model_path):
        model_path = os.path.join('/storage/jalverio/ddpg_her/models/', model_path)
        actor_path = os.path.join(model_path, 'actor.pkl')
        critic_path = os.path.join(model_path, 'critic.pkl')
        with open(actor_path, 'rb') as f:
            self.actor.load_state_dict(pickle.load(f))
        with open(critic_path, 'rb') as f:
            self.critic.load_state_dict(pickle.load(f))

        # now load the normalizers
        o_stats_path = os.path.join(model_path, 'o_stats.pkl')
        g_stats_path = os.path.join(model_path, 'g_stats.pkl')
        with open(o_stats_path, 'rb') as f:
            self.o_stats = pickle.load(f)
        with open(g_stats_path, 'rb') as f:
            self.g_stats = pickle.load(f)

    def save(self, episode, score):
        base_path = '/storage/jalverio/ddpg_her/models/'
        path = 'episode%s_score%s'% (episode, score)
        save_dir = os.path.join(base_path, path)
        try:
            os.mkdir(save_dir)
        except:
            pass
        actor_path = os.path.join(save_dir, 'actor.torch')
        critic_path = os.path.join(save_dir, 'critic.torch')
        torch.save(self.actor, actor_path)
        torch.save(self.critic, critic_path)
        o_stats_path = os.path.join(save_dir, 'o_stats.pkl')
        g_stats_path = os.path.join(save_dir, 'g_stats.pkl')
        o_stats_lock = copy.deepcopy(self.o_stats.lock)
        self.o_stats.lock = None
        with open(o_stats_path, 'wb') as f:
            pickle.dump(self.o_stats, f)
        self.o_stats.lock = o_stats_lock
        g_stats_lock = copy.deepcopy(self.g_stats.lock)
        self.g_stats.lock = None
        with open(g_stats_path, 'wb') as f:
            pickle.dump(self.g_stats, f)
        self.g_stats.lock = g_stats_lock




class Actor(nn.Module):
    def __init__(self, dims):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(in_features=dims['o'] + dims['g'], out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=256)
        self.linear3 = nn.Linear(in_features=256, out_features=256)
        self.linear4 = nn.Linear(in_features=256, out_features=4)
        self.actor = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            self.linear2,
            nn.ReLU(),
            self.linear3,
            nn.ReLU(),
            self.linear4,
            nn.Tanh()
        )

    def forward(self, input_tensor):
        return self.actor(input_tensor)


class Critic(nn.Module):
    def __init__(self, dims):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(in_features=dims['o'] + dims['g'] + dims['u'], out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=256)
        self.linear3 = nn.Linear(in_features=256, out_features=256)
        self.linear4 = nn.Linear(in_features=256, out_features=1)
        self.critic = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            self.linear2,
            nn.ReLU(),
            self.linear3,
            nn.ReLU(),
            self.linear4,
        )

    def forward(self, input_tensor):
        return self.critic(input_tensor)
