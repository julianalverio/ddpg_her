import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, o_stats, g_stats):
        super(ActorCritic, self).__init__()
        self.actor = Actor()
        self.critic = Critic()

        self.o_stats = o_stats
        self.g_stats = g_stats
        self.pi_tf = None
        self.q_pi_tf = None
        self.q_tf = None

    def compute_all(self, obs, goal, actions):
        self.pre_normalized_stats = obs
        obs = self.o_stats.normalize(obs)
        goal = self.g_stats.normalize(goal)
        obs = torch.tensor(obs)
        goal = torch.tensor(goal)
        self.critic_input_o = obs
        self.critic_input_g = goal
        policy_input = torch.cat([obs, goal], dim=1)
        self.policy_input = policy_input
        policy_output = self.actor(policy_input)
        # temporary
        self.policy_output = policy_output
        self.pi_tf = policy_output
        critic_input = torch.cat([obs, goal, policy_output], dim=1)
        self.critic_input = critic_input
        self.q_pi_tf = self.critic(critic_input)
        actions = torch.tensor(actions)
        critic_input = torch.cat([obs, goal, actions], dim=1)
        self.final_critic_input = critic_input
        self.q_tf = self.critic(critic_input)

    def get_action(self, obs, goals):
        obs = self.o_stats.normalize(obs)
        goals = self.g_stats.normalize(goals)
        obs = torch.tensor(obs)
        goals = torch.tensor(goals)
        policy_input = torch.cat([obs, goals], dim=1)
        return self.actor(policy_input)

    def compute_q_values(self, obs, goals, actions):
        obs = self.o_stats.normalize(obs)
        goals = self.g_stats.normalize(goals)
        obs = torch.tensor(obs)
        goals = torch.tensor(goals)
        input_tensor = torch.cat([obs, goals, actions], dim=1)
        return self.critic(input_tensor)


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(in_features=13, out_features=256)
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
    def __init__(self):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(in_features=17, out_features=256)
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
