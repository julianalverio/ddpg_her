import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, o_stats, g_stats, dims):
        super(ActorCritic, self).__init__()
        self.actor = Actor(dims)
        self.critic = Critic(dims)

        self.o_stats = o_stats
        self.g_stats = g_stats
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_all(self, obs, goal, actions):
        obs = obs.to(self.device)
        goal = goal.to(self.device)
        actions = actions.to(self.device)
        obs = self.o_stats.normalize(obs)
        goal = self.g_stats.normalize(goal)
        obs = torch.tensor(obs)
        goal = torch.tensor(goal)
        policy_input = torch.cat([obs, goal], dim=1)
        policy_output = self.actor(policy_input)
        # temporary
        self.pi_tf = policy_output
        critic_input = torch.cat([obs, goal, policy_output], dim=1)
        self.q_pi_tf = self.critic(critic_input)
        actions = torch.tensor(actions)
        critic_input = torch.cat([obs, goal, actions], dim=1)
        self.q_tf = self.critic(critic_input)

    def get_action(self, obs, goals):
        obs = obs.to(self.device)
        goals = goals.to(self.device)
        obs = self.o_stats.normalize(obs)
        goals = self.g_stats.normalize(goals)
        obs = torch.tensor(obs)
        goals = torch.tensor(goals)
        policy_input = torch.cat([obs, goals], dim=1)
        return self.actor(policy_input)

    def compute_q_values(self, obs, goals, actions):
        obs = obs.to(self.device)
        goals = goals.to(self.device)
        actions = actions.to(self.device)
        obs = self.o_stats.normalize(obs)
        goals = self.g_stats.normalize(goals)
        obs = torch.tensor(obs)
        goals = torch.tensor(goals)
        input_tensor = torch.cat([obs, goals, actions], dim=1)
        return self.critic(input_tensor)


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
