import os
import logging
import random
import copy
from collections import deque, namedtuple
import time

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# parameters
actor_lr = 1e-4
critic_lr = 1e-3
weight_decay = 0  # L2 weight decay
epsilon = 1.0
epsilon_decay = 1e-6
learning_period = 20  # learning frequency
update_factor = 10   # update times


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):

    def __init__(self, input_size, output_size, seed, hidden_size_1=128, hidden_size_2=128):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, output_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):

    def __init__(self, input_size, output_size, seed, hidden_size_1=128, hidden_size_2=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1 + output_size, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        # (state, action) pairs -> Q-values
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.state = copy.copy(self.mu)

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.transition = namedtuple("Transitions", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        t = self.transition(state, action, reward, next_state, done)
        self.memory.append(t)

    def sample(self):
        transitions = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([t.state for t in transitions if t is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([t.action for t in transitions if t is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([t.reward for t in transitions if t is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([t.next_state for t in transitions if t is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([t.done for t in transitions if t is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class AgentDDPG(Agent):
    def __init__(self, env, args, random_seed):
        super(AgentDDPG, self).__init__(env)
        self.env = env
        self.args = args
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.seed = random.seed(random_seed)
        self.epsilon = epsilon
        self.mean_100_reward_bound = self.args.reward

        # actor network
        self.local_actor = Actor(self.state_size, self.action_size, random_seed).to(device)
        self.target_actor = Actor(self.state_size, self.action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.local_actor.parameters(), lr=actor_lr)

        # critic network
        self.local_critic = Critic(self.state_size, self.action_size, random_seed).to(device)
        self.target_critic = Critic(self.state_size, self.action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr=critic_lr)

        self.noise = OUNoise(self.action_size, random_seed)

        self.memory = ReplayBuffer(self.args.buffer_size, self.args.batch_size, random_seed)

        self.hard_update(self.target_actor, self.local_actor)
        self.hard_update(self.target_critic, self.local_critic)

        self.scores_list = []  # save rewards
        self.last_100_scores_deque = deque(maxlen=100)
        self.last_100_scores_list = []

        self.save_dir = "./models/"
        os.makedirs(self.save_dir, exist_ok=True)
        handlers = [logging.FileHandler(os.path.join(self.save_dir, 'output.log'), mode='w'), logging.StreamHandler()]
        logging.basicConfig(handlers=handlers, level=logging.INFO, format='')
        self.logger = logging.getLogger()
        self.save_dir += 'ddpg_LLC/'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        print('args:', self.args)

    def step(self, state, action, reward, next_state, done, timestep):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > self.args.batch_size and timestep % learning_period == 0:
            for _ in range(update_factor):
                transitions = self.memory.sample()
                self.train(transitions, self.args.gamma)

    def train(self, transitions, gamma):
        states, actions, rewards, next_states, dones = transitions

        # update critic network
        actions_next = self.target_actor(next_states)
        q_targets_next = self.target_critic(next_states, actions_next)

        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        q_expected = self.local_critic(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_critic.parameters(), 1)
        self.critic_optimizer.step()

        # update actor network
        actions_pred = self.local_actor(states)
        actor_loss = -self.local_critic(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.local_critic, self.target_critic, self.args.tau)
        self.soft_update(self.local_actor, self.target_actor, self.args.tau)

        self.epsilon -= epsilon_decay
        self.noise.reset()

    def soft_update(self, source, target, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0-tau) * target_param.data)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def reset_noise(self):
        self.noise.reset()

    def make_action(self, observation, test=False, add_noise=True):
        state = torch.from_numpy(observation).float().to(device)

        self.local_actor.eval()
        with torch.no_grad():
            action = self.local_actor(state).cpu().data.numpy()
        self.local_actor.train()

        if add_noise:
            action += self.epsilon * self.noise.sample()
        return action

    def run(self):
        print("Training DDPG...")
        time_start = time.time()
        total_time_steps = 0
        episode = 0
        while True:
            episode += 1
            state = self.env.reset()
            self.reset_noise()
            one_episode_rewards = 0
            average_score = 0
            timestep = 0
            done = False

            while True:
                action = self.make_action(state)
                next_state, reward, done, info = self.env.step(action)
                one_episode_rewards += reward
                self.step(state, action, reward, next_state, done, timestep)
                state = next_state

                timestep += 1
                total_time_steps += 1
                if done:
                    break
            self.last_100_scores_deque.append(one_episode_rewards)
            self.scores_list.append(one_episode_rewards)

            average_score = np.mean(self.last_100_scores_deque)
            self.last_100_scores_list.append(average_score)

            if episode % 10 == 0 or (len(self.last_100_scores_deque) == 100 and average_score >= self.mean_100_reward_bound):
                duration = int(time.time() - time_start)
                self.logger.info("Episode: {} | Scores: {:.2f} | Mean_Scores: {:.2f} | Time: {:02}:{:02}:{:02}"
                                 .format(episode,
                                         one_episode_rewards,
                                         average_score,
                                         duration//3600,
                                         (duration % 3600)//60,
                                         duration % 60))
            if len(self.last_100_scores_deque) == 100 and average_score >= self.mean_100_reward_bound :
                print("Successfully!")
                torch.save(self.local_actor.state_dict(), self.save_dir + 'actor_network.dat')
                torch.save(self.local_critic.state_dict(), self.save_dir + 'critic_network.dat')
                break

        # plt!
        l1, = plt.plot(range(len(self.scores_list)), self.scores_list)
        l2, = plt.plot(range(len(self.last_100_scores_list)), self.last_100_scores_list)
        plt.xlabel("Episode")
        plt.ylabel("Scores")
        plt.legend([l1, l2], ['scores', 'mean_100_scores'], loc='best')
        plt.savefig(self.save_dir + 'ddpg_LLC_result.png')
        plt.show()

    def init_game_setting(self):
        self.env.reset()
        actor_dat = torch.load(self.save_dir + 'actor_network.dat')
        self.local_actor.load_state_dict(actor_dat)
        self.local_actor.eval()

    def test(self):
        self.init_game_setting()
        self.reset_noise()
        state = self.env.reset()

        scores = 0
        while True:
            action = self.make_action(state)
            self.env.render()
            next_state, score, is_done, _ = self.env.step(action)
            scores += score
            if not is_done:
                state = next_state
            else:
                self.env.close()
                break
        print("Test Reward is", scores)

