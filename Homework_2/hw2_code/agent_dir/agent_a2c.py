import os
import logging
import random
import copy
import collections
import time

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent
import matplotlib.pyplot as plt


# helper function to convert numpy arrays to tensors
def t(x):
    return torch.from_numpy(x).float()


class Actor(nn.Module):
    def __init__(self, input_size=4, output_size=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, output_size),
            nn.Softmax()
        )

    def forward(self, x):
        return self.model(x)


class Critic(nn.Module):
    def __init__(self, input_size=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


# Memory
# Stores results from the networks, instead of calculating the operations again from states, etc.
class Memory:
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def _zip(self):
        return zip(self.log_probs,
                   self.values,
                   self.rewards,
                   self.dones)

    def __iter__(self):
        for data in self._zip():
            return data

    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data

    def __len__(self):
        return len(self.rewards)


class AgentA2C(Agent):

    def __init__(self, env, args):

        super(AgentA2C, self).__init__(env)
        self.env = env
        self.args = args
        self.mean_100_reward_bound = self.args.reward
        self.state_dim = self.env.observation_space.shape[0]
        self.action_num = self.env.action_space.n
        self.actor_Net = Actor(self.state_dim, self.action_num)
        self.critic_Net = Critic(self.state_dim)
        self.actor_optimizer = optim.Adam(self.actor_Net.parameters(), lr=self.args.lr)
        self.critic_optimizer = optim.Adam(self.critic_Net.parameters(), lr=self.args.lr)
        self.memory = Memory()
        self.discounted_factor = self.args.gamma
        self.n_step = self.args.n_step   # n_step

        self.scores_list = []
        self.mean_100_scores_list = []

        self.save_dir = "./models/"

        os.makedirs(self.save_dir, exist_ok=True)
        handlers = [logging.FileHandler(os.path.join(self.save_dir, 'output.log'), mode='w'), logging.StreamHandler()]
        logging.basicConfig(handlers=handlers, level=logging.INFO, format='')
        self.logger = logging.getLogger()

        self.save_dir += 'a2c/'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        print('args:', self.args)

    def make_action(self, observation, test=False):
        probs = self.actor_Net(t(observation))
        distribution = torch.distributions.Categorical(probs=probs)
        action = distribution.sample()
        return action, distribution

    def train_process(self, q_val):
        values = torch.stack(self.memory.values)
        q_vals = np.zeros((len(self.memory), 1))

        for i, (_, _, reward, done) in enumerate(self.memory.reversed()):
            q_val = reward + self.discounted_factor * q_val * (1.0 - done)
            q_vals[len(self.memory)-1-i] = q_val
        advantage = torch.Tensor(q_vals) - values

        critic_loss = advantage.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = (-torch.stack(self.memory.log_probs) * advantage.detach()).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def run(self):
        episode = 0
        while True:
            episode += 1
            done = False
            one_episode_reward = 0
            state = self.env.reset()
            step = 0

            while not done:
                action, distribution = self.make_action(state)

                next_state, reward, done, info = self.env.step(action.item())

                one_episode_reward += reward
                step += 1
                self.memory.add(distribution.log_prob(action), self.critic_Net(t(state)), reward, done)

                state = next_state

                if done or (step % self.n_step == 0):
                    last_q_val = self.critic_Net(t(next_state)).detach().data.numpy()
                    self.train_process(last_q_val)
                    self.memory.clear()

            self.scores_list.append(one_episode_reward)
            mean_score = np.mean(self.scores_list[-100:])
            self.mean_100_scores_list.append(mean_score)
            self.logger.info("Episode: {} | Score: {} | Mean_Score: {}".format(episode, one_episode_reward, mean_score))

            if mean_score >= self.mean_100_reward_bound:
                torch.save(self.actor_Net.state_dict(), self.save_dir + 'actor_network.dat')
                torch.save(self.critic_Net.state_dict(), self.save_dir + 'critic_network.dat')
                print("Successfully!")
                break

        l1, = plt.plot(range(len(self.scores_list)), self.scores_list)
        l2, = plt.plot(range(len(self.mean_100_scores_list)), self.mean_100_scores_list)
        plt.xlabel("Episodes")
        plt.ylabel("Scores")
        plt.legend([l1, l2], ['scores', 'mean_100_scores'], loc='best')
        plt.savefig(self.save_dir + 'a2c_result.png')

    def init_game_setting(self):
        self.env.reset()
        actor_dat = torch.load(self.save_dir + 'actor_network.dat')
        self.actor_Net.load_state_dict(actor_dat)
        self.actor_Net.eval()

        critic_dat = torch.load(self.save_dir + 'critic_network.dat')
        self.critic_Net.load_state_dict(critic_dat)
        self.critic_Net.eval()

    def test(self):
        self.init_game_setting()
        state = self.env.reset()
        scores = 0
        while True:
            action, _ = self.make_action(state)
            next_state, score, is_done, _ = self.env.step(action.item())
            scores += score
            self.env.render()
            if not is_done:
                state = next_state
            else:
                self.env.close()
                break
        print("Test Reward is: ", scores)
