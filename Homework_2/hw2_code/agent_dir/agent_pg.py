import os
import logging
import random
import copy
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import matplotlib.pyplot as plt


class PGNetwork(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, output_size=2):
        super(PGNetwork, self).__init__()
        ##################
        # YOUR CODE HERE #
        ##################
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        ##################
        # YOUR CODE HERE #
        ##################
        x = F.relu(self.fc1(inputs))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


# 用来计算状态值函数作为baseline
class StateValueNetwork(nn.Module):

    def __init__(self, input_size=4, hidden_size=16):
        super(StateValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = F.relu(x)
        state_value = self.fc2(x)
        return state_value


# class ReplayBuffer:
#     def __init__(self, buffer_size):
#         """
#         Trajectory buffer. It will clear the buffer after updating.
#         """
#         ##################
#         # YOUR CODE HERE #
#         ##################
#         pass
#
#     def __len__(self):
#         ##################
#         # YOUR CODE HERE #
#         ##################
#         pass
#
#     def push(self, *transition):
#         ##################
#         # YOUR CODE HERE #
#         ##################
#         pass
#
#     def sample(self, batch_size):
#         """
#         Sample all the data stored in the buffer
#         """
#         ##################
#         # YOUR CODE HERE #
#         ##################
#         pass
#
#     def clean(self):
#         ##################
#         # YOUR CODE HERE #
#         ##################
#         pass


class AgentPG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentPG, self).__init__(env)
        ##################
        # YOUR CODE HERE #
        ##################
        self.args = args
        self.env = env
        self.env.seed(self.args.seed)
        self.baseline = self.args.baseline
        self.device = self.prepare_gpu()
        self.discount_factor = self.args.gamma
        self.mean_reward_bound = self.args.reward
        self.policy_Net = PGNetwork(self.env.observation_space.shape[0],  # 4
                                    self.args.hidden_size,  # 16
                                    self.env.action_space.n  # 2
                                    ).to(self.device)
        self.state_value_Net = StateValueNetwork(self.env.observation_space.shape[0],  # 4
                                                 self.args.hidden_size,
                                                 ).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy_Net.parameters(), lr=self.args.lr)
        self.state_value_optimizer = optim.Adam(self.state_value_Net.parameters(), lr=self.args.lr)

        self.scores_list = []  # 保存分数列表
        self.mean_100_scores_list = []  # 保存最后100个分数的平均值 的 列表
        self.last_100_scores_queue = deque(maxlen=100)  # 保存最后100个分数
        self.save_dir = "./models/"

        os.makedirs(self.save_dir, exist_ok=True)
        handlers = [logging.FileHandler(os.path.join(self.save_dir, 'pg_output.log'), mode='w'), logging.StreamHandler()]
        logging.basicConfig(handlers=handlers, level=logging.INFO, format='')
        self.logger = logging.getLogger()

        if self.baseline:
            print("using baseline")
            self.save_dir += "reinforce_baseline/"
        else:
            self.save_dir += "reinforce/"

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        print('All args:', self.args)

    def prepare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        print(device)
        return device

    def make_action(self, observation, test=False):
        """
        Return predicted action of your agent
        Input:observation
        Return: action
        """
        ##################
        # YOUR CODE HERE #
        ##################
        state = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        probs = self.policy_Net(state)
        distribution = Categorical(probs)
        action = distribution.sample()  # 从网络输出按概率采样
        # 返回动作值(0或1)和 其概率的对数值
        return action.item(), distribution.log_prob(action)

    def get_discounted_rewards(self, rewards):
        discount_rewards = []  # 保存一个轨迹(trajectory)的每一步的期望奖励
        total_r = 0
        for r in reversed(rewards):
            total_r = r + total_r * self.discount_factor
            discount_rewards.insert(0, total_r)
        discount_rewards = torch.tensor(discount_rewards).to(self.device)
        if self.baseline:
            discount_rewards = (discount_rewards - discount_rewards.mean()) / discount_rewards.std()
        return discount_rewards

    def train_policy_net(self, deltas, log_probs):
        policy_loss = []
        for d, lp in zip(deltas, log_probs):  # deltas if baseline else discounted_rewards , 由run函数控制
            policy_loss.append(-d * lp)

        self.policy_optimizer.zero_grad()
        sum(policy_loss).backward()
        self.policy_optimizer.step()

    def train_state_value_net(self, discount_rewards, state_vals):
        val_loss = F.mse_loss(state_vals, discount_rewards)

        self.state_value_optimizer.zero_grad()
        val_loss.backward()
        self.state_value_optimizer.step()

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        # print(self.args)
        episode = 0
        while True:
            episode += 1
            state = self.env.reset()
            trajectory = []  # 记录每个episode的轨迹
            score = 0        # 一个轨迹的分数

            while True:
                # select an action
                action, log_prob = self.make_action(state)

                next_state, reward, done, info = self.env.step(action)

                score += reward

                trajectory.append([state, action, reward, log_prob])

                if done:
                    break
                state = next_state

            self.scores_list.append(score)
            self.last_100_scores_queue.append(score)
            average_100_score = np.mean(self.last_100_scores_queue)
            self.mean_100_scores_list.append(average_100_score)

            self.logger.info("Episode: {} | Scores: {} | Mean_score: {}".format(episode, score, average_100_score))

            if average_100_score >= self.mean_reward_bound:
                if self.baseline:
                    torch.save(self.policy_Net.state_dict(), self.save_dir + 'policy_network.dat')
                    torch.save(self.state_value_Net.state_dict(), self.save_dir + 'state_value_network.dat')
                else:
                    torch.save(self.policy_Net.state_dict(), self.save_dir + 'policy_network.dat')
                print("Successfully!")
                break

            # 一个轨迹的相关数据列表:
            states = [step[0] for step in trajectory]
            actions = [step[1] for step in trajectory]
            rewards = [step[2] for step in trajectory]
            log_probs = [step[3] for step in trajectory]

            discounted_rewards = self.get_discounted_rewards(rewards)

            if self.baseline:
                state_vals = []
                for state in states:
                    state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                    state_vals.append(self.state_value_Net(state))
                state_vals = torch.stack(state_vals).squeeze()

                self.train_state_value_net(discounted_rewards, state_vals)

                deltas = [dt - val for dt, val in zip(discounted_rewards, state_vals)]
                deltas = torch.tensor(deltas).to(self.device)

                self.train_policy_net(deltas, log_probs)

            else:
                self.train_policy_net(discounted_rewards, log_probs)

        l1, = plt.plot(range(len(self.scores_list)), self.scores_list)
        l2, = plt.plot(range(len(self.mean_100_scores_list)), self.mean_100_scores_list)
        plt.xlabel("Episode")
        plt.ylabel("Scores")
        plt.legend([l1, l2], ['scores', 'mean_100_scores'], loc='best')
        plt.savefig(self.save_dir + 'result.png')

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.env.reset()
        if self.baseline:
            print("going to load two model")
            policy_data = torch.load(self.save_dir + 'policy_network.dat')
            self.policy_Net.load_state_dict(policy_data)
            self.policy_Net.eval()
            self.policy_Net.to(self.device)

            state_value_data = torch.load(self.save_dir + 'state_value_network.dat')
            self.state_value_Net.load_state_dict(state_value_data)
            self.state_value_Net.eval()
            self.state_value_Net.to(self.device)
        else:
            print("going to load one policy model")
            policy_data = torch.load(self.save_dir + 'policy_network.dat')
            self.policy_Net.load_state_dict(policy_data)
            self.policy_Net.eval()
            self.policy_Net.to(self.device)

    def test(self):
        self.init_game_setting()
        state = self.env.reset()
        scores = 0
        while True:
            action, _ = self.make_action(state)
            next_state, score, is_done, _ = self.env.step(action)
            scores += score
            self.env.render()
            if not is_done:
                state = next_state
            else:
                self.env.close()
                break
        print("Test Reward is: ", scores)


