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


class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dueling=False):
        super(QNetwork, self).__init__()
        ##################
        # YOUR CODE HERE #
        ##################
        self.dueling = dueling
        self.conv_layer_1 = nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4)
        self.conv_layer_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv_layer_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc_layer = nn.Linear(7 * 7 * 64, hidden_size)

        if self.dueling:
            print("Using Dueling DQN")
            # V(s) value of the state
            self.dueling_value = nn.Linear(hidden_size, 1)
            # Q(s,a) Q values of the state-action combination
            self.dueling_action = nn.Linear(hidden_size, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        ##################
        # YOUR CODE HERE #
        ##################
        x = F.relu(self.conv_layer_1(inputs))
        x = F.relu(self.conv_layer_2(x))
        x = F.relu(self.conv_layer_3(x))
        x = F.relu(self.fc_layer(x.view(x.size(0), -1)))

        if self.dueling:
            value = self.dueling_value(x)
            advantage = self.dueling_action(x)
            # q = value.expand_as(advantage) + (advantage - advantage.mean(1, keepdim=True).expand_as(advantage))
            q = value + (advantage - advantage.mean(1, keepdim=True))
            return q
        else:
            return self.fc(x)

    def __call__(self, inputs):
        return self.forward(inputs)


TransitionTable = collections.namedtuple('TransitionTable',
                                         field_names=['state', 'action', 'reward', 'done', 'new_state'])


# Class to handle storing and randomly sampling state transitions
class ReplayBuffer:
    def __init__(self, buffer_size):
        ##################
        # YOUR CODE HERE #
        ##################
        self.buffer = collections.deque(maxlen=buffer_size)

    def __len__(self):
        ##################
        # YOUR CODE HERE #
        ##################
        # print(len(self.buffer))
        return len(self.buffer)

    def push(self, transition):
        ##################
        # YOUR CODE HERE #
        ##################
        self.buffer.append(transition)  # 这个transition由上面的TransitionTable得到

    def sample(self, batch_size):
        ##################
        # YOUR CODE HERE #
        ##################
        indices = np.random.choice(
            len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)

    def clean(self):
        ##################
        # YOUR CODE HERE #
        ##################
        self.buffer.clear()


class AgentDQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentDQN, self).__init__(env)
        ##################
        # YOUR CODE HERE #
        ##################
        self.env = env
        self.state = self.env.reset()
        self._reset()
        self.args = args
        self.device = self._prepare_gpu()
        self.exp_buffer = ReplayBuffer(self.args.buffer_size)  # pong: 10000  建议说是1000000
        # print(self.args.buffer_size)
        # print(self.args.target_update_freq)
        self.discount_factor = self.args.gamma  # 0.99
        self.mean_reward_bound = self.args.reward
        self.initial_eps = 1.0
        self.final_eps = 0.02
        self.exploration_eps = 1000000  # 1000000
        self.batch_size = self.args.batch_size  # 32
        self.replay_start = 30000         # pong: 10000  建议说是50000
        self.target_update_freq = self.args.target_update_freq  # pong: 1000  建议说是10000
        # self.episode = 10000000  # 10000000 for breakout
        # self.action_dict = {0: 0, 1: 2, 2: 3}
        self.current_QNet = QNetwork(input_size=env.observation_space.shape,
                                     hidden_size=args.hidden_size,
                                     output_size=env.action_space.n,
                                     dueling=self.args.dueling).to(self.device)
        print(self.current_QNet)
        self.target_QNet = QNetwork(input_size=env.observation_space.shape,
                                    hidden_size=args.hidden_size,
                                    output_size=env.action_space.n,
                                    dueling=self.args.dueling).to(self.device)

        self.target_QNet.load_state_dict(self.current_QNet.state_dict())  # 先复制一份当前要训练的网络
        self.optimizer = torch.optim.Adam(self.current_QNet.parameters(), lr=self.args.lr)
        self.reward_list = []
        self.mean_100_reward_list = []
        self.log_list = []
        self.save_dir = "./models/"

        os.makedirs(self.save_dir, exist_ok=True)
        handlers = [logging.FileHandler(os.path.join(self.save_dir, 'output.log'), mode = 'w'), logging.StreamHandler()]
        logging.basicConfig(handlers=handlers, level=logging.INFO, format='')
        self.logger = logging.getLogger()

        self.save_dir += 'dqn_breakout'  # dqn_pong或dqn_breakout
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        print('args:', self.args)

    def _prepare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:2' if n_gpu > 0 else 'cpu')
        print(device)
        return device

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0  # one episode reward

    def step_env(self, epsilon=0.0):
        done_reward = None

        # choose random action every once in a while
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()  # action = random.choice([0, 2, 3])  # ##########------------
        else:
            action = self.make_action(self.state, False)
        new_state, reward, is_done, info = self.env.step(action)
        self.total_reward += reward
        # self.env.render()
        # save transition to EXP buffer
        exp_tuple = TransitionTable(self.state, action, reward, is_done, new_state)
        self.exp_buffer.push(exp_tuple)
        self.state = new_state

        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

    def train(self, batch):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        states, actions, rewards, dones, next_states = batch

        # convert everything into a torch tensor so that we can compute derivatives
        states_vector = torch.tensor(states).to(self.device)
        next_states_vector = torch.tensor(next_states).to(self.device)
        actions_vector = torch.tensor(actions).to(self.device)
        rewards_vector = torch.tensor(rewards).to(self.device)
        done_mask = torch.BoolTensor(dones).to(self.device)

        # compute the predictions our model would make for this batch of state
        # transitions (represented as tensors)
        state_action_values = self.current_QNet(states_vector).gather(1, actions_vector.unsqueeze(-1).type(torch.int64)).squeeze(-1)

        # # compute the actual values we got for those transitions
        # next_state_values = self.target_QNet(next_states_vector).max(1)[0]
        # # make sure future values aren't being considered for end states
        # next_state_values[done_mask] = 0.0
        # next_state_values = next_state_values.detach()

        # Double DQN
        # 由当前网络得到下一个状态输入进去后得到的下一个动作向量，用来目标网络得到label
        _, next_action_vector = self.current_QNet(next_states_vector).max(1)
        # 同时由目标网络也得到一个作为label
        next_state_values = self.target_QNet(next_states_vector)\
            .gather(1, next_action_vector.unsqueeze(-1).type(torch.int64)).squeeze(-1)
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()
        # Double DQN Finish

        # add the future reward of a state reached to the rewards of the current
        # transition to correctly represent the actual value of the state reached
        expected_state_action_values = next_state_values * self.discount_factor + rewards_vector

        # compute the Mean Squared Error Loss between our predictions and the
        # actual values of the transitions
        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return:action
        """
        ##################
        # YOUR CODE HERE #
        ##################
        state_actions = np.array([observation], copy=False)
        state_vector = torch.tensor(state_actions).to(self.device)
        q_values_vector = self.current_QNet(state_vector)
        _, action_value = torch.max(q_values_vector, dim=1)
        action = int(action_value.item())
        return action

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        curr_frame_idx = 0
        prev_frame_idx = 0
        curr_time = time.time()
        best_mean_reward = None

        # loss = [] #
        current_eps = self.initial_eps
        while True:
            curr_frame_idx += 1

            # compute current epsilon
            if curr_frame_idx > self.replay_start:  # 大于reply_start时开始降epsilon
                current_eps = max(self.final_eps, self.initial_eps - (curr_frame_idx - self.replay_start)/self.exploration_eps)

            reward = self.step_env(current_eps)

            if reward is not None:
                self.reward_list.append(reward)
                speed = (curr_frame_idx - prev_frame_idx) / (time.time() - curr_time)
                prev_frame_idx = curr_frame_idx
                curr_time = time.time()
                mean_reward = np.mean(self.reward_list[-100:])  # 最后100个reward的平均值
                self.mean_100_reward_list.append(mean_reward)
                self.logger.info("episode: {} | reward: {} | mean_reward: {:.2f} | epsilon: {:.2f} | speed: {:.2f} frames/s"
                                 .format(len(self.reward_list), reward, mean_reward, current_eps, speed))

                if best_mean_reward is None or best_mean_reward < mean_reward:
                    torch.save(self.current_QNet.state_dict(), 'double_dqn_'+self.args.env_name+'_best.dat')
                    if best_mean_reward is not None:
                        print("Best mean reward updated from {:.3f} -> {:.3f} and model saved"
                              .format(best_mean_reward, mean_reward))
                    best_mean_reward = mean_reward
                if mean_reward > self.mean_reward_bound:
                    print("Successful!")
                    break
            if self.exp_buffer.__len__() < self.args.buffer_size:
                continue
            if curr_frame_idx % self.target_update_freq == 0:
                self.target_QNet.load_state_dict(self.current_QNet.state_dict())

            self.optimizer.zero_grad()
            batch = self.exp_buffer.sample(self.batch_size)
            loss_t = self.train(batch)
            loss_t.backward()
            self.optimizer.step()

        x1 = range(len(self.reward_list))
        x2 = range(len(self.mean_100_reward_list))
        l1, = plt.plot(x1, self.reward_list, color='b')
        l2, = plt.plot(x2, self.mean_100_reward_list, color='r')
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.legend([l1, l2], ['rewards', 'last 100 mean rewards'], loc='best')
        plt.savefig(self.save_dir + '/double_dqn_reward_' + self.args.env_name + '.png')
        plt.show()

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.state = self.env.reset()
        check_point = torch.load(self.args.check_path)
        self.current_QNet.load_state_dict(check_point)
        self.current_QNet.eval()
        self.current_QNet.to(self.device)
        print("load dict successfully")
        pass

    def test(self):
        self.init_game_setting()
        rewards = 0
        while True:
            state_actions = np.array([self.state], copy=False)
            state_vector = torch.tensor(state_actions).to(self.device)
            q_values_vector = self.current_QNet(state_vector)
            _, action_value = torch.max(q_values_vector, dim=1)
            action = int(action_value.item())
            new_state, reward, is_done, info = self.env.step(action)
            rewards += reward
            self.env.render()
            if not is_done:
                self.state = new_state
            else:
                self.env.close()
                break
        print("Test Reward is: ", rewards)
