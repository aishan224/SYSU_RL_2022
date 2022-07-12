import torch
import os
import numpy as np
from tensorboardX import SummaryWriter
from gym.spaces import Box, Tuple
from pathlib import Path
from utils.make_env import make_env
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from utils.buffer import ReplayBuffer
from utils.rollout import RolloutWorker
from utils.arguments import get_common_args, get_vdn_args, get_maddpg_args
from agent import Agents

import matplotlib.pyplot as plt


def make_parallel_env(env_id, n_parallel_envs, seed, discrete_action=True):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_parallel_envs == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_parallel_envs)])


def get_shape(sp):
    space, dim = 0, 0
    if isinstance(sp, Box):
        space = sp.shape[0]
        dim = sp.shape[0]
    elif isinstance(sp, Tuple):
        for p in sp.spaces:
            if isinstance(p, Box):
                space += p.shape[0]
                dim += p.shape[0]
            else:
                space += p.n
                dim += 1
    else:  # if the instance is 'Discrete', the action dim is 1
        space = sp.n
        dim = 1
    return space, dim


def get_env_scheme(env):
    # simple spread
    agent_init_params = {}

    num_agents = len(env.observation_space)
    observation_space, observation_dim = get_shape(env.observation_space[0])
    action_space, action_dim = get_shape(env.action_space[0])

    agent_init_params['n_agents'] = num_agents
    agent_init_params['observation_space'] = observation_space
    agent_init_params['observation_dim'] = observation_dim
    agent_init_params['action_space'] = action_space
    agent_init_params['action_dim'] = action_dim

    return agent_init_params


def runner(env, args):
    result_dir = Path('./results') / args.algo
    if not result_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in result_dir.iterdir()
                         if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)

    run_dir = result_dir / curr_run
    log_dir = run_dir / 'logs'
    pics_dir = run_dir / 'pics'

    os.makedirs(str(log_dir))
    os.makedirs(str(pics_dir))
    logger = SummaryWriter(str(log_dir))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not args.use_cuda and args.n_training_threads is not None:
        torch.set_num_threads(args.n_training_threads)

    agents = Agents(args)
    rolloutWorker = RolloutWorker(env, agents, args)
    buffer = ReplayBuffer(args)

    train_step = 0
    mean_episode_rewards = []
    last_100_mean_list = []

    for ep_i in range(0, args.n_episodes, args.n_parallel_envs):
        # print("Episodes %i of %i" % (ep_i + 1, args.n_episodes))

        # Using the RolloutWork to interact with the environment (rollout the episodes >= 1)
        episodes, rews, mean_rews = [], [], []
        for episode_idx in range(args.n_rollouts):
            episode, ep_rew, mean_ep_rew = rolloutWorker.generate_episode(episode_idx)
            episodes.append(episode)
            rews.append(ep_rew)
            mean_rews.append(mean_ep_rew)
        episodes_batch = episodes[0]
        episodes.pop(0)
        for episode in episodes:
            for key in episodes_batch.keys():
                episodes_batch[key] = np.concatenate((episodes_batch[key], episode[key]), axis=0)
        buffer.push(episodes_batch)

        # VDN needs the buffer but not the epsilon to train agents
        if args.algo == 'vdn':
            for _ in range(args.training_steps):
                mini_batch = buffer.sample(min(buffer.current_size, args.batch_size))
                agents.train(mini_batch, train_step)
                train_step += 1
        # maddpg needs the buffer and the epsilon to train agents
        elif args.algo == 'maddpg':
            for _ in range(args.training_steps):
                mini_batch = buffer.sample(min(buffer.current_size, args.batch_size))
                agents.train(mini_batch, train_step, rolloutWorker.epsilon)
                train_step += 1

        rews = np.mean(rews)
        mean_rews = np.mean(mean_rews)
        mean_episode_rewards.append(mean_rews)
        last_100_mean = np.mean(mean_episode_rewards[-100:])
        last_100_mean_list.append(last_100_mean)

        logger.add_scalar('mean_episode_rewards', mean_rews, ep_i)
        print("Episode {} of {} : Total reward {:.2f} , Mean reward {:.2f} , last_100_mean {:.2f}".format(ep_i + 1,args.n_episodes, rews, mean_rews, last_100_mean))

        if ep_i % args.save_cycle < args.n_parallel_envs:
            os.makedirs(str(run_dir / 'incremental'), exist_ok=True)
            agents.save(str(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1))))
            agents.save(str(run_dir / 'model.pt'))

    agents.save(str(run_dir / 'model.pt'))
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()

    index = list(range(1, len(mean_episode_rewards) + 1))
    plt.plot(index, mean_episode_rewards)
    plt.plot(index, last_100_mean_list)
    plt.plot(index, [-5.5 for i in range(len(mean_episode_rewards))])
    plt.ylabel("Mean Episode Rewards")
    plt.savefig(str(pics_dir) + '/mean_episode_rewards.png')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    args = get_common_args()
    if args.algo == 'vdn':
        args = get_vdn_args(args)
    elif args.algo == 'maddpg':
        args = get_maddpg_args(args)

    env_id = "simple_spread"
    env = make_parallel_env(env_id, args.n_parallel_envs, args.seed)
    scheme = get_env_scheme(env)
    args.n_agents = scheme['n_agents']
    args.obs_shape = scheme['observation_space']
    args.n_actions = scheme['action_space']

    runner(env, args)
