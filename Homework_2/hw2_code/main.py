import argparse
from wrappers import make_env

import gym
from argument import dqn_arguments, pg_arguments, ddpg_arguments


def parse():
    parser = argparse.ArgumentParser(description="SYSU_RL_HW2")
    parser.add_argument('--train_pg', default=False, type=bool, help='whether train policy gradient')
    parser.add_argument('--test_pg', default=False, type=bool, help='whether test policy gradient')
    parser.add_argument('--train_a2c', default=False, type=bool, help='whether train a2c')  # using pg_argument
    parser.add_argument('--test_a2c', default=False, type=bool, help='whether test a2c')

    parser.add_argument('--train_ddpg', default=False, type=bool, help='whether train ddpg llc')
    parser.add_argument('--test_ddpg', default=False, type=bool, help="whether test ddpg")

    parser.add_argument('--train_dqn', default=False, type=bool, help='whether train DQN')
    parser.add_argument('--test_dqn', default=False, type=bool, help='whether test DQN')

    # parser = dqn_arguments(parser)
    # parser = pg_arguments(parser)
    parser = ddpg_arguments(parser)
    args = parser.parse_args()
    return args


def run(args):
    if args.train_pg:
        env_name = args.env_name
        env = gym.make(env_name)
        # print(env.observation_space)
        from agent_dir.agent_pg import AgentPG
        agent = AgentPG(env, args)
        agent.run()

    if args.train_dqn:
        env_name = args.env_name
        env = make_env(env_name)
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        agent.run()

    if args.test_dqn:
        env_name = args.env_name
        env = make_env(env_name)
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        agent.test()

    if args.test_pg:
        env_name = args.env_name
        env = gym.make(env_name)
        from agent_dir.agent_pg import AgentPG
        agent = AgentPG(env, args)
        agent.test()

    if args.train_a2c:
        env_name = args.env_name
        env = gym.make(env_name)
        from agent_dir.agent_a2c import AgentA2C
        agent = AgentA2C(env, args)
        agent.run()

    if args.test_a2c:
        env_name = args.env_name
        env = gym.make(env_name)
        from agent_dir.agent_a2c import AgentA2C
        agent = AgentA2C(env, args)
        agent.test()

    if args.train_ddpg:
        env_name = args.env_name
        env = gym.make(env_name)
        from agent_dir.agent_ddpg import AgentDDPG
        agent = AgentDDPG(env, args, random_seed=8)
        agent.run()

    if args.test_ddpg:
        env_name = args.env_name
        env = gym.make(env_name)
        from agent_dir.agent_ddpg import AgentDDPG
        agent = AgentDDPG(env, args, random_seed=8)
        agent.test()


if __name__ == '__main__':
    args = parse()
    run(args)
