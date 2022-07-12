import sys
import os
import numpy as np
from pathlib import Path
import torch
from arguments import get_arguments
from utils.util import get_cent_act_dim, get_dim_from_space
from envs.mpe.MPE_Env import MPEEnv
from envs.env_wrappers import DummyVecEnv, SubprocVecEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = MPEEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = MPEEnv(all_args)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int, default=3, help="number of agents")
    parser.add_argument('--use_same_share_obs', action='store_false',
                        default=True, help="Whether to use available actions")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_arguments()
    all_args = parse_args(args, parser)
    print(all_args)

    # just use cpu
    print("choose to use cpu...")
    device = torch.device("cpu")
    torch.set_num_threads(all_args.n_training_threads)

    # setup file to output tensorboard, hyper parameters, and saved models
    run_dir = Path(os.path.dirname(os.path.abspath(__file__)) + "/results") / all_args.algorithm_name

    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[
                                 1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))


    # set seeds
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # create env
    env = make_train_env(all_args)
    num_agents = all_args.num_agents

    # create policies and mapping fn
    if all_args.share_policy:
        policy_info = {
            'policy_0': {"cent_obs_dim": get_dim_from_space(env.share_observation_space[0]),
                         "cent_act_dim": get_cent_act_dim(env.action_space),
                         "obs_space": env.observation_space[0],
                         "share_obs_space": env.share_observation_space[0],
                         "act_space": env.action_space[0]}
        }

        def policy_mapping_fn(id):
            return 'policy_0'
    else:
        policy_info = {
            'policy_' + str(agent_id): {"cent_obs_dim": get_dim_from_space(env.share_observation_space[agent_id]),
                                        "cent_act_dim": get_cent_act_dim(env.action_space),
                                        "obs_space": env.observation_space[agent_id],
                                        "share_obs_space": env.share_observation_space[agent_id],
                                        "act_space": env.action_space[agent_id]}
            for agent_id in range(num_agents)
        }

        def policy_mapping_fn(agent_id):
            return 'policy_' + str(agent_id)

    from runner.mpe_runner import MPERunner as Runner
    assert all_args.n_rollout_threads == 1, (
        "only support 1 env in recurrent version.")
    eval_env = env

    config = {"args": all_args,
              "policy_info": policy_info,
              "policy_mapping_fn": policy_mapping_fn,
              "env": env,
              "eval_env": eval_env,
              "num_agents": num_agents,
              "device": device,
              "use_same_share_obs": all_args.use_same_share_obs,
              "run_dir": run_dir
              }

    total_num_steps = 0
    runner = Runner(config=config)
    while total_num_steps < all_args.num_env_steps:
        total_num_steps = runner.run()

    env.close()
    if all_args.use_eval and (eval_env is not env):
        eval_env.close()

    # if all_args.use_wandb:
    #     run.finish()
    # else:
    runner.writter.export_scalars_to_json(
        str(runner.log_dir + '/summary.json'))
    runner.writter.close()


if __name__ == "__main__":
    print(sys.argv[1:])
    main(sys.argv[1:])
