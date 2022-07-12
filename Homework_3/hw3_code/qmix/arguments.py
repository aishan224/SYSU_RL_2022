import argparse


def get_arguments():
    parser = argparse.ArgumentParser(
        description="MPE-QMIX", formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str, default="qmix", choices=["qmix"])
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for numpy/torch")
    parser.add_argument('--n_training_threads', type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument('--n_rollout_threads', type=int, default=1,
                        help="Number of parallel envs for training rollout")
    parser.add_argument('--n_eval_rollout_threads', type=int, default=1,
                        help="Number of parallel envs for evaluating rollout")
    parser.add_argument('--num_env_steps', type=int,
                        default=2500000, help="Number of env steps to train for")

    # replay buffer parameters
    parser.add_argument('--episode_length', type=int,
                        default=25, help="Max length for any episode")
    parser.add_argument('--buffer_size', type=int, default=5000,
                        help="Max # of transitions that replay buffer can contain")
    parser.add_argument('--use_reward_normalization', action='store_false',
                        default=True, help="Whether to normalize rewards in replay buffer")
    parser.add_argument('--popart_update_interval_step', type=int, default=2,
                        help="After how many train steps popart should be updated")

    # network parameters
    parser.add_argument("--use_centralized_Q", action='store_false',
                        default=True, help="Whether to use centralized Q function")
    parser.add_argument('--share_policy', action='store_false',
                        default=True, help="Whether agents share the same policy")
    parser.add_argument('--hidden_size', type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument('--layer_N', type=int, default=1,
                        help="Number of layers for actor/critic networks")
    parser.add_argument('--use_ReLU', action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument('--use_feature_normalization', action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument('--use_orthogonal', action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")
    parser.add_argument("--use_conv1d", action='store_true',
                        default=False, help="Whether to use conv1d")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")

    # recurrent parameters
    parser.add_argument('--prev_act_inp', action='store_true', default=False,
                        help="Whether the actor input takes in previous actions as part of its input")
    parser.add_argument("--use_rnn_layer", action='store_false',
                        default=True, help='Whether to use a recurrent policy')
    parser.add_argument("--use_naive_recurrent_policy", action='store_false',
                        default=True, help='Whether to use a naive recurrent policy')

    parser.add_argument("--recurrent_N", type=int, default=1)
    parser.add_argument('--data_chunk_length', type=int, default=80,
                        help="Time length of chunks used to train via BPTT")
    parser.add_argument('--burn_in_time', type=int, default=0,
                        help="Length of burn in time for RNN training, see R2D2 paper")

    # optimizer parameters
    parser.add_argument('--lr', type=float, default=7e-4,  # from 5e-4
                        help="Learning rate for Adam")
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # algo common parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Number of buffer transitions to train on at once")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="Discount factor for env")
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True)
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--huber_delta", type=float, default=10.0)

    # soft update parameters
    parser.add_argument('--use_soft_update', action='store_false',
                        default=True, help="Whether to use soft update")
    parser.add_argument('--tau', type=float, default=0.005,
                        help="Polyak update rate")
    # hard update parameters
    parser.add_argument('--hard_update_interval_episode', type=int, default=100,
                        help="After how many episodes the lagging target should be updated")
    parser.add_argument('--hard_update_interval', type=int, default=200,
                        help="After how many timesteps the lagging target should be updated")

    # qmix parameters
    parser.add_argument('--use_double_q', action='store_false',
                        default=True, help="Whether to use double q learning")
    parser.add_argument('--hypernet_layers', type=int, default=2,
                        help="Number of layers for hypernetworks. Must be either 1 or 2")
    parser.add_argument('--mixer_hidden_dim', type=int, default=32,
                        help="Dimension of hidden layer of mixing network")
    parser.add_argument('--hypernet_hidden_dim', type=int, default=64,
                        help="Dimension of hidden layer of hypernetwork (only applicable if hypernet_layers == 2")

    # exploration parameters
    parser.add_argument('--num_random_episodes', type=int, default=5,
                        help="Number of episodes to add to buffer with purely random actions")
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                        help="Starting value for epsilon, for eps-greedy exploration")
    parser.add_argument('--epsilon_finish', type=float, default=0.02,
                        help="Ending value for epsilon, for eps-greedy exploration")
    parser.add_argument('--epsilon_anneal_time', type=int, default=50000,
                        help="Number of episodes until epsilon reaches epsilon_finish")
    parser.add_argument('--act_noise_std', type=float,
                        default=0.1, help="Action noise")

    # train parameters
    parser.add_argument('--actor_train_interval_step', type=int, default=2,
                        help="After how many critic updates actor should be updated")
    parser.add_argument('--train_interval_episode', type=int, default=1,
                        help="Number of env steps between updates to actor/critic")
    parser.add_argument('--train_interval', type=int, default=100,
                        help="Number of episodes between updates to actor/critic")

    # eval parameters
    parser.add_argument('--use_eval', action='store_false',
                        default=True, help="Whether to conduct the evaluation")
    parser.add_argument('--eval_interval', type=int, default=2500,  # 每100episode eval一次
                        help="After how many steps the policy should be evaled")
    parser.add_argument('--num_eval_episodes', type=int, default=32,
                        help="How many episodes to collect for each eval")

    # save parameters
    parser.add_argument('--save_interval', type=int, default=125000,  # 5000 episodes save once
                        help="After how many episodes of training the policy model should be saved")

    # log parameters
    parser.add_argument('--log_interval', type=int, default=25,
                        help="After how many episodes of training the policy model should be saved")

    # pretained parameters
    parser.add_argument("--model_dir", type=str, default=None)

    return parser
