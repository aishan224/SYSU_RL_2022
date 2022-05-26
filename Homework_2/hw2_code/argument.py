def dqn_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="PongNoFrameskip-v4", help='environment name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--buffer_size", default=int(1e4), type=int)
    parser.add_argument("--lr", default=0.0001, type=float)  # pong: 0.0001  建议说是0.00025
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--grad_norm_clip", default=10, type=float)
    parser.add_argument("--dueling", action='store_true')  # whether using Dueling
    parser.add_argument('--check_path', default=None, type=str, help='the path to load checkpoint')
    parser.add_argument('--reward', default=17, type=float, help='mean reward bound to end')

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_frames", default=int(400000), type=int)
    parser.add_argument("--learning_freq", default=1, type=int)
    parser.add_argument("--target_update_freq", default=1000, type=int)

    return parser


def pg_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="CartPole-v0", help='environment name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=16, type=int)
    parser.add_argument("--lr", default=0.02, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--grad_norm_clip", default=10, type=float)
    parser.add_argument("--baseline", default=False, action='store_true', help="whether to use baseline")
    parser.add_argument("--reward", default=180.0, help="mean reward bound to end")
    parser.add_argument('--n_step', default=20, help="n step A2C")

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_frames", default=int(30000), type=int)

    return parser


def ddpg_arguments(parser):
    parser.add_argument('--env_name', default="LunarLanderContinuous-v2", help='environment name')
    parser.add_argument("--seed", default=8, type=int)
    parser.add_argument('--buffer_size', default=int(1e6), type=int, help="buffer size")
    parser.add_argument('--batch_size', default=256, type=int, help="batch size")
    parser.add_argument('--tau', default=1e-3, help='for soft update of target parameters')
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward", default=180.0, help="mean reward bound to end")

    return parser
