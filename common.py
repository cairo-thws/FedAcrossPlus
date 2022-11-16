from lightningflower.utility import boolean_string


class Defaults(object):
    # client learning attributes
    SERVER_LR = 0.01
    SERVER_LR_GAMMA = 0.0002
    SERVER_LR_MOMENTUM = 0.9
    SERVER_LR_WD = 0.001
    SERVER_LOSS_EPSILON = 0.1

    # server federated learning attributes
    SERVER_ROUNDS = 1

    # client learning attributes
    CLIENT_LR = 0.1
    # digit5 dataset defaults
    IMG_SIZE_OFFICE_HOME = 256


def add_project_specific_args(parent_parser):
    parser = parent_parser.add_argument_group("FedProtoShot")
    parser.add_argument("--pretrain", type=boolean_string, default=False)
    parser.add_argument("--ckpt_path", type=str, default="")
    return parent_parser


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s