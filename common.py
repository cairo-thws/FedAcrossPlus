from lightningflower.utility import boolean_string
import torch
import sys
import random

from torch.utils.data import Subset, DataLoader


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


def signal_handler_free_cuda(sig, frame):
    # clear cuda cache
    torch.cuda.empty_cache()
    print('You pressed Ctrl+C! Free CUDA and exit')
    sys.exit(0)


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


def test_prototypes(model, prototypes, loaders, device):
    with torch.no_grad():
        model.eval()
        predictions_total = 0
        true_predictions = 0
        learned_category_idx = list(loaders.keys())
        for category_idx in learned_category_idx:
            query_loader = loaders[category_idx][1]
            for data, labels in query_loader:
                data = data.to(device)
                f, _ = model(data)
                # TODO: make more efficient
                for idx, sample in enumerate(f):
                    dist = torch.squeeze(
                        torch.cdist(prototypes[None].flatten(2), torch.unsqueeze(sample, dim=0)))
                    index_sorted = torch.argsort(dist)
                    predicted_label = learned_category_idx[index_sorted[0]]
                    if predicted_label == labels[idx].item():
                        true_predictions = true_predictions + 1
                    predictions_total = predictions_total + 1
        acc = float(true_predictions) / predictions_total
    return acc


def create_fewshot_loaders(datamodule, episodic_categories, K):
    loaders = dict()
    train_set = datamodule.train_set.dataset

    # create dictionary with index lists
    index_dict = dict()
    for category in episodic_categories:
        index_dict[str(category)] = list()

    # run through dataset
    for batch_idx, (_, labels) in enumerate(train_set):
        for label_idx, label in enumerate(labels):
            if label in episodic_categories:
                batch_size = len(labels)
                idx = batch_idx * batch_size + label_idx
                index_dict[str(label)].append(idx)

    # create subsets form indices
    for key in index_dict:
        idx_subset = index_dict[key]
        # shuffle indices
        random.shuffle(idx_subset)
        # create support set
        k_samples = idx_subset[-K:]
        del idx_subset[-K:]
        support_subset = Subset(train_set, k_samples)
        support_dataloader = DataLoader(support_subset, batch_size=len(k_samples), shuffle=True)

        # create query set
        query_samples = idx_subset  # [-query:]
        query_subset = Subset(train_set, query_samples)
        query_dataloader = DataLoader(query_subset, batch_size=len(query_samples))
        # query_loaders.append(dataloader)
        loaders[int(key)] = (support_dataloader, query_dataloader)

    return loaders
