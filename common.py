import metric
import random
import sys
import torch

from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from lightningflower.utility import boolean_string
from lightningdata.common import pre_process
from models import ServerDataModel


class ClientAdaptationType(object):
    NONE = 0,
    MEAN_EMBEDDING = 1,
    CENTERED_MEAN_EMBEDDING = 2,
    END_2_END = 3


class NetworkType(object):
    PROTOTYPICAL = 0,
    MATCHING = 1


class DistanceMetric(object):
    EUCLIDEAN = 0,
    KL_DIV = 1


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
    parser.add_argument("--net", type=str, default="resnet50")
    parser.add_argument("--pretrain", type=boolean_string, default=False)
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--fast_server_startup", default=False, type=boolean_string)
    parser.add_argument("--dataset", type=str, default="office31")
    return parent_parser


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def parse_metric(param_str):
    if param_str == "euclidean":
        return DistanceMetric.EUCLIDEAN
    elif param_str == "kl_divergence":
        return DistanceMetric.KL_DIV


def parse_network_type(type_str):
    if type_str == "prototypical":
        return NetworkType.PROTOTYPICAL
    elif type_str == "matching":
        return NetworkType.MATCHING


def parse_adaptation_type(type_str):
    if type_str == "mean_embedding":
        return ClientAdaptationType.MEAN_EMBEDDING
    elif type_str == "centered_mean_embedding":
        return ClientAdaptationType.CENTERED_MEAN_EMBEDDING
    elif type_str == "end_2_end":
        return ClientAdaptationType.END_2_END
    else:
        return ClientAdaptationType.NONE

def test_prototypes_on_client(model, prototypes, episodic_categories, loaders, device, network_type=NetworkType.PROTOTYPICAL, dataset_mean=None):
    with torch.no_grad():
        model.eval()
        if dataset_mean is not None:
            # center prototypes
            prototypes = prototypes - dataset_mean
            prototypes = prototypes / torch.norm(prototypes, p=2)
        predictions_total = 0
        true_predictions = 0
        for data, labels in loaders:
            data = data.to(device)
            f, _ = model(data)
            #f = torch.mean(f, dim=0)
            if network_type == NetworkType.PROTOTYPICAL:
                if dataset_mean is not None:
                    # center feature vector
                    f = f - dataset_mean
                    # gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
                    f = f / torch.norm(f, p=2)
                dist = metric.euclidean_distance(prototypes, f)
                max_indices = torch.argmax(-dist, dim=0)
            # @todo this can be optimized
           # for label_id, max_idx in enumerate(max_indices):
            predicted_label = episodic_categories[max_indices.item()]
            if predicted_label == labels[0].item():
                true_predictions = true_predictions + 1
            predictions_total = predictions_total + 1
    acc = float(true_predictions) / predictions_total
    return acc

def test_prototypes(model, prototypes, loaders, device, network_type=NetworkType.PROTOTYPICAL, dataset_mean=None):
    with torch.no_grad():
        model.eval()
        if dataset_mean is not None:
            print("Dataset mean detected, centering prototypes for test")
            # center prototypes
            prototypes = prototypes - dataset_mean
            prototypes = prototypes / torch.norm(prototypes, p=2)
        predictions_total = 0
        true_predictions = 0
        learned_category_idx = list(loaders[1].keys())
        for category_idx in learned_category_idx:
            query_loader = loaders[1][category_idx]
            for data, labels in query_loader:
                data = data.to(device)
                f, _ = model(data)
                f = torch.mean(f, dim=0)
                if network_type == NetworkType.PROTOTYPICAL:
                    if dataset_mean is not None:
                        # center feature vector
                        f = f - dataset_mean
                        # gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
                        f = f / torch.norm(f, p=2)
                    dist = metric.euclidean_distance(prototypes, f)
                    max_indices = torch.argmax(-dist, dim=0)
                # @todo this can be optimized
               # for label_id, max_idx in enumerate(max_indices):
                predicted_label = learned_category_idx[max_indices.item()]
                if predicted_label == labels[0].item():
                    true_predictions = true_predictions + 1
                predictions_total = predictions_total + 1
        acc = float(true_predictions) / predictions_total
    return acc


def create_fewshot_loaders(datamodule, episodic_categories, K):
    support_loaders = dict()
    query_loaders = dict()
    train_set = datamodule.train_set.dataset
    test_set = datamodule.test_set.dataset

    # create dictionary with index lists
    index_dict_train = dict()
    index_dict_test = dict()
    for category in episodic_categories:
        index_dict_train[str(category)] = list()
        index_dict_test[str(category)] = list()

    # run through dataset
    for batch_idx, (_, labels) in enumerate(train_set):
        for label_idx, label in enumerate(labels):
            if label in episodic_categories:
                batch_size = len(labels)
                idx = batch_idx * batch_size + label_idx
                index_dict_train[str(label)].append(idx)

    # run through dataset
    for batch_idx, (_, labels) in enumerate(test_set):
        for label_idx, label in enumerate(labels):
            if label in episodic_categories:
                batch_size = len(labels)
                idx = batch_idx * batch_size + label_idx
                index_dict_test[str(label)].append(idx)

    # create subsets form indices
    for key in index_dict_train:
        idx_subset = index_dict_train[key]
        # shuffle indices
        random.shuffle(idx_subset)
        # create support set
        k_samples = idx_subset[-K:]
        #del idx_subset[-K:]
        support_subset = Subset(train_set, k_samples)
        support_dataloader = DataLoader(support_subset, batch_size=len(k_samples), shuffle=True)
        support_loaders[int(key)] = support_dataloader

    for key in index_dict_test:
        idx_subset = index_dict_test[key]
        # shuffle indices
        random.shuffle(idx_subset)
        # create query set
        query_samples = idx_subset[-K:]
        query_subset = Subset(test_set, query_samples)
        query_dataloader = DataLoader(query_subset, batch_size=len(query_samples))
        query_loaders[int(key)] = query_dataloader

    return support_loaders, query_loaders


def create_reduced_fewshot_loaders(datamodule, episodic_categories, K):
    support_loader_idx = list()
    query_loader_idx = list()
    val_loader_idx = list()
    # source for support set
    train_set = datamodule.train_set.dataset
    # source for query set
    test_set = datamodule.test_set.dataset
    # source for validation set
    val_set = datamodule.val_set.dataset

    # create dictionary with index lists
    index_dict_train = dict()
    index_dict_test = dict()
    index_dict_val = dict()
    for category in episodic_categories:
        index_dict_train[str(category)] = list()
        index_dict_test[str(category)] = list()
        index_dict_val[str(category)] = list()

    # run through train dataset
    for batch_idx, (_, labels) in enumerate(train_set):
        for label_idx, label in enumerate(labels):
            if label in episodic_categories:
                batch_size = len(labels)
                idx = batch_idx * batch_size + label_idx
                index_dict_train[str(label)].append(idx)

    # run through test dataset
    for batch_idx, (_, labels) in enumerate(test_set):
        for label_idx, label in enumerate(labels):
            if label in episodic_categories:
                batch_size = len(labels)
                idx = batch_idx * batch_size + label_idx
                index_dict_test[str(label)].append(idx)

    # run through val dataset
    for batch_idx, (_, labels) in enumerate(val_set):
        for label_idx, label in enumerate(labels):
            if label in episodic_categories:
                batch_size = len(labels)
                idx = batch_idx * batch_size + label_idx
                index_dict_val[str(label)].append(idx)

    # create train subset from indices
    for key in index_dict_train:
        idx_subset = index_dict_train[key]
        # shuffle indices
        random.shuffle(idx_subset)
        # create support set
        support_loader_idx.extend(idx_subset[-K:])
        # create train subset from indices

    for key in index_dict_test:
        idx_subset = index_dict_test[key]
        # shuffle indices
        random.shuffle(idx_subset)
        # create support set
        # few-shot query
        #query_loader_idx.extend(idx_subset[-K:])
        query_loader_idx.extend(idx_subset)
        # remove indices
        # del idx_subset[-K:]

    for key in index_dict_val:
        idx_subset = index_dict_val[key]
        # shuffle indices
        random.shuffle(idx_subset)
        # create support set
        # few-shot query
        #query_loader_idx.extend(idx_subset[-K:])
        val_loader_idx.extend(idx_subset[-K:])

    random.shuffle(support_loader_idx)
    random.shuffle(query_loader_idx)
    random.shuffle(val_loader_idx)

    query_subset = Subset(test_set, query_loader_idx)
    support_subset = Subset(train_set, support_loader_idx)
    val_subset = Subset(val_set, val_loader_idx)
    query_dataloader = DataLoader(query_subset, batch_size=1, shuffle=True)
    support_dataloader = DataLoader(support_subset, batch_size=K, shuffle=False)
    validation_dataloader = DataLoader(val_subset, batch_size=K, shuffle=True)
    return support_dataloader, query_dataloader, validation_dataloader


def create_empty_server_model(name,
                              num_classes,
                              lr,
                              momentum,
                              gamma,
                              weight_decay,
                              epsilon,
                              net,
                              pretrain):
    model = ServerDataModel(name=name,
                            num_classes=num_classes,
                            lr=lr,
                            momentum=momentum,
                            gamma=gamma,
                            weight_decay=weight_decay,
                            epsilon=epsilon,
                            net=net,
                            pretrain=pretrain)
    return model


def base_augmentation(resize_size=256, crop_size=224, resnet_normalization=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if resnet_normalization:
        ret_transform = transforms.Compose([pre_process.ResizeImage(resize_size),
                                            transforms.RandomResizedCrop(crop_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)])
    else:
        ret_transform = transforms.Compose([pre_process.ResizeImage(resize_size),
                                            transforms.RandomResizedCrop(crop_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(),
                                            transforms.ToTensor()])
    return ret_transform


def boost_augmentation(resize_size=256, crop_size=224, resnet_normalization=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if resnet_normalization:
        ret_transform = transforms.Compose([pre_process.ResizeImage(resize_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)])
    else:
        ret_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(),
                                            transforms.ToTensor()])
    return ret_transform


def no_augmentation(resize_size=224, resnet_normalization=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if resnet_normalization:
        ret_transform = transforms.Compose([pre_process.ResizeImage(resize_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)])
    else:
        ret_transform = transforms.Compose([pre_process.ResizeImage(resize_size), transforms.ToTensor()])
    return ret_transform