import torch
import os
import random

import pytorch_lightning
from lightningdata import Office31DataModule
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader

from common import Defaults, create_empty_server_model, create_fewshot_loaders
from client import ClientDataModel

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# seed to make plots reproducible
pytorch_lightning.seed_everything(42)


def min_max_norm_tsne(x):
    tsne = TSNE(n_components=2, random_state=42)

    data = tsne.fit_transform(x)
    data_max, data_min = np.max(data, 0), np.min(data, 0)
    d = (data - data_min) / (data_max - data_min)
    return d


def plot_prototypes(prototypes, labels, class_features=None, dataset_mean=None):
    markers = list()
    sizes = list()
    num_protos = prototypes.shape[0]
    for i in range(num_protos):
        markers.append('v')
        sizes.append(1)
    #sns.set()
    sns.set_context("paper", font_scale=1.1)
    sns.set_style("ticks")
    # number of observed classes
    nr_classes = prototypes.shape[0]
    if class_features:
        for id, class_feature in enumerate(class_features):
            feature = class_feature.cpu().numpy()
            prototypes = np.concatenate((prototypes, feature), axis=0)
            for x in range(0, len(class_feature)):
                labels.append(labels[id])
                markers.append('o')
                sizes.append(1)
    # apply tsne
    transformed_data = min_max_norm_tsne(prototypes)
    # transform to dataframe
    df = pd.DataFrame({'tsne_1': transformed_data[:, 0], 'tsne_2': transformed_data[:, 1], 'label': labels, 'markers': markers})

    df_labels = df.iloc[num_protos: , :]
    df_protos = df.iloc[:num_protos, :]
    # plot
    _, ax = plt.subplots(1)
    plt.figure(figsize=(8, 8), dpi=300)
    # scatterplot of ground truth labels projected into the embedding space
    sns.scatterplot(
        x="tsne_1", y="tsne_2",
        hue="label",
        palette=sns.color_palette("hls", nr_classes),
        data=df_labels,
        legend=False,
        alpha=0.5,
        # from http://mirrors.ibiblio.org/CTAN/fonts/stix/doc/stix.pdf
        marker='o',
        s=350)

    # scatterplot of prototypes projected into the embedding space
    sns.scatterplot(
        x="tsne_1", y="tsne_2",
        #hue="label",
        #palette=sns.color_palette("hls", nr_classes),
        c="red",
        data=df_protos,
        legend=False,
        alpha=1.0,
        # from http://mirrors.ibiblio.org/CTAN/fonts/stix/doc/stix.pdf
        marker='v',
        s=400)
    lim = (transformed_data.min() - 0.1, transformed_data.max() + 0.1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')

    plt.title("")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig("data/plot/out/tuned.png")
    #plt.xlabel("Dimension 1")
    plt.show()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENERATE = False
NORMALIZE = True
USE_PRETRAIN = False
NET = "resnet50"
#PATH_TO_PLOTS = os.path.join("data", "plot", "pretrain") if USE_PRETRAIN else os.path.join("data", "plot", "not_pretrained")
PATH_TO_PLOTS = os.path.join("data", "plot", "tuned")
DATASET = "office31"
SOURCE_SUBDOMAIN_ID = "amazon"
TARGET_SUBDOMAIN_ID = "webcam"
FILE_ENDING = ".pt"
SOURCE_FILE_NAME = NET + "_" + DATASET + "_" + SOURCE_SUBDOMAIN_ID + "_model"
TARGET_FILE_NAME = NET + "_" + DATASET + "_" + TARGET_SUBDOMAIN_ID + "_model"
MODEL_PATH = os.path.join("data", "pretrained", SOURCE_FILE_NAME + FILE_ENDING) if USE_PRETRAIN else os.path.join("data", "not_pretrained", SOURCE_FILE_NAME + FILE_ENDING)


def create_prototypes():
    pass


def main() -> None:
    # the dataset under testing
    dataset = Office31DataModule
    # the subdomain to extract features and prototypes from
    domain = TARGET_SUBDOMAIN_ID
    source_dm = dataset(data_dir="data",
                        domain=domain,
                        batch_size=32,
                        num_workers=6,
                        shuffle=True
                        )

    if GENERATE:
        # load pretrained model
        server_model = create_empty_server_model(name="office31",
                                                 num_classes=31,
                                                 lr=Defaults.SERVER_LR,
                                                 momentum=Defaults.SERVER_LR_MOMENTUM,
                                                 gamma=Defaults.SERVER_LR_GAMMA,
                                                 weight_decay=Defaults.SERVER_LR_WD,
                                                 epsilon=Defaults.SERVER_LOSS_EPSILON,
                                                 net=NET,
                                                 pretrain=False if USE_PRETRAIN else True,
                                                 optimizer="sgd")
        if USE_PRETRAIN:
            server_model.load_state_dict(torch.load("data/pretrained/resnet50_office31_amazon_model_webcam_k_10.pt", map_location=DEVICE))
            #server_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        pretrained_model = server_model.model  # .to(DEVICE)
        client_data_model = ClientDataModel(server_model.hparams_initial, pretrained_model=pretrained_model)
        client_data_model = client_data_model.to(DEVICE)
        model = client_data_model.model

        # prepare dataset mean
        source_dm.prepare_data()
        source_dm.setup()
        data_loader = source_dm.train_dataloader()
        with torch.no_grad():
            model.eval()
            dataset_mean_vec = []
            for data, _ in data_loader:
                data = data.to(DEVICE)
                f, _ = model(data)
                dataset_mean_vec.extend(f)

            dataset_mean = torch.mean(torch.stack(dataset_mean_vec, 0), dim=0)
            torch.save(dataset_mean, os.path.join(PATH_TO_PLOTS, TARGET_FILE_NAME + "_dataset_mean.pt"))
            print("[MODEL] Finished creating dataset mean.")

        # prepare class prototypes
        source_dm.prepare_data()
        source_dm.setup()
        data_loader = source_dm.train_dataloader()
        with torch.no_grad():
            model.eval()
            class_protos = []
            class_features = []
            # for i in range(train_set.num_classes):
            for i in range(len(data_loader.dataset.dataset.classes)):
                # idx_subset = train_set.labels_to_idx[i]
                idx_subset = [j for j, (_, y) in enumerate(data_loader.dataset) if y == i]
                subset = Subset(data_loader.dataset, idx_subset)
                dataloader = DataLoader(subset, batch_size=len(idx_subset))
                for data, labels in dataloader:
                    data = data.to(DEVICE)
                    f, _ = model(data)
                    if NORMALIZE:
                        # center feature vector
                        f = f - dataset_mean
                        f = f / torch.norm(f, p=2)
                    class_features.append(f)
                    class_protos.append(torch.mean(f, dim=0))
            class_prototypes = torch.stack(class_protos, 0)
        print("[MODEL] Finished creating class prototypes.")
        torch.save(class_prototypes, os.path.join(PATH_TO_PLOTS, TARGET_FILE_NAME + "_protos.pt"))
        torch.save(class_features, os.path.join(PATH_TO_PLOTS, TARGET_FILE_NAME + "_class_features.pt"))
    else:
        class_prototypes = torch.load(os.path.join(PATH_TO_PLOTS, TARGET_FILE_NAME + "_protos.pt"))
        class_features = torch.load(os.path.join(PATH_TO_PLOTS, TARGET_FILE_NAME + "_class_features.pt"))
        dataset_mean = torch.load(os.path.join(PATH_TO_PLOTS, TARGET_FILE_NAME + "_dataset_mean.pt"))

    #labels = random.sample(range(31), 5)
    labels = list([1, 2, 3, 4, 5])
    reduced_class_features = list()
    for idx, feature in enumerate(class_features):
        if idx in labels:
            reduced_class_features.append(feature)
    plot_prototypes(class_prototypes[labels].cpu().numpy(), labels, reduced_class_features, dataset_mean)


if __name__ == "__main__":
    main()