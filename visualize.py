import torch
import os

from lightningdata import Office31DataModule
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader

from server import create_class_prototypes

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def min_max_norm_tsne(x):
    tsne = TSNE(n_components=2, random_state=42)

    data = tsne.fit_transform(x)
    data_max, data_min = np.max(data, 0), np.min(data, 0)
    d = (data - data_min) / (data_max - data_min)
    return d


def plot_prototypes(prototypes, labels, class_features=None, dataset_mean=None):
    markers = list()
    for i in range(prototypes.shape[0]):
        markers.append("X")
    #sns.set()
    sns.set_context("paper", font_scale=1.1)
    sns.set_style("ticks")
    # number of observed classes
    nr_classes = prototypes.shape[0]
    if class_features:
        for id, class_feature in enumerate(class_features):
            feature = class_feature.cpu().numpy()
            prototypes = np.concatenate((prototypes, feature),axis=0)
            for x in range(0, len(class_feature)):
                labels.append(labels[id])
                markers.append("o")
    # apply tsne
    transformed_data = min_max_norm_tsne(prototypes)
    # transform to dataframe
    df = pd.DataFrame({'tsne_1': transformed_data[:, 0], 'tsne_2': transformed_data[:, 1], 'label': labels})
    # plot
    fig, ax = plt.subplots(1)
    plt.figure(figsize=(10, 10), dpi=300)
    sns.scatterplot(
        x="tsne_1", y="tsne_2",
        hue="label",
        palette=sns.color_palette("hls", nr_classes),
        data=df,
        legend=False,
        alpha=0.3)#,
        #markers=True,
        #style=markers)
    lim = (transformed_data.min() - 0.1, transformed_data.max() + 0.1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')

    #plt.tight_layout()
    plt.title("t-SNE Results")
    plt.xlabel("Dimension 1", xlabel="CHANGE ME")
    plt.show()
    #plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
    #plt.show()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENERATE = False
NORMALIZE = True


def main() -> None:

    if GENERATE:
        # load pretrained model
        model = torch.load(os.path.join("data", "pretrained", "office31.pt"))
        model = model.to(DEVICE)
        # PREPARE SOURCE DATASET
        dataset = Office31DataModule
        # the first domain is server source domain
        source_idx = 0
        domain = dataset.get_domain_names()[source_idx]
        source_dm = dataset(data_dir="data",
                            domain=domain,
                            batch_size=96,
                            num_workers=6,
                            shuffle=True
                            )

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
            torch.save(dataset_mean, os.path.join("data", "plot", "office31_source_dataset_mean.pt"))
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
        torch.save(class_prototypes, os.path.join("data", "plot", "office31_protos.pt"))
        torch.save(class_features, os.path.join("data", "plot", "office31_source_class_features.pt"))
    else:
        class_prototypes = torch.load(os.path.join("data", "plot", "office31_protos.pt"))
        class_features = torch.load(os.path.join("data", "plot", "office31_source_class_features.pt"))
        dataset_mean = torch.load(os.path.join("data", "plot", "office31_source_dataset_mean.pt"))

    labels = list([5, 6, 7, 8])
    reduced_class_features = list()
    for idx, feature in enumerate(class_features):
        if idx in labels:
            reduced_class_features.append(feature)
    plot_prototypes(class_prototypes[labels].cpu().numpy(), labels, reduced_class_features, dataset_mean)


if __name__ == "__main__":
    main()