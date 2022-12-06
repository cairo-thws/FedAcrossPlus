import gc
import os
import random

import torch
import signal

# Argument parser
from argparse import ArgumentParser

# TorchVision
from flwr.common import FitRes, FitIns, GetPropertiesRes, GetPropertiesIns, GetParametersIns, GetParametersRes, parameters_to_ndarrays
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# flower framework
import flwr as fl
from flwr.client import start_client
from flwr.common.typing import Code, Parameters, Status, EvaluateIns, EvaluateRes

# pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer

# lightningflower framework imports
from lightningflower.client import LightningFlowerClient
from lightningflower.data import LightningFlowerData
from lightningflower.model import LightningFlowerModel

# lightningdata wrappers
from lightningdata.modules.domain_adaptation.office31_datamodule import Office31DataModule
# from lightningdata.common.pre_process import ResizeImage

# project imports
from common import add_project_specific_args, signal_handler_free_cuda, test_prototypes, create_fewshot_loaders

from models import ClientDataModel, ServerDataModel

import timeit

"""
If you get an error like: “failed to connect to all addresses” “grpc_status”:14 
Then uncomment the lines bellow:
"""
if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]


"""
Capture KB interrupt and free cuda memory
"""
signal.signal(signal.SIGINT, signal_handler_free_cuda)


class ProtoFewShotPlusClient(LightningFlowerClient):
    def __init__(self,
                 model,
                 trainer_args,
                 c_id,
                 datamodule=None):
        super().__init__(model=model,
                         trainer_args=trainer_args,
                         c_id=c_id,
                         datamodule=datamodule)
        # for saving the source class names
        self.source_classes = list()
        # the source prototypes
        self.prototypes = None
        # few-shot parameters
        self.K = None
        self.N = None
        # client-side learning configuration
        self.episodic_categories = list()
        self.loaders = dict()
        self.training_episodes = None

        print("[CLIENT " + str(self.c_id) + "] Init ProtoFewShotPlusClient with id" + str(c_id))

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        print("[CLIENT " + str(self.c_id) + "] Get params")
        ret_status = Status(code=Code.OK, message="")
        ret_params = Parameters(tensor_type="", tensors=[])
        return GetParametersRes(status=ret_status, parameters=ret_params)

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        print("[CLIENT " + str(self.c_id) + "] Get props")
        return None

    def check_fitIns(self, ins):
        # get source classes
        if not self.source_classes and "source_classes" in ins.config:
            self.source_classes = ins.config["source_classes"].split(",")
            if len(self.source_classes) > 0:
                print("[CLIENT " + str(self.c_id) + "] Received " + str(
                    len(self.source_classes)) + " source classes: " + ",".join(self.source_classes))

        if not self.episodic_categories and "K" in ins.config and "N" in ins.config:
            self.N = int(ins.config["N"])
            self.K = int(ins.config["K"])
            print("[CLIENT " + str(self.c_id) + "] Received K=" + str(self.K) + " and N=" + str(self.N))
            # randomly sample categories according to server configuration
            self.episodic_categories = random.sample(range(0, len(self.source_classes)), self.N)

        if "training_episodes" in ins.config:
            self.training_episodes = int(ins.config["training_episodes"])

        if ins.parameters.tensors is not None:
            # saving the most recent prototypes
            np_list = parameters_to_ndarrays(ins.parameters)
            self.prototypes = torch.stack([torch.from_numpy(item) for item in np_list], 0)
            self.prototypes = self.prototypes.to(DEVICE)
            print("[CLIENT " + str(self.c_id) + "] Received new prototypes")

    def check_evalIns(self, ins):
        pass

    def generate_base_dataloaders(self):
        """Generates query and support sets according to the N/K parameters"""

        # check if the dataloaders are already set
        if len(self.loaders) > 0:
            return

        self.loaders = create_fewshot_loaders(self.datamodule, self.episodic_categories, self.K)
        """
        train_set = self.datamodule.train_set.dataset

        # create dictionary with index lists
        index_dict = dict()
        for category in self.episodic_categories:
            index_dict[str(category)] = list()

        # run through dataset
        for batch_idx, (_, labels) in enumerate(train_set):
            for label_idx, label in enumerate(labels):
                if label in self.episodic_categories:
                    batch_size = len(labels)
                    idx = batch_idx * batch_size + label_idx
                    index_dict[str(label)].append(idx)

        # create subsets form indices
        for key in index_dict:
            idx_subset = index_dict[key]
            # shuffle indiceson
            random.shuffle(idx_subset)
            # create support set
            k_samples = idx_subset[-self.K:]
            del idx_subset[-self.K:]
            support_subset = Subset(train_set, k_samples)
            support_dataloader = DataLoader(support_subset, batch_size=len(k_samples), shuffle=True)

             # create query set
            query_samples = idx_subset #[-query:]
            query_subset = Subset(train_set, query_samples)
            query_dataloader = DataLoader(query_subset, batch_size=len(query_samples))
            #query_loaders.append(dataloader)
            self.loaders[key] = (support_dataloader, query_dataloader)
        """

        print("[CLIENT " + str(self.c_id) + "] Query and support sets generated")

    def train_episodic_prototypes(self):
        print("[CLIENT " + str(self.c_id) + "] Train episodic prototypes")
        return None, None

    def evaluate_client_model(self):
        acc = test_prototypes(self.localModel.model, self.prototypes[self.episodic_categories], self.loaders, DEVICE)
        """
        with torch.no_grad():
            self.localModel.model.eval()
            predictions_total = 0
            true_predictions = 0
            learned_category_idx = list(self.loaders.keys())
            for category_idx in learned_category_idx:
                query_loader = self.loaders[category_idx][1]
                for data, labels in query_loader:
                    data = data.to(DEVICE)
                    _, preds = self.localModel.model(data)
                    # TODO: make more efficient
                    for idx, sample in enumerate(preds):
                        dist = torch.squeeze(
                            torch.cdist(prototypes[None].flatten(2), torch.unsqueeze(sample, dim=0)))
                        index_sorted = torch.argsort(dist)
                        predicted_label = learned_category_idx[index_sorted[0]]
                        if predicted_label == labels[idx].item():
                            true_predictions = true_predictions + 1
                        predictions_total = predictions_total + 1
        acc = float(true_predictions) / predictions_total"""
        print("[CLIENT " + str(self.c_id) + "] Accuracy on query categories= " + str(acc))
        return acc

    def fit(self, ins: FitIns) -> FitRes:
        """Refine the provided weights using the locally held dataset.

        Parameters
        ----------
        ins : FitIns
            The training instructions containing (global) model parameters
            received from the server and a dictionary of configuration values
            used to customize the local training process.

        Returns
        -------
        FitRes
            The training result containing updated parameters and other details
            such as the number of local training examples used for training.
        """
        print("[CLIENT " + str(self.c_id) + "] Fit in global round " + ins.config["global_round"])

        # begin time measurement
        fit_begin = timeit.default_timer()

        # check new incoming parameters and configuration from server side
        self.check_fitIns(ins)
        self.generate_base_dataloaders()

        #best_episodic_prototypes, best_model = self.prototypes, self.model#self.train_episodic_prototypes()

        ret_status = Status(code=Code.OK, message="OK")
        ret_params = Parameters(tensor_type="", tensors=[])
        ret_metrics = dict()
        ret_metrics["duration"] = timeit.default_timer() - fit_begin
        ret_metrics["client_id"] = self.c_id


        return FitRes(status=ret_status,
                      parameters=ret_params,
                      num_examples=0,
                      metrics=ret_metrics)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # begin time measurement
        eval_begin = timeit.default_timer()

        self.check_evalIns(ins)
        acc = self.evaluate_client_model()

        ret_metrics = dict()
        ret_metrics["duration"] = timeit.default_timer() - eval_begin
        ret_metrics["client_id"] = self.c_id
        ret_metrics["accuracy"] = str(acc)

        return EvaluateRes(status=Status(code=Code.OK,
                                         message="OK"),
                           loss=0.0,
                           num_examples=0,
                           metrics=ret_metrics)


def get_client_train_augmentation():
    pass


def get_client_test_augmentation():
    pass


def main() -> None:
    parser = ArgumentParser()
    # paper-specific arguments
    parser = add_project_specific_args(parser)
    # Data-specific arguments
    parser = LightningFlowerData.add_data_specific_args(parser)
    # Trainer-specific arguments
    parser = pl.Trainer.add_argparse_args(parser)
    # Client-specific arguments
    parser = LightningFlowerClient.add_client_specific_args(parser)
    args = parser.parse_args()

    # fixed seeding
    seed_everything(42, workers=True)

    # select dataset
    dataset = Office31DataModule

    # limit client id number
    client_id = args.client_id % len(dataset.get_domain_names())
    # client ids start with 1, 0 is reserved for server
    domain = dataset.get_domain_names()[client_id]
    transform_train = get_client_train_augmentation()
    transform_test = get_client_test_augmentation()
    dm_train = dataset(data_dir=args.dataset_path,
                       domain=domain,
                       batch_size=args.batch_size_train,
                       num_workers=args.num_workers,
                       # train_transforms=transform_train,
                       # test_transforms=transform_test,
                       #pre_split=True,
                       drop_last=True,
                       shuffle=False)  # do not shuffle data for self supervised label discovery

    # load pretrained server model
    path_to_file = os.path.join("data", "pretrained", str(Office31DataModule.get_dataset_name()) + ".pt")
    model_file_exists = os.path.exists(path_to_file)
    if model_file_exists:
        server_model = torch.load(path_to_file)
        pretrained_model = server_model.model#.to(DEVICE)
        client_data_model = ClientDataModel(pretrained_model=pretrained_model, args=server_model.hparams)
        client_data_model = client_data_model.to(DEVICE)
        client_model = LightningFlowerModel(model=client_data_model,
                                            name=Office31DataModule.get_dataset_name() + "_client_model",
                                            strict_params=True)
        # free some memory
        del server_model
        del pretrained_model
        gc.collect()

        # prepare datamodule, overwrite presplit setting
        #dm_train.pre_split = True
        dm_train.prepare_data()
        dm_train.setup()

        # Start Flower client
        try:
            start_client(server_address=args.host_address,
                         client=ProtoFewShotPlusClient(model=client_model,
                                                       trainer_args=args,
                                                       datamodule=dm_train,
                                                       c_id=args.client_id),
                         grpc_max_message_length=args.max_msg_size)

        except RuntimeError as err:
            print(repr(err))
    else:
        print("[CLIENT] Client " + str(client_id) + " could not load base model")


if __name__ == "__main__":
    # available gpu checks
    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        # clear the cache of the current device
        torch.cuda.empty_cache()
        print("[CLIENT] Using CUDA acceleration")
    else:
        print("[CLIENT] Using CPU acceleration")

    # run main
    main()

    # clear cuda cache
    torch.cuda.empty_cache()
    print("[CLIENT] Graceful shutdown")
