"""
MIT License

Copyright (c) 2023 Manuel Roeder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import gc
import os
import random
import signal
import statistics

import torch
import timeit

# config file parser
from jsonargparse import ActionConfigFile

# flower framework
from flwr.client import start_client
from flwr.common.typing import Code, Parameters, Status, EvaluateIns, EvaluateRes
from flwr.common import FitRes, FitIns, GetPropertiesRes, GetPropertiesIns, GetParametersIns, GetParametersRes, parameters_to_ndarrays

# pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.cli import LightningArgumentParser

# lightningflower framework imports
from lightningflower.client import LightningFlowerClient
from lightningflower.data import LightningFlowerData
from lightningflower.model import LightningFlowerModel

# lightningdata wrappers
from lightningdata import Digit5DataModule, Office31DataModule, OfficeHomeDataModule
from lightningdata.modules.domain_adaptation.domainNet_datamodule import DomainNetDataModule

from torch.utils.data import Subset, DataLoader

# project imports
import common
from common import add_project_specific_args, signal_handler_free_cuda, test_prototypes_on_client, create_reduced_fewshot_loaders, \
    parse_network_type, NetworkType, ClientAdaptationType, Defaults, parse_adaptation_type, LogParameters, print_args
from models import ClientDataModel
from domainNet_waste_datamodule import DomainNetWasteDataModule
from officeHome_waste_datamodule import OfficeHomeWasteDataModule


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


class FedAcrossClient(LightningFlowerClient):
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
        # the adapted target prototypes
        self.tuned_prototypes = None
        # the source dataset mean
        self.source_dataset_mean = None
        # few-shot parameters
        self.K = None
        self.N = None
        # client-side learning configuration
        self.episodic_categories = list()
        self.loaders = dict()
        self.training_episodes = None
        self.network_type = None
        self.adaptation_type = None
        self.local_trainer = None
        self.local_loaders = None
        # track the number of federated fit rounds
        self.fed_round_fit = 0

        print("[CLIENT " + str(self.c_id) + "] Init FedAcrossClient with id" + str(c_id))

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

        if "network_type" in ins.config:
            self.network_type = parse_network_type(ins.config["network_type"])

        if "adaptation_type" in ins.config:
            self.adaptation_type = parse_adaptation_type(ins.config["adaptation_type"])

        if ins.parameters.tensors is not None:
            np_list = parameters_to_ndarrays(ins.parameters)
            # saving the most recent prototypes
            self.prototypes = torch.stack([torch.from_numpy(item) for item in np_list[0]], 0)
            self.prototypes = self.prototypes.to(DEVICE)
            # saving the source dataset mean vector
            self.source_dataset_mean = torch.from_numpy(np_list[1])
            self.source_dataset_mean = self.source_dataset_mean.to(DEVICE)
            print("[CLIENT " + str(self.c_id) + "] Received new prototypes and source dataset mean")

    def check_evalIns(self, ins):
        pass

    def generate_base_dataloaders(self):
        """Generates query and support sets according to the N/K parameters"""
        if not self.loaders:
            self.loaders = create_reduced_fewshot_loaders(self.datamodule, self.episodic_categories, self.K)

        print("[CLIENT " + str(self.c_id) + "] Query and support sets generated")

    def finetune_model(self, trainer, train_loader, val_loader=None, create_prototypes=True):
        # start training with limited examples
        trainer.fit(self.localModel.model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # obtain the best model from checkpoint
        #best_model_pl = ClientDataModel.load_from_checkpoint(cb.best_model_path, map_location=DEVICE)
        torch.save(self.localModel.model.state_dict(), "data/pretrained/resnet50_office31_amazon_model_webcam_k_10.pt")
        print("Created webcam model - finished")

        if create_prototypes:
            # move model back on device
            self.localModel.model = self.localModel.model.to(DEVICE)
            with torch.no_grad():
                self.localModel.model.eval()
                protos = []
                for cat in self.episodic_categories:
                    # idx_subset = train_set.labels_to_idx[i]
                    idx_subset = [j for j, (_, y) in enumerate(train_loader.dataset) if y == cat]
                    subset = Subset(train_loader.dataset, idx_subset)
                    dataloader = DataLoader(subset, batch_size=len(idx_subset))
                    for data, labels in dataloader:
                        data = data.to(DEVICE)
                        f, _ = self.localModel.model(data)
                        protos.append(torch.mean(f, dim=0))
                prototypes = torch.stack(protos, 0)
                return trainer.logged_metrics, prototypes.detach().clone()
        return trainer.logged_metrics, None

    def prototypes_adaptation(self, adaptation_type=ClientAdaptationType.END_2_END):
        print("[CLIENT " + str(self.c_id) + "] Adapt global prototypes to target samples")
        if ClientAdaptationType.NONE == adaptation_type:
            print("[CLIENT " + str(self.c_id) + "] Prototype creation using mean embedding vector of target samples")
            # move model back on device
            self.localModel.model = self.localModel.model.to(DEVICE)
            with torch.no_grad():
                self.localModel.model.eval()
                protos = []
                for cat in self.episodic_categories:
                    # idx_subset = train_set.labels_to_idx[i]
                    idx_subset = [j for j, (_, y) in enumerate(self.loaders[0].dataset) if y == cat]
                    subset = Subset(self.loaders[0].dataset, idx_subset)
                    dataloader = DataLoader(subset, batch_size=len(idx_subset))
                    for data, labels in dataloader:
                        data = data.to(DEVICE)
                        f, _ = self.localModel.model(data)
                        protos.append(torch.mean(f, dim=0))
                prototypes = torch.stack(protos, 0)
            return dict(), prototypes.detach().clone()
        elif ClientAdaptationType.END_2_END == adaptation_type:
            print("[CLIENT " + str(self.c_id) + "] End-to-End target adaptation")
            # overwrite class level prototypes
            self.localModel.model.set_class_prototypes(self.prototypes[self.episodic_categories])
            # set source dataset mean
            self.localModel.model.set_source_dataset_mean(self.source_dataset_mean[self.episodic_categories])
            if not self.local_loaders:
                # resample from target images
                self.local_loaders = common.create_reduced_fewshot_loaders(self.datamodule, self.episodic_categories, self.K)
            if not self.local_trainer:
                cb_list = list()
                if self.trainer_config.log_parameters == 1:
                    logparams_cb = LogParameters()
                    cb_list.append(logparams_cb)
                # Init ModelCheckpoint callback, monitoring "val_loss"
                checkpoint_callback = ModelCheckpoint(monitor="val_acc", verbose=True, auto_insert_metric_name=True,
                                                      mode="max")
                #cb_list.append(checkpoint_callback)
                early_stopping_callback = EarlyStopping(monitor="classifier_loss", min_delta=0.001, patience=20,
                                                        verbose=True,
                                                        mode="min")
                cb_list.append(early_stopping_callback)
                lr_monitor = LearningRateMonitor(logging_interval='step')
                cb_list.append(lr_monitor)
                # create new trainer instance for client adaptation
                self.local_trainer = Trainer.from_argparse_args(self.trainer_config, callbacks=cb_list, deterministic=True)
            # fine tune model on few examples
            return self.finetune_model(self.local_trainer, self.loaders[0], self.loaders[2], create_prototypes=True)

    def evaluate_client_model(self):
        total_acc = list()
        test_queries = 5
        print("[CLIENT " + str(self.c_id) + "] Calculating mean few-shot accuracy of class prototypes on target test data over " + str(
            test_queries) + " runs... ")
        for i in range(test_queries):
            acc = test_prototypes_on_client(self.localModel.model, self.tuned_prototypes, self.episodic_categories, self.loaders[1], DEVICE, network_type=NetworkType.PROTOTYPICAL, dataset_mean=self.source_dataset_mean)
            total_acc.append(acc)
        total_mean = statistics.mean(total_acc)
        total_stdev = statistics.stdev(total_acc)
        print("[CLIENT " + str(self.c_id) + "] Mean few-shot accuracy of class prototypes on target test data is " + str(
            total_mean) + " with stdev " + str(total_stdev))
        return total_mean, total_stdev

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

        # update local fit rounds tracker
        self.fed_round_fit = self.fed_round_fit + 1

        # perform end to end local client training
        adaptation_metrics, client_prototypes = self.prototypes_adaptation(self.adaptation_type)

        # persist new prototypes
        self.tuned_prototypes = client_prototypes

        # return metrics
        ret_status = Status(code=Code.OK, message="OK")
        ret_params = Parameters(tensor_type="", tensors=[])
        ret_metrics = dict()
        ret_metrics["duration"] = timeit.default_timer() - fit_begin
        ret_metrics["client_id"] = self.c_id
        if not adaptation_metrics:
            ret_metrics["classifier_loss"] = 0.0
        else:
            ret_metrics["classifier_loss"] = adaptation_metrics["classifier_loss"].item()

        return FitRes(status=ret_status,
                      parameters=ret_params,
                      num_examples=self.K * self.N,
                      metrics=ret_metrics)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # begin time measurement
        eval_begin = timeit.default_timer()

        self.check_evalIns(ins)
        self.generate_base_dataloaders()
        mean_acc, deviation = self.evaluate_client_model()

        ret_metrics = dict()
        ret_metrics["duration"] = timeit.default_timer() - eval_begin
        ret_metrics["client_id"] = self.c_id
        ret_metrics["mean_accuracy"] = str(mean_acc)
        ret_metrics["st_dev"] = str(deviation)

        return EvaluateRes(status=Status(code=Code.OK,
                                         message="OK"),
                           loss=0.0,
                           num_examples=0,
                           metrics=ret_metrics)


def get_client_train_augmentation(client_id, input_image_size=224, normalize=False):
    print("[CLIENT] Client " + str(client_id) + " - Client training augmentation")
    return common.boost_augmentation(resize_size=input_image_size, resnet_normalization=normalize)


def get_client_test_augmentation(client_id, input_image_size=224, normalize=False):
    print("[CLIENT] Client " + str(client_id) + " - No client test augmentation")
    return common.no_augmentation(resize_size=input_image_size, resnet_normalization=normalize)


def main() -> None:
    parser = LightningArgumentParser()
    # paper-specific arguments
    parser = add_project_specific_args(parser)
    # Data-specific arguments
    parser = LightningFlowerData.add_data_specific_args(parser)
    # Trainer-specific arguments
    parser = pl.Trainer.add_argparse_args(parser)
    # Client-specific arguments
    parser = LightningFlowerClient.add_client_specific_args(parser)
    # add parsing from config file
    parser.add_argument('--config_file', action=ActionConfigFile)
    # parse arguments, skip checks
    args = parser.parse_args(_skip_check=True)

    # print args to stdout
    print(common.print_args(args))

    # fixed seeding
    seed_everything(42, workers=True)

    # PREPARE SOURCE DATASET
    if args.dataset == Office31DataModule.get_dataset_name():
        dataset = Office31DataModule
        num_classes = 31
    elif args.dataset == OfficeHomeDataModule.get_dataset_name():
        dataset = OfficeHomeDataModule
        num_classes = 65
    elif args.dataset == Digit5DataModule.get_dataset_name():
        dataset = Digit5DataModule
        num_classes = 10
    elif args.dataset == DomainNetDataModule.get_dataset_name():
        dataset = DomainNetDataModule
        num_classes = 345
    elif args.dataset == DomainNetWasteDataModule.get_dataset_name():
        dataset = DomainNetWasteDataModule
        num_classes = 30
    elif args.dataset == OfficeHomeWasteDataModule.get_dataset_name():
        dataset = OfficeHomeWasteDataModule
        num_classes = 25

    # limit client id number
    client_id = args.client_id % len(dataset.get_domain_names())
    # client ids start with 1, 0 is reserved for server
    domain = dataset.get_domain_names()[client_id]
    dm_train = dataset(data_dir=args.dataset_path,
                       domain=domain,
                       batch_size=args.batch_size_train,
                       num_workers=args.num_workers,
                       drop_last=True,
                       train_transform_fn=get_client_train_augmentation(client_id, normalize=True),
                       test_transform_fn=get_client_test_augmentation(client_id, normalize=True),
                       shuffle=False)

    # server source domain
    source_domain_id = args.subdomain_id
    source_domain = dataset.get_domain_names()[source_domain_id]
    # load pretrained server model
    path_to_file = os.path.join("data", "pretrained", args.net + "_" + str(dataset.get_dataset_name()) + "_" + str(source_domain) + "_model.pt")
    model_file_exists = os.path.exists(path_to_file)
    if model_file_exists:
        print("[CLIENT] Client " + str(client_id) + " Loading model: " + path_to_file)
        server_model = common.create_empty_server_model(name=str(dataset.get_dataset_name() + "_" + str(domain)),
                                                        num_classes=num_classes,
                                                        lr=Defaults.SERVER_LR,
                                                        momentum=Defaults.SERVER_LR_MOMENTUM,
                                                        gamma=Defaults.SERVER_LR_GAMMA,
                                                        weight_decay=Defaults.SERVER_LR_WD,
                                                        epsilon=Defaults.SERVER_LOSS_EPSILON,
                                                        net=args.net,
                                                        optimizer=args.optimizer,
                                                        pretrain=False)
        # move server model to device
        server_model.load_state_dict(torch.load(path_to_file, map_location=DEVICE))
        # remove server part, extract internal model
        pretrained_model = server_model.model
        # wrap into client model
        client_data_model = ClientDataModel(server_model.hparams_initial, pretrained_model=pretrained_model)
        # make sure client model is on device
        client_data_model = client_data_model.to(DEVICE)
        # wrap into LightingFlower model that is compatible with federated learning framework
        client_model = LightningFlowerModel(model=client_data_model,
                                            name=str(dataset.get_dataset_name()) + "_" + str(domain) + "_lf_client_model",
                                            strict_params=True)
        # free some memory
        # @todo check if more memory can be freed
        del server_model
        gc.collect()

        # prepare datamodule manually(PL usually takes care)
        dm_train.prepare_data()
        dm_train.setup()

        # start flwr client
        try:
            start_client(server_address=args.host_address,
                         client=FedAcrossClient(model=client_model,
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

"""
# client 1 gpu setup
--fast_dev_run=False --net="resnet34" --num_workers=4 --max_epochs=50 --dataset_path="data/" --batch_size_train=32 --batch_size_test=32 --log_every_n_steps=1 --client_id=1 --precision=16 --accelerator="gpu" --devices=1 --dataset="officeHome" --check_val_every_n_epoch=10
# client 2 gpu setup
--fast_dev_run=False --net="resnet34" --num_workers=4 --max_epochs=50 --dataset_path="data/" --batch_size_train=32 --batch_size_test=32 --log_every_n_steps=1 --client_id=2 --precision=16 --accelerator="gpu" --devices=1 --dataset="officeHome" --check_val_every_n_epoch=10
"""