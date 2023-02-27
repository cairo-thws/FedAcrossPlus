import gc
import os
import random
import signal
import torch
import copy

from argparse import ArgumentParser

# TorchVision
import pytorch_lightning

# Flower framework
from flwr.server import start_server, ServerConfig
from flwr.common import ndarrays_to_parameters

# Pytorch/ Lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# LightningFlower
#import officeHome.image_source
from lightningflower.model import LightningFlowerModel
from lightningflower.server import LightningFlowerServer
from lightningflower.data import LightningFlowerData

# lightningdata wrappers
from lightningdata.modules.domain_adaptation.office31_datamodule import Office31DataModule

# project imports
from torch.utils.data import Subset, DataLoader

import common
from strategy import ProtoFewShotPlusStrategy
from common import add_project_specific_args, signal_handler_free_cuda, Defaults, test_prototypes, NetworkType
from models import ServerDataModel


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
os.environ["GRPC_VERBOSITY"] = "debug"

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


class LightningFlowerServerModel(LightningFlowerModel):
    def __init__(self, model, prototypes, prototype_classes, name="", strict_params=False):
        super().__init__(model=model, name=name, strict_params=strict_params)
        # convert tensor to ndarray
        self.source_prototypes = prototypes
        self.source_classes = prototype_classes

    def get_source_classes(self):
        return self.source_classes

    def get_initial_params(self):
        print("[SERVER] Providing initial server params to strategy")
        return ndarrays_to_parameters(self.get_prototypes().cpu().detach().numpy())

    def get_prototypes(self):
        print("[SERVER] Providing converted ndarray protos to strategy")
        return self.source_prototypes.clone()

    def get_flwr_params(self):
        # get the weights
        weights = [val.cpu().numpy() for key, val in self.model.state_dict().items()]
        # global feature extractor weights
        #weights = ([val.cpu().numpy() for key, val in self.model.feature_extractor.state_dict().items()])
        # global classifier weights
        #weights.extend([val.cpu().numpy() for key, val in self.model.global_classifier.state_dict().items()])
        # source gmm weights
        #weights.extend(self.model.source_gmm._get_parameters())
        return None#weights_to_parameters(weights)


def check_dataset_mean(model, path, dataloader):
    mean_file_exists = os.path.exists(path)
    if mean_file_exists:
        dataset_mean = torch.load(path, map_location=DEVICE)
    else:
        dataset_mean = generate_dataset_mean(model, dataloader)
        torch.save(dataset_mean, path)
    dataset_mean = dataset_mean.to(DEVICE)
    return dataset_mean


def generate_dataset_mean(model, data_loader: DataLoader):
    print("[MODEL] Creating dataset mean tensor, please wait ...")
    with torch.no_grad():
        model.eval()
        dataset_mean = []
        for data, _ in data_loader:
            data = data.to(DEVICE)
            f, _ = model(data)
            dataset_mean.append(torch.sum(f, dim=0))
    mean_tensor = torch.stack(dataset_mean, 0)
    ds_size = len(data_loader.dataset)
    mean_tensor = torch.sum(mean_tensor, dim=0)/ds_size
    print("[MODEL] Finished creating mean tensor.")
    return mean_tensor.detach().clone()


def create_class_prototypes(model, data_loader: DataLoader):
    # subsets = {target: Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y == target]) for _, target in
    # train_set.class_to_idx.items()}
    print("[MODEL] Creating class prototypes, please wait ...")
    with torch.no_grad():
        model.eval()
        class_protos = []
        # for i in range(train_set.num_classes):
        for i in range(len(data_loader.dataset.dataset.classes)):
            # idx_subset = train_set.labels_to_idx[i]
            idx_subset = [j for j, (_, y) in enumerate(data_loader.dataset) if y == i]
            subset = Subset(data_loader.dataset, idx_subset)
            dataloader = DataLoader(subset, batch_size=len(idx_subset))
            for data, labels in dataloader:
                data = data.to(DEVICE)
                f, _ = model(data)
                class_protos.append(torch.mean(f, dim=0))
        class_prototypes = torch.stack(class_protos, 0)
    print("[MODEL] Finished creating class prototypes.")
    return class_prototypes.detach().clone()


def pre_train_server_model(model, datamodule, trainer_args, create_prototypes=False):
    # Init ModelCheckpoint callback, monitoring "val_loss"
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", verbose=True, auto_insert_metric_name=True, mode="max")
    early_stopping_callback = EarlyStopping(monitor="classifier_loss", min_delta=0.005, patience=5, verbose=True, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer.from_argparse_args(trainer_args, callbacks=[early_stopping_callback, checkpoint_callback, lr_monitor], deterministic=True)
    static_pt_path_model = os.path.join(trainer_args.dataset_path, "pretrained", trainer_args.net + "_" + datamodule.get_dataset_name() + ".pt")
    static_pt_path_protos = os.path.join(trainer_args.dataset_path, "pretrained", trainer_args.net + "_" + datamodule.get_dataset_name() + "_protos.pt")

    checkpoint_path = trainer_args.ckpt_path if trainer_args.ckpt_path != "" else None
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)
    # update current model
    if checkpoint_callback.best_model_path != "":
        # obtain the best model from checkpoint
        best_model_pl = ServerDataModel.load_from_checkpoint(checkpoint_callback.best_model_path, map_location=DEVICE)
        best_model = best_model_pl.model.to(DEVICE)
        if create_prototypes:
            best_model_prototypes = create_class_prototypes(best_model, datamodule.train_dataloader())
        # save state dict of best model to disk
        torch.save(best_model_pl.state_dict(), static_pt_path_model)
        # save prototypes of best model to disk
        torch.save(best_model_prototypes, static_pt_path_protos)
        return best_model_pl, best_model_prototypes
    # trainer.validate(model=model, datamodule=datamodule)
    print("[MODEL] No best model available")
    return None, None


def evaluate_server_model(model, datamodule, trainer_args):
    trainer = Trainer.from_argparse_args(trainer_args,
                                         deterministic=True,
                                         logger=False,
                                         enable_checkpointing=False)
    trainer.test(model=model, datamodule=datamodule, verbose=True)


def get_source_train_augmentation():
    """ Train data augmentation on source data here"""
    pass


def get_source_test_augmentation():
    """ Test data augmentation on source data here"""
    pass


def evaluate_server_prototypes(best_source_model, best_source_protos, source_dm, args, dataset_mean=None):
    episodic_categories = random.sample(range(0, source_dm.num_classes), args.N)
    loaders = common.create_fewshot_loaders(source_dm, episodic_categories, args.K)
    best_source_protos = best_source_protos.to(DEVICE)
    best_source_model = best_source_model.to(DEVICE)
    acc = test_prototypes(best_source_model, best_source_protos[episodic_categories], loaders, DEVICE, NetworkType.PROTOTYPICAL, dataset_mean)
    print("[SERVER] Accuracy of class prototypes on source training data is " + str(acc))


def main() -> None:
    parser = ArgumentParser()
    # project-specific arguments
    parser = add_project_specific_args(parser)
    # Data-specific arguments
    parser = LightningFlowerData.add_data_specific_args(parser)
    # server-specific arguments
    parser = LightningFlowerServer.add_server_specific_args(parser)
    # server side evaluation training configuration
    parser = Trainer.add_argparse_args(parser)
    # strategy-specific arguments
    parser = ProtoFewShotPlusStrategy.add_strategy_specific_args(parser)
    # parse args
    args = parser.parse_args()

    # SEED everything
    pytorch_lightning.seed_everything(seed=42)

    # PREPARE SOURCE DATASET
    dataset = Office31DataModule
    # the first domain is server source domain
    source_idx = 0
    domain = dataset.get_domain_names()[source_idx]
    transform_train = get_source_train_augmentation()
    transform_test = get_source_test_augmentation()
    source_dm = dataset(data_dir=args.dataset_path,
                        domain=domain,
                        batch_size=args.batch_size_train,
                        num_workers=args.num_workers,
                        shuffle=True
                        )

    best_source_model = None
    path_to_file = os.path.join("data", "pretrained", args.net + "_" + str(Office31DataModule.get_dataset_name()) + ".pt")
    path_to_protos = os.path.join("data", "pretrained", args.net + "_" + str(Office31DataModule.get_dataset_name()) + "_protos.pt")
    path_to_mean_file = os.path.join("data", "pretrained", args.net + "_" + str(Office31DataModule.get_dataset_name()) + "_mean.pt")
    model_file_exists = os.path.exists(path_to_file)

    # pre-train the model on plain source data
    if args.pretrain:
        if not model_file_exists:
            print("[SERVER] Train and test pretrained source model and create prototypes")
            source_model = common.create_empty_server_model(name=str(source_dm.get_dataset_name()),
                                                            num_classes=31,
                                                            lr=Defaults.SERVER_LR,
                                                            momentum=Defaults.SERVER_LR_MOMENTUM,
                                                            gamma=Defaults.SERVER_LR_GAMMA,
                                                            weight_decay=Defaults.SERVER_LR_WD,
                                                            epsilon=Defaults.SERVER_LOSS_EPSILON,
                                                            net=args.net,
                                                            pretrain=True)

            # move source model to current device
            source_model = source_model.to(DEVICE)

            # start the server-side source training
            best_source_model, best_source_protos = pre_train_server_model(model=source_model,
                                                                           datamodule=source_dm,
                                                                           trainer_args=args,
                                                                           create_prototypes=True)

            # create dataset mean
            source_dm.prepare_data()
            source_dm.setup()
            dataset_mean = check_dataset_mean(best_source_model, path_to_mean_file, source_dm.train_dataloader())
            best_source_model.set_training_dataset_mean(dataset_mean)
        else:
            print("[SERVER] Load and test pretrained source model and create prototypes on demand")
            best_source_model = common.create_empty_server_model(name=str(source_dm.get_dataset_name()),
                                                                 num_classes=31,
                                                                 lr=Defaults.SERVER_LR,
                                                                 momentum=Defaults.SERVER_LR_MOMENTUM,
                                                                 gamma=Defaults.SERVER_LR_GAMMA,
                                                                 weight_decay=Defaults.SERVER_LR_WD,
                                                                 epsilon=Defaults.SERVER_LOSS_EPSILON,
                                                                 net=args.net,
                                                                 pretrain=False)
            best_source_model.load_state_dict(torch.load(path_to_file, map_location=DEVICE))

            # calculate source prototypes
            if os.path.exists(path_to_protos):
                best_source_protos = torch.load(path_to_protos)
            else:
                source_dm.prepare_data()
                source_dm.setup()
                best_source_protos = create_class_prototypes(best_source_model, source_dm.train_dataloader())
                torch.save(best_source_protos, path_to_protos)

            # create dataset mean
            source_dm.prepare_data()
            source_dm.setup()
            dataset_mean = check_dataset_mean(best_source_model, path_to_mean_file, source_dm.train_dataloader())
            best_source_model.set_training_dataset_mean(dataset_mean)

        # move data and model to DEVICE
        best_source_model = best_source_model.to(DEVICE)
        best_source_protos = best_source_protos.to(DEVICE)

        print("[SERVER] Done. Evaluating pretrained model")
        # evaluation
        source_dm.prepare_data()
        source_dm.setup()
        evaluate_server_model(best_source_model, source_dm, args)
    elif model_file_exists:
        print("[SERVER] Load and test pretrained source model")
        best_source_model = common.create_empty_server_model(name=str(source_dm.get_dataset_name()),
                                                             num_classes=31,
                                                             lr=Defaults.SERVER_LR,
                                                             momentum=Defaults.SERVER_LR_MOMENTUM,
                                                             gamma=Defaults.SERVER_LR_GAMMA,
                                                             weight_decay=Defaults.SERVER_LR_WD,
                                                             epsilon=Defaults.SERVER_LOSS_EPSILON,
                                                             net=args.net,
                                                             pretrain=False)
        best_source_model.load_state_dict(torch.load(path_to_file, map_location=DEVICE))
        best_source_protos = torch.load(path_to_protos)

        # move source model and prototypes to current device
        best_source_model = best_source_model.to(DEVICE)
        best_source_protos = best_source_protos.to(DEVICE)

        # create dataset mean
        source_dm.prepare_data()
        source_dm.setup()
        dataset_mean = check_dataset_mean(best_source_model, path_to_mean_file, source_dm.train_dataloader())
        best_source_model.set_training_dataset_mean(dataset_mean)

        # evaluation
        evaluate_server_model(best_source_model, source_dm, args)

    # evaluation of source prototypes on source test set
    evaluate_server_prototypes(best_source_model, best_source_protos, source_dm, args, dataset_mean)

    # bring source model into server mode and wrap it into LF
    lightning_flower_server_model = LightningFlowerServerModel(model=best_source_model,
                                                               prototypes=best_source_protos,
                                                               prototype_classes=source_dm.classes,
                                                               name=Office31DataModule.get_dataset_name() + "_model",
                                                               strict_params=True)
    # release memory of source data
    del source_dm
    gc.collect()

    # STRATEGY CONFIGURATION: pass pretrained model to server
    strategy = ProtoFewShotPlusStrategy.from_argparse_args(args,
                                                           server_model=lightning_flower_server_model,
                                                           server_trainer_args=args)
    # SERVER SETUP
    server = LightningFlowerServer(strategy=strategy)

    # Server config
    server_config = ServerConfig(num_rounds=args.num_rounds, round_timeout=300.0)

    try:
        # Start Lightning Flower server for three rounds of federated learning
        start_server(server=server,
                     server_address=args.host_address,
                     config=server_config,
                     grpc_max_message_length=args.max_msg_size)
    except RuntimeError as err:
        print(repr(err))


if __name__ == "__main__":
    # available gpu checks
    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        # clear the cache of the current device
        torch.cuda.empty_cache()
        print("[SERVER] Using CUDA acceleration")
    else:
        print("[SERVER] Using CPU acceleration")

    # start and run FL server
    main()

    # clear cuda cache
    torch.cuda.empty_cache()
    print("[SERVER] Graceful shutdown")


"""
# cluster 1 gpu setup
--fast_dev_run=False --net="resnet34" --num_workers=6 --dataset_path="data/" --batch_size_train=128 --batch_size_test=128 --pretrain=False --num_rounds=3 --min_fit_clients=1 --min_available_clients=1 --min_eval_clients=1 --accelerator="gpu" --devices=1 --max_epochs=50 --log_every_n_steps=1 --host_address="localhost:8080
"""