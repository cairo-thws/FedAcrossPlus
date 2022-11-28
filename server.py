import gc
import os
import signal
import torch

from argparse import ArgumentParser

# TorchVision
import pytorch_lightning

# Flower framework
from flwr.server import start_server, ServerConfig
#from flwr.common import weights_to_parameters

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

from strategy import ProtoFewShotPlusStrategy
from common import add_project_specific_args, signal_handler_free_cuda, Defaults
from models import ServerDataModel

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
        self.source_prototypes = prototypes
        self.source_classes = prototype_classes

    def get_initial_params(self):
        print("[SERVER] Providing initial server params to strategy")
        return self.get_flwr_params()

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
                _, preds = model(data)
                class_protos.append(torch.mean(preds, dim=0))
        class_prototypes = torch.stack(class_protos, 0)
    print("[MODEL] Finished creating class prototypes.")
    return class_prototypes.detach().clone()


def pre_train_server_model(model, datamodule, trainer_args, create_prototypes=False):
    # Init ModelCheckpoint callback, monitoring "val_loss"
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", verbose=True, auto_insert_metric_name=True)
    early_stopping_callback = EarlyStopping(monitor="classifier_loss", min_delta=0.01, patience=3, verbose=True, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer.from_argparse_args(trainer_args, callbacks=[early_stopping_callback, checkpoint_callback, lr_monitor], deterministic=True)
    static_pt_path_model = os.path.join(trainer_args.dataset_path, "pretrained", datamodule.get_dataset_name() + ".pt")
    static_pt_path_protos = os.path.join(trainer_args.dataset_path, "pretrained", datamodule.get_dataset_name() + "_protos.pt")
    checkpoint_path = trainer_args.ckpt_path if trainer_args.ckpt_path != "" else None
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)
    # update current model
    if checkpoint_callback.best_model_path != "":
        # obtain the best model from checkpoint
        best_model_pl = ServerDataModel.load_from_checkpoint(checkpoint_callback.best_model_path, map_location=DEVICE)
        best_model = best_model_pl.model.to(DEVICE)
        if create_prototypes:
            best_model_prototypes = create_class_prototypes(best_model, datamodule.train_dataloader())
        # save to disk
        torch.save(best_model_pl, static_pt_path_model)
        torch.save(best_model_prototypes, static_pt_path_protos)
        return best_model_pl, best_model_prototypes
    # trainer.validate(model=model, datamodule=datamodule)
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
                        #train_transforms=transform_train,
                        #test_transforms=transform_test,
                        shuffle=True
                        )

    best_source_model = None
    path_to_file = os.path.join("data", "pretrained", str(Office31DataModule.get_dataset_name()) + ".pt")
    model_file_exists = os.path.exists(path_to_file)

    # pre-train the model on plain source data
    if args.pretrain:
        print("[SERVER] Load and test pretrained source model")
        source_model = ServerDataModel(name=str(source_dm.get_dataset_name()),
                                       num_classes=31,
                                       lr=Defaults.SERVER_LR,
                                       momentum=Defaults.SERVER_LR_MOMENTUM,
                                       gamma=Defaults.SERVER_LR_GAMMA,
                                       weight_decay=Defaults.SERVER_LR_WD,
                                       epsilon=Defaults.SERVER_LOSS_EPSILON)

        # move source model to current device
        source_model = source_model.to(DEVICE)

        # print model information
        #print(source_model)

        # start the server-side source training
        best_source_model, best_source_protos = pre_train_server_model(source_model, source_dm, args, create_prototypes=True)
        print("[SERVER] Done. Evaluating pretrained model")

        # evaluation
        evaluate_server_model(best_source_model, source_dm, args)
        return
    elif model_file_exists:
        print("[SERVER] Load and test pretrained source model")
        #best_source_model = ServerDataModel.load_from_checkpoint(
            #checkpoint_path=os.path.join("data", "pretrained", str(Office31DataModule.get_dataset_name()) + ".ckpt"))
        best_source_model = torch.load(os.path.join("data", "pretrained", str(Office31DataModule.get_dataset_name()) + ".pt"))
        best_source_protos = torch.load(
            os.path.join("data", "pretrained", str(Office31DataModule.get_dataset_name()) + "_protos.pt"))
        # move source model and prototypes to current device
        best_source_model = best_source_model.to(DEVICE)
        best_source_protos = best_source_protos.to(DEVICE)
        # evaluation
        evaluate_server_model(best_source_model, source_dm, args)

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
    server_config = ServerConfig(num_rounds=args.num_rounds)

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
--fast_dev_run=False --num_workers=6 --max_epochs=1 --dataset_path="data/" --batch_size_train=64 --batch_size_test=192 --pretrain=False --backbone="resnet50" --num_rounds=3 --min_fit_clients=1 --min_available_clients=1 --min_eval_clients=1
"""