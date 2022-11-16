import os
from argparse import ArgumentParser

# TorchVision
import pytorch_lightning

# Flower framework
from flwr.server import start_server
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
from strategy import FedShotPlusPlusStrategy
from common import add_project_specific_args, Defaults
from models import ServerModel

"""
If you get an error like: “failed to connect to all addresses” “grpc_status”:14 
Then uncomment the lines bellow:
"""
if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]


class LightningFlowerServerModel(LightningFlowerModel):
    def __init__(self, model, name="", strict_params=False):
        super().__init__(model=model, name=name, strict_params=strict_params)

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


def pre_train_server_model(model, datamodule, trainer_args):
    # Init ModelCheckpoint callback, monitoring "val_loss"
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", verbose=True, auto_insert_metric_name=True)
    early_stopping_callback = EarlyStopping(monitor="classifier_loss", min_delta=0.01, patience=2, verbose=True, mode="max")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer.from_argparse_args(trainer_args, callbacks=[early_stopping_callback, checkpoint_callback, lr_monitor], deterministic=True)
    static_ckpt_path = os.path.join(trainer_args.dataset_path, "pretrained", datamodule.get_dataset_name() + ".ckpt")
    checkpoint_path = trainer_args.ckpt_path if trainer_args.ckpt_path != "" else None
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)
    # update mode and phase after pre-training
    trainer.save_checkpoint(static_ckpt_path)
    # trainer.validate(model=model, datamodule=datamodule)


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
    #parser = FedShotPlusPlusStrategy.add_strategy_specific_args(parser)
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

    source_model = None
    # pre-train the model on plain source data
    if args.pretrain:
        print("[SERVER] Pretrain source model")
        source_model = ServerModel(name=str(source_dm.get_dataset_name()),
                                   num_classes=31,
                                   lr=Defaults.SERVER_LR,
                                   momentum=Defaults.SERVER_LR_MOMENTUM,
                                   gamma=Defaults.SERVER_LR_GAMMA,
                                   weight_decay=Defaults.SERVER_LR_WD,
                                   epsilon=Defaults.SERVER_LOSS_EPSILON)
        print(source_model)
        pre_train_server_model(source_model, source_dm, args)
        print("[SERVER] Finished model pre-training")
        return
    else:
        print("[SERVER] Load and test pretrained source model")

        source_model = ServerModel.load_from_checkpoint(
            checkpoint_path=os.path.join("data", "pretrained", str(Office31DataModule.get_dataset_name()) + ".ckpt"))

    """
    # bring source model into server mode and wrap it into LF
    lightning_flower_server_model = FedShotPlusPlusServerModel(model=source_model,
                                                               name=args.model_name,
                                                               strict_params=True)

    # STRATEGY CONFIGURATION: pass pretrained model to server
    strategy = FedShotPlusPlusStrategy.from_argparse_args(args,
                                                          server_model=lightning_flower_server_model,
                                                          source_data=source_dm,
                                                          server_trainer_args=args)
    # SERVER SETUP
    server = LightningFlowerServer(strategy=strategy)

    try:
        # Start Lightning Flower server for three rounds of federated learning
        start_server(server=server,
                     server_address=args.host_address,
                     config={"num_rounds": args.num_rounds},
                     grpc_max_message_length=args.max_msg_size)
    except RuntimeError as err:
        print(repr(err))
    """


if __name__ == "__main__":
    main()


"""
--fast_dev_run=False --num_workers=6 --max_epochs=1 --dataset_path="data/" --batch_size_train=64 --batch_size_test=192 --pretrain=False --backbone="resnet50" --num_rounds=3 --min_fit_clients=1 --min_available_clients=1 --min_eval_clients=1
"""