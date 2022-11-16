import os

# Argument parser
from argparse import ArgumentParser

# TorchVision
from flwr.common import FitRes, FitIns
from torch.utils.data import DataLoader
from torchvision import transforms

# flower framework
import flwr as fl
from flwr.client import start_client

# pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

# lightningflower framework imports
from lightningflower.client import LightningFlowerClient
from lightningflower.data import LightningFlowerData
from lightningflower.model import LightningFlowerModel

# lightningdata wrappers
from lightningdata.modules.domain_adaptation.officeHome_datamodule import OfficeHomeDataModule
# from lightningdata.common.pre_process import ResizeImage

# project imports
from common import add_project_specific_args, Defaults

import officeHome.image_source
from models import OfficeHomeModel

import timeit

"""
If you get an error like: “failed to connect to all addresses” “grpc_status”:14 
Then uncomment the lines bellow:
"""
if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]


class FedShotPlusPlusClient(LightningFlowerClient):
    def __init__(self,
                 model,
                 trainer_args,
                 c_id,
                 datamodule=None,
                 train_ds=None,
                 test_ds=None,
                 train_sampler=None):
        super().__init__(model=model,
                         trainer_args=trainer_args,
                         c_id=c_id,
                         datamodule=datamodule,
                         train_ds=train_ds,
                         test_ds=test_ds,
                         train_sampler=train_sampler)
        print("[CLIENT] Init FedShotPlusPlusClient with id" + str(c_id))

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
        print(f"[CLIENT]  Client {self.c_id}: fit")
        weights: fl.common.Weights = fl.common.parameters_to_weights(ins.parameters)
        # read configuration
        config = ins.config
        fed_round = config["rnd"]
        fed_phase = config["phase"]
        # begin time measurement
        fit_begin = timeit.default_timer()
        # update local client model parameters
        self.set_parameters(weights)
        # fetch data loader
        data_loader = self.datamodule
        # update model phase
        self.localModel.model.update_state(FedShotPlusPlusMode.CLIENT, fed_phase)
        if fed_phase == FedShotPlusPlusPhase.SELF_SUPERVISED_LOSS:
            # use some preconfigured values for rotation loss
            trainer = pl.Trainer.from_argparse_args(self.trainer_config,
                                                    max_epochs=1,
                                                    logger=False,
                                                    enable_checkpointing=False,
                                                    terminate_on_nan=True,
                                                    deterministic=True,
                                                    detect_anomaly=True)
            trainer.fit(model=self.localModel.model, train_dataloaders=data_loader)
        elif fed_phase == FedShotPlusPlusPhase.INFORMATION_MAXIMIZATION:
            print("[CLIENT] Starting post self supervised loss")
            # use some preconfigured values for post ssl
            trainer = pl.Trainer.from_argparse_args(self.trainer_config,
                                                    max_epochs=1,
                                                    logger=False,
                                                    enable_checkpointing=False,
                                                    terminate_on_nan=True,
                                                    deterministic=True,
                                                    detect_anomaly=True)
            # @todo add early stopping / model checkpointing
            trainer.fit(model=self.localModel.model, train_dataloaders=data_loader)
        elif fed_phase == FedShotPlusPlusPhase.LABEL_TRANSFER:
            # @todo impl
            print("[CLIENT]  Starting label transfer")
        elif fed_phase == FedShotPlusPlusPhase.NORMAL:
            # @todo impl
            print("[CLIENT] Starting normal operation")
        # calculate nr. of examples used by Trainer for train
        num_train_examples = (data_loader.batch_size * trainer.num_training_batches)
        # return updated model parameters
        weights_prime: fl.common.Weights = self.get_trainable_weights()
        params_prime = fl.common.weights_to_parameters(weights_prime)
        metrics = {}
        metrics["duration"] = timeit.default_timer() - fit_begin
        metrics["client_id"] = self.c_id
        metrics["phase"] = fed_phase
        return FitRes(parameters=params_prime,
                      num_examples=num_train_examples,
                      metrics=metrics)


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
    dataset = OfficeHomeDataModule

    # limit client id number
    client_id = args.client_id % len(dataset.get_domain_names())
    # client ids start with 1, 0 is reserved for server
    domain = dataset.get_domain_names()[client_id]
    dm_train = dataset(data_dir=args.dataset_path,
                       domain=domain,
                       batch_size=args.batch_size_train,
                       num_workers=args.num_workers,
                       train_transforms=officeHome.image_source.image_train(),
                       test_transforms=officeHome.image_source.image_test(),
                       drop_last=True,
                       shuffle=False) # do not shuffle data for self supervised label discovery

    # load pretrained server model
    net = OfficeHomeModel.load_from_checkpoint(checkpoint_path=os.path.join("data", "pretrained", str(OfficeHomeDataModule.get_dataset_name()) + ".ckpt"), mode=FedShotPlusPlusMode.CLIENT, phase=FedShotPlusPlusPhase.SELF_SUPERVISED_LOSS, strict=False)
    client_model = LightningFlowerModel(model=net,
                                        name="office_home_model",
                                        strict_params=True)

    # Start Flower client
    try:
        start_client(server_address=args.host_address,
                     client=FedShotPlusPlusClient(model=client_model,
                                                  trainer_args=args,
                                                  datamodule=dm_train,
                                                  c_id=args.client_id),
                     grpc_max_message_length=args.max_msg_size)
        
    except RuntimeError as err:
        print(repr(err))


if __name__ == "__main__":
    main()
