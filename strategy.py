import os
from typing import Optional, Dict, Tuple, List

# Flwr
from flwr.common import Parameters, Scalar, FitRes #, weights_to_parameters, parameters_to_weights, FitIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

# LightingFlower / Pytorch Lightning
from lightningflower.client import LightningFlowerClient
from lightningflower.strategy import LightningFlowerBaseStrategy
from lightningflower.utility import boolean_string
from pytorch_lightning import Trainer

# project impoprts
from common import Defaults
#from models import FedShotPlusPlusPhase, phase_to_str


class FedShotPlusPlusStrategy(LightningFlowerBaseStrategy, FedAvg):
    """Configurable FedShot++ strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(self,
                 fraction_fit,
                 fraction_eval,
                 min_fit_clients,
                 min_eval_clients,
                 min_available_clients,
                 accept_failures,
                 server_model,
                 source_data,
                 server_trainer_args=None,
                 eval_fn=None
                 ) -> None:

        FedAvg.__init__(self,
                        fraction_fit=fraction_fit,
                        fraction_eval=fraction_eval,
                        min_fit_clients=min_fit_clients,
                        min_eval_clients=min_eval_clients,
                        min_available_clients=min_available_clients,
                        accept_failures=accept_failures,
                        eval_fn=eval_fn
                        )
        # init base strategy
        LightningFlowerBaseStrategy.__init__(self)

        # set the config func
        self.on_fit_config_fn = FedShotPlusPlusStrategy.fit_round

        self.source_data = source_data
        self.server_trainer_args = server_trainer_args
        self.server_model = server_model

        # initial parameter from source model
        self.initial_parameters = self.server_model.get_initial_params()

        # set internal model mode to server
        #self.server_model.model.mode = "server"

        print("Init FedShotPlusPlus Strategy")

    @staticmethod
    def add_strategy_specific_args(parent_parser):
        # add base LightningFlowerFedAvgStrategy argument group
        parser = parent_parser.add_argument_group("FedShotPlusPlusStrategy")
        # FedAvg specific arguments
        parser.add_argument("--fraction_fit", type=float, default=0.5)
        parser.add_argument("--fraction_eval", type=float, default=0.5)
        parser.add_argument("--min_fit_clients", type=int, default=2)
        parser.add_argument("--min_eval_clients", type=int, default=2)
        parser.add_argument("--min_available_clients", type=int, default=2)
        parser.add_argument("--accept_failures", type=boolean_string, default=True)
        return parent_parser

    def create_trainer_instance(self):
        trainer: Trainer = None
        if self.server_trainer_args:
            trainer = Trainer.from_argparse_args(self.server_trainer_args,
                                                 deterministic=True,
                                                 logger=False,
                                                 enable_checkpointing=False,
                                                 terminate_on_nan=True,
                                                 check_val_every_n_epoch=1,
                                                 detect_anomaly=True)
        else:
            trainer = Trainer(max_epochs=1,
                              logger=False,
                              enable_checkpointing=False,
                              terminate_on_nan=True,
                              deterministic=True,
                              check_val_every_n_epoch=3,
                              detect_anomaly=True)
        return trainer

    @staticmethod
    def fit_round(rnd: int) -> Dict:
        """Sends the current server configuration to the client"""
        print("[STRATEGY] Federated Round " + str(rnd))
        return {}

    def aggregate_fit(
            self,
            rnd: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        """
        # somehow pas
        # s parameters to source model
        print("Server collecting parameters from round " + str(rnd) + " from number of clients=" + str(len(results)))
        # zero-out local gmms list
        self.server_model.model.target_gmm.clear()
        for (client, fit_res) in results:
            client_id = fit_res.metrics["client_id"]
            print("Client id= " + str(client_id) + " delivered results to server")
            client_weights = parameters_to_weights(fit_res.parameters)
            gmm_params = client_weights[-Defaults.GMM_PARAMS_DIGIT5:]
            local_classifier_params = client_weights[:-Defaults.CLASSIFIER_PARAMS_DIGIT5]
            client_gmm = gaussian_mm_from_params(params=gmm_params, warm_start=True)
            self.server_model.model.target_gmm.append((client_id, client_gmm))
            LightningFlowerClient.set_model_parameters(model=self.server_model.model.client_classifiers[client_id],
                                                       parameters=local_classifier_params,
                                                       strict=True)

        # sort by client ids
        self.server_model.model.target_gmm.sort(key=lambda tup: tup[0])
        # freeze the client classifiers
        # self.server_model.model.freeze_client_classifiers(True)

        # create new trainer instance
        trainer = self.create_trainer_instance()

        # train source model in server mode
        trainer.fit(model=self.server_model.model,
                    datamodule=self.source_data)

        # save a model checkpoint with newly updated weights
        cp_path = os.path.join("data", "full", str(self.source_data.get_dataset_name()) + ".ckpt")
        trainer.save_checkpoint(cp_path)

        # extract updated global model weights
        new_weights = self.server_model.get_flwr_params() 

        for (client, fit_res) in results:
            client_id = fit_res.metrics["client_id"]
            duration = fit_res.metrics["duration"]
            phase = fit_res.metrics["phase"]
            print("[STRATEGY] Client " + str(client_id) + " returned result from phase " + phase_to_str(phase) + " with duration " + str(duration))
        """

        return None, {}
