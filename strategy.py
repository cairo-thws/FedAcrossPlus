import os
from typing import Optional, Dict, Tuple, List, Union

# Flwr
from flwr.common import Parameters, Scalar, FitRes, \
    EvaluateRes  # , weights_to_parameters, parameters_to_weights, FitIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

# LightingFlower / Pytorch Lightning
from lightningflower.client import LightningFlowerClient
from lightningflower.strategy import LightningFlowerBaseStrategy
from lightningflower.utility import boolean_string
from pytorch_lightning import Trainer


class ProtoFewShotPlusStrategy(LightningFlowerBaseStrategy, FedAvg):
    """Configurable ProtoFewShotPlus strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(self,
                 fraction_fit,
                 fraction_eval,
                 min_fit_clients,
                 min_eval_clients,
                 min_available_clients,
                 accept_failures,
                 server_model,
                 N,
                 K,
                 server_trainer_args=None,
                 eval_fn=None
                 ) -> None:

        FedAvg.__init__(self,
                        fraction_fit=fraction_fit,
                        fraction_evaluate=fraction_eval,
                        min_fit_clients=min_fit_clients,
                        min_evaluate_clients=min_eval_clients,
                        min_available_clients=min_available_clients,
                        accept_failures=accept_failures,
                        evaluate_fn=eval_fn
                        )
        # init base strategy
        LightningFlowerBaseStrategy.__init__(self)

        # set the config func
        self.on_fit_config_fn = self.fit_round

        self.server_trainer_args = server_trainer_args
        self.server_model = server_model

        # initial parameter from source model
        self.initial_parameters = self.server_model.get_initial_params()

        # few-shot arguments
        self.N = N
        self.K = K

        print("[STRATEGY] Init ProtoFewShotPlus Strategy")

    @staticmethod
    def add_strategy_specific_args(parent_parser):
        # add base LightningFlowerFedAvgStrategy argument group
        parser = parent_parser.add_argument_group("ProtoFewShotPlusStrategy")
        # FewShot specific arguments
        parser.add_argument("--K", type=int, default=7)
        parser.add_argument("--N", type=int, default=10)
        parser.add_argument("--episodes", type=int, default=10)
        parser.add_argument("--network_type", type=str, default="prototypical")
        parser.add_argument("--adaptation_type", type=str, default="mean_embedding")
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

    def fit_round(self, rnd: int) -> Dict:
        """Sends the current server configuration to the client"""
        print("[STRATEGY] Federated Round " + str(rnd))
        ret_dict = dict()
        if rnd == 1:
            print("[STRATEGY] Sending initial prototype configuration to client - N=" + str(self.N) + ", K=" + str(self.K))
            ret_dict["source_classes"] = b",".join(self.server_model.get_source_classes()).decode("utf-8")
            ret_dict["K"] = str(self.K)
            ret_dict["N"] = str(self.N)
        ret_dict["global_round"] = str(rnd)
        ret_dict["training_episodes"] = str(self.server_trainer_args.episodes)
        ret_dict["network_type"] = str(self.server_trainer_args.network_type)
        ret_dict["adaptation_type"] = str(self.server_trainer_args.adaptation_type)
        return ret_dict

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        print("[STRATEGY] Aggregate_fit called")
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        print("[STRATEGY] Server collecting parameters from round " + str(server_round) + " from number of clients=" + str(len(results)))

        for (client, fit_res) in results:
            client_id = fit_res.metrics["client_id"]
            duration = fit_res.metrics["duration"]
            classifier_loss = fit_res.metrics["classifier_loss"]
            status = fit_res.status
            print("[STRATEGY] Client " + str(client_id) + " returned result message= " + status.message + " with duration " + str(duration) + " and classifier_loss=" + str(classifier_loss))

        return None, {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        print("[STRATEGY] Server evaluate on query set in round " + str(
            server_round) + " from number of clients=" + str(len(results)))

        for (client, eval_res) in results:
            client_id = eval_res.metrics["client_id"]
            duration = eval_res.metrics["duration"]
            accuracy = eval_res.metrics["mean_accuracy"]
            deviation = eval_res.metrics["st_dev"]
            status = eval_res.status
            print("[STRATEGY] Client " + str(client_id) + " returned result message= " + status.message + " with duration " + str(duration) + " and mean eval accuracy=" + accuracy + " with deviation=" + str(deviation))
        return None, {}
