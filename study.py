import os
import sys
import uuid
from argparse import ArgumentParser

import torch
import subprocess
import time
from lightningdata import Office31DataModule, OfficeHomeDataModule, Digit5DataModule
from lightningdata.modules.domain_adaptation.domainNet_datamodule import DomainNetDataModule


def run_server_pretrain_on_domain(server_idx, dataset, subdir, net):
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    source_file = os.path.join(subdir, "out_server_pretrain_source_" + dataset.get_domain_names()[server_idx] + ".txt")
    server_log = open(source_file, 'w')
    try:
        server_call = "python server.py --config_file=config/server_config.yaml --dataset=" + dataset.get_dataset_name() + " --subdomain_id=" + str(
            server_idx) + " --pretrain=True --fast_server_startup=False" + " --net=" + net + " --default_root_dir=" + subdir
        print(server_call)
        server_job = subprocess.Popen(server_call.split(), stdout=server_log, stderr=subprocess.STDOUT, shell=False)
        print("Server pretraining, please wait")
        if server_job.wait() != 0:
            print("[STUDY] There was an error in server pretraining")
    except RuntimeError as err:
        print(repr(err))
        torch.cuda.empty_cache()
    server_log.close()
    print("Server pretraining finished")


def run_server_client_study(server_idx, client_id_1, client_id_2, dataset, subdir, K=5, adaptation_enabled=True, net="resnet34"):
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    if adaptation_enabled:
        adaptation = "end_2_end"
    else:
        adaptation = "none"
    # unique run id
    id = uuid.uuid4()
    source_file = os.path.join(subdir, str(id) + "_out_server_source_" + dataset.get_domain_names()[server_idx] + "_K_" + str(K) +".txt")
    c1_file = os.path.join(subdir, str(id) + "_out_client_source_" + dataset.get_domain_names()[server_idx] + "_target_" + dataset.get_domain_names()[client_id_1] + "_adapt_" + adaptation + "_K_" + str(K) + ".txt")
    c2_file = os.path.join(subdir, str(id) + "_out_client_source_" + dataset.get_domain_names()[server_idx] + "_target_" + dataset.get_domain_names()[client_id_2] + "_adapt_" + adaptation + "_K_" + str(K) + ".txt")
    server_log = open(source_file, 'w')
    c1_log = open(c1_file, 'w')
    c2_log = open(c2_file, 'w')
    try:
        processes = []
        server_call = "python server.py --config_file=config/server_config.yaml --dataset=" + dataset.get_dataset_name() + " --subdomain_id=" + str(server_idx) + " --adaptation_type=" + adaptation + " --net=" + net + " --default_root_dir=" + subdir + " --K=" + str(K)
        print(server_call)
        server_job = subprocess.Popen(server_call.split(), stdout=server_log, stderr=subprocess.STDOUT, shell=False)
        print("Server boot, waiting...")
        time.sleep(10)
        client_1_call = "python client.py --config_file=config/client_config.yaml --dataset=" + dataset.get_dataset_name() + " --client_id=" + str(client_id_1) + " --subdomain_id=" + str(server_idx) + " --net=" + net + " --default_root_dir=" + subdir
        print(client_1_call)
        c1_job = subprocess.Popen(client_1_call.split(), stdout=c1_log, stderr=subprocess.STDOUT, shell=False)
        print("Client 1 boot done")
        time.sleep(5)
        client_2_call = "python client.py --config_file=config/client_config.yaml --dataset=" + dataset.get_dataset_name() + " --client_id=" + str(client_id_2) + " --subdomain_id=" + str(server_idx) + " --net=" + net + " --default_root_dir=" + subdir
        print(client_2_call)
        c2_job = subprocess.Popen(client_2_call.split(), stdout=c2_log, stderr=subprocess.STDOUT, shell=False)

        processes.append(server_job)
        processes.append(c1_job)
        processes.append(c2_job)

        print("Client 2 boot done, waiting for study to shut down...")
        for p in processes:
            if p.wait() != 0:
                print("[STUDY] There was an error")

    except RuntimeError as err:
        print(repr(err))
        torch.cuda.empty_cache()

    # close all logs
    server_log.close()
    c1_log.close()
    c2_log.close()

    return "Study " + str(id) + " finished"



def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--net', type=str, default="resnet34")
    parser.add_argument('--subdir', type=str, default="experiment")
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--adapt', type=int, default=1)

    # parse arguments, skip checks
    args = parser.parse_args()

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

    subdomain_count = len(dataset.get_domain_names())
    K_list = [3, 5, 10]
    folder = os.path.join(args.subdir, dataset.get_dataset_name())

    for server_idx in range(subdomain_count):
        if server_idx == 0:
            client_id_1 = 1
            client_id_2 = 2
        elif server_idx == 1:
            client_id_1 = 0
            client_id_2 = 2
        elif server_idx == 2:
            client_id_1 = 0
            client_id_2 = 1

        # check if server pretrained files exist
        net = args.net
        dataset_name = dataset.get_dataset_name()
        subdomain_name = dataset.get_domain_names()[server_idx]
        model_ending = "model.pt"
        model_filename = net + "_" + dataset_name + "_" + subdomain_name + "_" + model_ending
        model_filepath = os.path.join("data", "pretrained", model_filename)
        run_folder = os.path.join(folder, "source_" + subdomain_name)

        if not os.path.exists(model_filepath):
            # need to do server pretraining:
            run_server_pretrain_on_domain(server_idx, dataset, run_folder, args.net)
            if os.path.exists(model_filepath):
                print("Pretraining model success")
            else:
                print("Pretraining model failure, skipping client adaptation")
                continue

        # no adaptation run
        run_server_client_study(server_idx,
                                client_id_1,
                                client_id_2,
                                dataset,
                                K=3, #wont be used without adaptation
                                subdir=run_folder,
                                adaptation_enabled=False,
                                net=args.net)
        for K in K_list:
            run_server_client_study(server_idx,
                                    client_id_1,
                                    client_id_2,
                                    dataset,
                                    K=K,
                                    subdir=run_folder,
                                    adaptation_enabled=True)


    print("Finished study")



if __name__ == "__main__":
    main()
    # clear cache
    torch.cuda.empty_cache()