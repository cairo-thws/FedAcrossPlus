import os
import uuid
from argparse import ArgumentParser

import torch
import subprocess
import time
from lightningdata import Office31DataModule, OfficeHomeDataModule, Digit5DataModule, DomainNetDataModule

from domainNet_waste_datamodule import DomainNetWasteDataModule
from officeHome_waste_datamodule import OfficeHomeWasteDataModule


def run_server_pretrain_on_domain(server_idx, dataset, subdir, net, N):
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    source_file = os.path.join(subdir, "out_server_pretrain_source_" + dataset.get_domain_names()[server_idx] + ".txt")
    server_log = open(source_file, 'w')
    try:
        root_dir = subdir.replace(" ", "")
        server_call = "python server.py --config_file=config/server_config.yaml --dataset=" + dataset.get_dataset_name() + " --subdomain_id=" + str(
            server_idx) + " --pretrain=True --fast_server_startup=False" + " --net=" + net + " --default_root_dir=" + root_dir + " --N=" + str(N)
        print(server_call)
        server_job = subprocess.Popen(server_call.split(), stdout=server_log, stderr=subprocess.STDOUT, shell=False)
        print("Server pretraining, please wait")
        if server_job.wait() != 0:
            print("[STUDY] There was an error in server pretraining")
    except RuntimeError as err:
        print(repr(err))
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    server_log.close()
    print("Server pretraining finished")


def run_server_client_study(server_idx, client_ids, dataset, subdir, N=31, K=5, adaptation_enabled=True, net="resnet34"):
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    if adaptation_enabled:
        adaptation = "end_2_end"
    else:
        adaptation = "none"

    # numer of clients to participate in federated training
    num_clients = str(len(client_ids))

    # unique run id
    id = uuid.uuid4()

    try:
        source_file = os.path.join(subdir, str(id) + "_out_server_source_" + dataset.get_domain_names()[server_idx] + "_K_" + str(K) +".txt")
        server_log = open(source_file, 'w')
        # start the server
        root_dir = subdir.replace(" ", "")
        server_call = "python server.py --config_file=config/server_config.yaml --dataset=" + dataset.get_dataset_name() + " --subdomain_id=" + str(
            server_idx) + " --adaptation_type=" + adaptation + " --net=" + net + " --default_root_dir=" + root_dir + " --K=" + str(K) + " --N=" + str(N) + " --min_fit_clients=" + num_clients + " --min_available_clients=" + num_clients + " --min_eval_clients=" + num_clients
        print(server_call)
        server_job = subprocess.Popen(server_call.split(), stdout=server_log, stderr=subprocess.STDOUT, shell=False)
        print("Server boot, waiting...")
        time.sleep(10)

        # client logic
        study_tracker = list() # list with client id, log file path and job
        for client_id in client_ids:
            # file path
            fp = os.path.join(subdir, str(id) + "_out_client_source_" + dataset.get_domain_names()[server_idx] + "_target_" + dataset.get_domain_names()[client_id] + "_adapt_" + adaptation + "_K_" + str(K) + ".txt")
            log = open(fp, 'w')
            client_call = "python client.py --config_file=config/client_config.yaml --dataset=" + dataset.get_dataset_name() + " --client_id=" + str(client_id) + " --subdomain_id=" + str(server_idx) + " --net=" + net + " --default_root_dir=" + root_dir
            print(client_call)
            job = subprocess.Popen(client_call.split(), stdout=log, stderr=subprocess.STDOUT, shell=False)
            study_tracker.append((client_id, (log, job)))
            time.sleep(5)

        print("[STUDY] All clients booted, study running ...")
        for client in study_tracker:
            if client[1][1].wait() != 0:
                print("[STUDY] There was an error")

    except RuntimeError as err:
        print(repr(err))
        torch.cuda.empty_cache()
        # close the log files
        for client in study_tracker:
            client[1][0].close()
        return "Study " + str(id) + " failed"

    # close the log files
    for client in study_tracker:
        client[1][0].close()

    return "Study " + str(id) + " finished"


def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--net', type=str, default="resnet34")
    parser.add_argument('--subdir', type=str, default="experiment")
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--N', type=int, default=0)
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
    elif args.dataset == DomainNetWasteDataModule.get_dataset_name():
        dataset = DomainNetWasteDataModule
        num_classes = 30
    elif args.dataset == OfficeHomeWasteDataModule.get_dataset_name():
        dataset = OfficeHomeWasteDataModule
        num_classes = 25

    # how many subdomains exist
    subdomain_count = len(dataset.get_domain_names())
    # subdomain id list from 0 to subdomain_count
    subdomain_ids = list(range(0, subdomain_count))

    # parameter K for this study
    K_list = [3, 5, 10]
    folder = os.path.join(args.subdir, dataset.get_dataset_name())

    for server_idx in subdomain_ids:
        # get the full list of available domain ids
        c_ids = subdomain_ids.copy()
        # remove the server id from the list
        c_ids.remove(server_idx)

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
            run_server_pretrain_on_domain(server_idx=server_idx,
                                          dataset=dataset,
                                          subdir=run_folder,
                                          net=args.net,
                                          N=num_classes)
            if os.path.exists(model_filepath):
                print("Pretraining model success")
            else:
                print("Pretraining model failure, skipping client adaptation")
                continue

        if args.N == 0:
            N = num_classes
        else:
            N = args.N

        # no adaptation run
        run_server_client_study(server_idx,
                                c_ids,
                                dataset,
                                N=N,
                                K=3, #wont be used without adaptation
                                subdir=run_folder,
                                adaptation_enabled=False,
                                net=args.net)
        for K in K_list:
            run_server_client_study(server_idx,
                                    c_ids,
                                    dataset,
                                    N=N,
                                    K=K,
                                    subdir=run_folder,
                                    adaptation_enabled=True,
                                    net=args.net)

    print("Finished study")


if __name__ == "__main__":
    # clear cache
    torch.cuda.empty_cache()
    main()
    # clear cache
    torch.cuda.empty_cache()