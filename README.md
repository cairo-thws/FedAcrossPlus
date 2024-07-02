
# [FedAcross+] Domain Borders Are There to Be Crossed With Federated Few-Shot Adaptation

This respository is an implementation of the algorithm proposed in the paper "Domain Borders Are There to Be Crossed With Federated Few-Shot Adaptation".
Please note that the code for the streaming extension is currently in integration and not publicly available.

This code was built on [M. Roeder's FedAcross repository](https://github.com/cairo-thws/FedAcross).



An efficient and  scalable Federated Learning framework designed specifically for
real-world client adaptation in production environments.

## Prerequisites
The project is optimized to run with Pytorch 3.8

## Dependency Installation

Fetch the dependencies for FedAcross ( will install LightningFlower, a Lightning-powered FL framework and datamodules extension )
```sh
pip install -r requirements.txt
```
Install CUDA drivers as needed. 
## Data sets

All data sets from the experiments are automatically downloaded and unpacked vie lightningdata_modules extension.
If you do not have access to the Google APIs, download and unpack the data set under inspection manually from
```sh
https://drive.google.com/drive/folders/1YlAUF_SrXTUkVSIxUvUDuPXIgUIOnNSR?usp=drive_link
```
before running the server or client script. Make sure to stick to the folder structure, e.g.:
```
FEDACROSS_ROOT/data/office31/amazon/data.db
```

## Run
To boot up the server-client environment, the server script is the first executable to run, bringing up the Flower-based central server instance.
After pre-training finished on the server, the script will signal that federated learning is ready. Subsequently, clients may join the FL cycle and adaptation
starts depending on the server configuration.
### Server
The server script is controlled via command line arguments as well as configuration files.
This project ships with a pre-configured configuration file for the Office31 data (server pretraining on Amazon domain, clients fine-tuned on Webcam and DSLR) set to be found at
```
FEDACROSS_ROOT/config/server_config.yaml
```
To start the server pre-training, make sure all dependencies are installed and that the data set under inspection is ready( See Data sets section ):
```
python server.py --config_file="config/server_config.yaml"
```
### Clients
Clients can connect to the gRPC socket locally or remote, depending on your setup.
To start pre-configured client with different ids, run
```
python client.py --config_file="config/client_config.yaml" --client_id=1
python client.py --config_file="config/client_config.yaml" --client_id=2
```
## Study
We developed a study script to automate the process of pre-training and fine-tuning with all combinations
of domains included in DA-specific data sets. To start a study, configure your server/client configuration file
and run:
```
python study.py --dataset="office31" --net="resnet50" --subdir="study_1" --K=5
```
with dataset referring to the set under inspection, net denoting the backbone, subdir pointing to the directory to log to, and K being the
number of support samples per class available for client fine-tuning.

## Server-Client Deployment
Since the server-client architecture of FedAcross is build upon Flower, follow their installation instructions for on-device deployment (e.g. on Nvidia Jetson devices or RaspberryPis)
```
https://flower.dev/
https://flower.dev/blog/2020-12-16-running_federated_learning_applications_on_embedded_devices_with_flower/
```

## Logging
Pretraining results (model parameters, hyperparameters, source prototypes, training telemetry from PL Trainer) are automatically checkpointed via PyTorch Lightning.
Per default, trainings are logged via Tensorboard logger.

To view the logs, make sure to install tensorboard first:
```
pip install  tensorboard
```
and open the logging directory with
```
tensorboard --logdir=MYLOGDIR
```

## Need help? Did I forget to give credits? Please contact me -
manuel.roeder@thws.de - I am glad to help

