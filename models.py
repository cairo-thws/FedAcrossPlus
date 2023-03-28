import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy
import resnet as backbones
import classifier

LOG_INPUT_IMG_EVERY_N_BATCH = 20

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = F.one_hot(targets.squeeze(1), self.num_classes)
        #targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr'] * param_group['lr_mult']
        # @todo: see if this works here
        #param_group['weight_decay'] = 1e-3
        #param_group['momentum'] = 0.9
        #param_group['nesterov'] = True
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


class DataModelBase(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # training dataset mean
        self.training_dataset_mean = None
        self.input_shape = None

    def set_training_dataset_mean(self, mean_tensor):
        self.training_dataset_mean = mean_tensor

    def forward(self, x):
        return self.model(x)


class ServerDataModel(DataModelBase):
    def __init__(self, name, num_classes, lr, momentum, gamma, weight_decay, epsilon, optimizer="sgd", net="resnet50", pretrain=True):
        super().__init__()

        # make hyperparameter available via self.hparams
        self.save_hyperparameters()

        # fetch backbone as base network
        if net == "resnet50":
            backbone = backbones.resnet50(pretrained=pretrain)
        elif net == "resnet34":
            backbone = backbones.resnet34(pretrained=pretrain)
        elif net == "resnet18":
            backbone = backbones.resnet18(pretrained=pretrain)
        else:
            print("[MODEL]: Undefined backbone")


        # model
        self.model = classifier.ImageClassifier(backbone=backbone,
                                                num_classes=self.hparams.num_classes,
                                                bottleneck_dim=self.hparams.num_classes)

        # initialize model bottleneck
        self.model.bottleneck.apply(init_weights)
        # initialize classifier head
        self.model.head.apply(init_weights)

        # set params learnable/frozen
        for param in self.model.backbone.parameters():
            param.requires_grad = True
        for param in self.model.bottleneck.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = True

        # initialize accuracy tracker for a multiclass prediction task
        self.val_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes, top_k=1)
        self.train_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes, top_k=1)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes, top_k=1)

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed, we log the graph only once
        if self.current_epoch == 0:
            # check if shape is set, try dummy otherwise
            if self.input_shape:
                sampleImg = torch.rand((self.input_shape[0],
                                        self.input_shape[1],
                                        self.input_shape[2],
                                        self.input_shape[3]))
            else:
                sampleImg = torch.rand((64, 3, 224, 224))
            # move to model device
            sampleImg = sampleImg.to(self.device)
            # fetch logger and add graph
            tensorboard_logger = self.logger.experiment
            tensorboard_logger.add_graph(model=self, input_to_model=sampleImg)

    def on_train_start(self):
        """Called at the beginning of training after sanity check."""
        pass

    def training_step(self, train_batch, batch_idx):
        data, labels = train_batch
        # track model input shape for creating the graph histogram
        if not self.input_shape:
            self.input_shape = data.shape
        _, predictions = self(data)
        _, predict = torch.max(predictions, 1)
        # update train accuracy
        self.train_acc(predict, labels.squeeze())
        # calculate classifier loss
        classifier_loss = CrossEntropyLabelSmooth(num_classes=self.hparams.num_classes,
                                                  epsilon=self.hparams.epsilon,
                                                  use_gpu=self.device == "cuda")(predictions, labels)
        self.log("classifier_loss", classifier_loss)
        self.log("train_acc", self.train_acc)
        lr_scheduler(optimizer=self.optimizers(), iter_num=self.trainer.global_step,
                     max_iter=self.trainer.estimated_stepping_batches)
        # return train loss
        return {'loss': classifier_loss, 'train_acc': self.train_acc}

    def validation_step(self, train_batch, batch_idx):
        data, labels = train_batch
        _, predictions = self(data)
        _, predict = torch.max(predictions, 1)
        self.val_acc(predict, labels.squeeze())
        self.log("val_acc", self.val_acc)
        # return validation accuracy
        return {'val_acc': self.val_acc}

    def test_step(self, test_batch, batch_idx):
        data, labels = test_batch
        _, predictions = self(data)
        _, predict = torch.max(predictions, 1)
        self.test_acc(predict, labels.squeeze())
        self.log("test_acc", self.test_acc)
        # return test accuracy
        return {'test_acc': self.test_acc}

    def configure_optimizers(self):
        param_group = self.model.get_parameters()
        # create optimizer with parameter group
        if self.hparams.optimizer == "sgd":
            optimizer = optim.SGD(params=param_group, lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay, nesterov=True)
        elif self.hparams.optimizer == "adam":
            optimizer = optim.Adam(params=param_group, lr=self.hparams.lr)

        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=self.hparams.gamma)
        optimizer = op_copy(optimizer)
        return optimizer


class ClientDataModel(DataModelBase):
    def __init__(self, args, pretrained_model=None):
        super().__init__()
        self.model = pretrained_model
        self.class_prototypes_source = None
        self.dataset_mean_source = None
        self.episodic_prototypes = None
        self.K = None
        self.N = None
        self.save_hyperparameters(ignore=['pretrained_model'])

        # set params learnable/frozen
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        for param in self.model.bottleneck.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = False

        # initialize accuracy tracker for a multiclass prediction task
        self.val_acc = Accuracy(task="multiclass", num_classes=self.hparams.args.num_classes, top_k=1)
        self.train_acc = Accuracy(task="multiclass", num_classes=self.hparams.args.num_classes, top_k=1)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.hparams.args.num_classes, top_k=1)

    def configure_optimizers(self):
        param_group = self.model.get_parameters(target_adaptation=True)
        # create optimizer with parameter group
        if self.hparams.args.optimizer == "sgd":
            optimizer = optim.SGD(params=param_group, lr=self.hparams.args.lr, momentum=self.hparams.args.momentum,
                                  weight_decay=self.hparams.args.weight_decay, nesterov=True)
        elif self.hparams.args.optimizer == "adam":
            optimizer = optim.Adam(params=param_group, lr=self.hparams.args.lr)
        optimizer = op_copy(optimizer)
        return optimizer

    def set_class_prototypes(self, prototypes):
        self.class_prototypes_source = prototypes.to(self.device)

    def set_source_dataset_mean(self, mean):
        self.dataset_mean_source = mean.to(self.device)

    def log_tb_images(self, viz_batch, batch_idx) -> None:

        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError('TensorBoard Logger not found')

            # Log the images (Give them different names)
        for img_idx, (image, y_true, y_pred) in enumerate(zip(*viz_batch)):
            tb_logger.add_image(f"Image/{batch_idx}_{img_idx}", image, 0, dataformats='CHW')
            #tb_logger.add_image(f"GroundTruth/{batch_idx}_{img_idx}", y_true, 0, dataformats='CHW')
            #tb_logger.add_image(f"Prediction/{batch_idx}_{img_idx}", y_pred, 0, dataformats='CHW')

    def training_step(self, train_batch, batch_idx):
        data, labels = train_batch
        # track model input shape for creating the graph histogram
        if not self.input_shape:
            self.input_shape = data.shape
        _, predictions = self(data)
        _, predict = torch.max(predictions, 1)
        # update train accuracy
        self.train_acc(predict, labels.squeeze())
        # calculate classifier loss
        classifier_loss = CrossEntropyLabelSmooth(num_classes=self.hparams.args.num_classes,
                                                  epsilon=self.hparams.args.epsilon,
                                                  use_gpu=self.device == "cuda")(predictions, labels)
        # log loss and accuracy
        self.log("classifier_loss", classifier_loss)
        self.log("train_acc", self.train_acc)
        lr_scheduler(optimizer=self.optimizers(), iter_num=self.trainer.global_step,
                     max_iter=self.trainer.estimated_stepping_batches)
        # return train loss
        return {'loss': classifier_loss, 'train_acc': self.train_acc}

    def validation_step(self, train_batch, batch_idx):
        data, labels = train_batch
        _, predictions = self(data)
        _, predict = torch.max(predictions, 1)
        self.val_acc(predict, labels.squeeze())
        self.log("val_acc", self.val_acc)
        # log validation images
        if batch_idx % LOG_INPUT_IMG_EVERY_N_BATCH:  # Log every N batches
            self.log_tb_images((data, labels, predict), batch_idx)
        # return validation accuracy
        return {'val_acc': self.val_acc}

    def test_step(self, test_batch, batch_idx):
        data, labels = test_batch
        _, predictions = self(data)
        _, predict = torch.max(predictions, 1)
        self.test_acc(predict, labels.squeeze())
        self.log("test_acc", self.test_acc)
        # return test accuracy
        return {'test_acc': self.test_acc}
