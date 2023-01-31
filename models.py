import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchmetrics import Accuracy
import resnet as backbones
import classifier
from torchsummary import summary


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
        param_group['lr0'] = param_group['lr']
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
        #self.save_hyperparameters()

    def set_training_dataset_mean(self, mean_tensor):
        self.training_dataset_mean = mean_tensor

    def forward(self, x):
        return self.model(x)


class ServerDataModel(DataModelBase):
    def __init__(self, name, num_classes, lr, momentum, gamma, weight_decay, epsilon, pretrain=True):
        super().__init__()

        # make hyperparameter available via self.hparams
        self.save_hyperparameters()

        # fetch backbone as base network
        backbone = backbones.resnet50(pretrained=pretrain)

        # model
        self.model = classifier.ImageClassifier(backbone=backbone,
                                                num_classes=self.hparams.num_classes,
                                                bottleneck_dim=self.hparams.num_classes)

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
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)

        # print the model summary
        # summary(classifier, input_size=(3, 224, 224))
        #summary(self.model, input_size=(3, 224, 224))

    #def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed
        #if self.current_epoch == 1:
            #sampleImg = torch.rand((1, 3, 224, 224))
            #self.logger.experiment.add_graph(self, sampleImg)

    def on_train_start(self):
        """Called at the beginning of training after sanity check."""
        pass

    def training_step(self, train_batch, batch_idx):
        data, labels = train_batch
        _, predictions = self(data)
        #logits = nn.Softmax(dim=1)(predictions)
        classifier_loss = CrossEntropyLabelSmooth(num_classes=self.hparams.num_classes,
                                                  epsilon=self.hparams.epsilon,
                                                  use_gpu=self.device == "cuda")(predictions, labels)
        self.log("classifier_loss", classifier_loss)
        lr_scheduler(optimizer=self.optimizers(), iter_num=self.trainer.global_step,
                     max_iter=self.trainer.estimated_stepping_batches)
        # return train loss
        return {'loss': classifier_loss}

    def validation_step(self, train_batch, batch_idx):
        data, labels = train_batch
        _, predictions = self(data)
        #logits = nn.Softmax(dim=1)(predictions)
        _, predict = torch.max(predictions, 1)
        self.val_acc(predict, labels.squeeze())
        self.log("val_acc", self.val_acc)
        # return validation accuracy
        return {'val_acc': self.val_acc}

    def test_step(self, test_batch, batch_idx):
        data, labels = test_batch
        _, predictions = self(data)
        #logits = nn.Softmax(dim=1)(predictions)
        _, predict = torch.max(predictions, 1)
        self.test_acc(predict, labels.squeeze())
        self.log("test_acc", self.test_acc)
        # return validation accuracy
        return {'val_acc': self.test_acc}

    def configure_optimizers(self):
        param_group = self.model.get_parameters()
        # create optimizer with parameter group
        optimizer = optim.SGD(params=param_group, lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay, nesterov=True)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=self.hparams.gamma)
        optimizer = op_copy(optimizer)
        return optimizer


class ClientDataModel(DataModelBase):
    def __init__(self, pretrained_model=None):
        super().__init__()
        self.model = pretrained_model
        self.class_prototypes_source = None
        self.episodic_prototypes = None
        self.K = None
        self.N = None
        self.save_hyperparameters(ignore=['pretrained_model'])

        # set params learnable/frozen
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        for param in self.model.bottleneck.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

    def configure_optimizers(self):
        params = [{"params": self.attention.parameters(), "lr_mult": 1.}]
        # create optimizer with parameter group
        optimizer = optim.SGD(params=params, lr=self.hparams.args.lr, momentum=self.hparams.args.momentum, weight_decay=self.hparams.args.weight_decay, nesterov=True)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=self.hparams.gamma)
        optimizer = op_copy(optimizer)
        return optimizer

    def set_class_prototypes(self, prototypes):
        self.class_prototypes_source = prototypes.to(self.device)

    def training_step(self, train_batch, batch_idx):
        mean_support_features = []
        for category in train_batch:
            data = train_batch[category][0]
            label = train_batch[category][1]
            f, _ = self(data)
            mean_support_features.append(torch.mean(f, dim=0))
        mean_features = torch.stack(mean_support_features, 0)
        refined_f, _ = self.attention(mean_features, mean_features, mean_features)
        distance = torch.nn.PairwiseDistance(p=2)(refined_f, self.class_prototypes_source).mean(0)
        print("Done")
        #logits = nn.Softmax(dim=1)(predictions)
        #classifier_loss = CrossEntropyLabelSmooth(num_classes=self.hparams.num_classes,
                                                  #epsilon=self.hparams.epsilon,
                                                  #use_gpu=self.device == "cuda")(predictions, labels)
        #self.log("classifier_loss", classifier_loss)
        #lr_scheduler(optimizer=self.optimizers(), iter_num=self.trainer.global_step,
                     #max_iter=self.trainer.estimated_stepping_batches)
        # return train loss
        return {'loss': distance}
