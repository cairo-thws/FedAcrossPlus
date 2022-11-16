import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
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
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


class ModelBase(pl.LightningModule):
    def __init__(self, name, num_classes, lr, momentum, gamma, weight_decay, epsilon):
        super().__init__()
        # make hyperparameter available via self.hparams
        self.save_hyperparameters()


class ServerModel(ModelBase):
    def __init__(self, name, num_classes, lr, momentum, gamma, weight_decay, epsilon):
        super().__init__(name, num_classes, lr, momentum, gamma, weight_decay, epsilon)

        ## set base network
        backbone = backbones.resnet50(pretrained=True)

        # model
        self.model = classifier.ImageClassifier(backbone=backbone, num_classes=self.hparams.num_classes)

        # initialize classifier head
        self.model.head.apply(init_weights)
        # set params learnable/frozen
        for param in self.model.backbone.parameters():
            param.requires_grad = True
        for param in self.model.bottleneck.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = True

        # print the model summary
        # summary(classifier, input_size=(3, 224, 224))
        summary(self.model, input_size=(3, 224, 224))

        # metric
        self.acc = Accuracy()

        # criterion for server pretraining
        self.criterion = CrossEntropyLabelSmooth(num_classes=self.hparams.num_classes,
                                                 epsilon=self.hparams.epsilon,
                                                 use_gpu=False)

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        """Called at the beginning of training after sanity check."""
        pass

    def training_step(self, train_batch, batch_idx):
        data, labels = train_batch
        _, predictions = self(data)
        classifier_loss = self.criterion(predictions, labels)
        self.log("classifier_loss", classifier_loss)
        # return train loss
        return {'loss': classifier_loss}

    def validation_step(self, train_batch, batch_idx):
        data, labels = train_batch
        _, predictions = self(data)
        logits = nn.Softmax(dim=1)(predictions)
        _, predict = torch.max(logits, 1)
        self.acc(predict, labels.squeeze())
        self.log("val_acc", self.acc)
        return {'val_acc': self.acc}

    def test_step(self, test_batch, batch_idx):
        print("Check if needed: Test step")
        pass

    def configure_optimizers(self):
        param_group = self.model.get_parameters()
        # create optimizer with parameter group
        optimizer = optim.SGD(params=param_group, lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay, nesterov=True)
        #optimizer = op_copy(optimizer)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=self.hparams.gamma)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class ClientModel(ModelBase):
    def __init__(self):
        super().__init__()