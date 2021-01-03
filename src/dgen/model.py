from collections import OrderedDict
from argparse import ArgumentParser, Namespace
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
import numpy as np



def total_variation(img: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(img):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")
    img_shape = img.shape
    if len(img_shape) == 3 or len(img_shape) == 4:
        pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
        pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]
        reduce_axes = (-3, -2, -1)
    else:
        raise ValueError("Expected input tensor to be of ndim 3 or 4, but got " + str(len(img_shape)))
    tv_loss = pixel_dif1.abs().sum(dim=reduce_axes) + pixel_dif2.abs().sum(dim=reduce_axes)
    return tv_loss
    #n = np.prod(img_shape[1:])
    #return torch.mean(tv_loss/n)
    #eps = 1e-5
    #tv_loss = (tv_loss - tv_loss.min())/(tv_loss.max() - tv_loss.min() + eps)
    #return tv_loss.mean()

class Conv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, tv_losses=[]):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.tv_losses = tv_losses

    def forward(self, x):
        x = super(Conv2d, self).forward(x)
        if self.training:
            self.tv_losses.append(total_variation(x))
        return x


class TVModel(torch.nn.Module):
    def __init__(self, model, num_tv_layers=None):
        super(TVModel, self).__init__()
        self.model = model
        self.num_tv_layers = num_tv_layers
        self.tv_losses = []
        self.tv_layer_cnt = 0
        self._add_tv_to_conv(self.model, self.tv_losses)

    def _add_tv_to_conv(self, model, losses):
        for child_name, child in model.named_children():
            if isinstance(child, torch.nn.Conv2d):
                setattr(model, child_name, Conv2d(in_channels=child.in_channels,
                                                  out_channels=child.out_channels,
                                                  kernel_size=child.kernel_size,
                                                  stride=child.stride,
                                                  padding=child.padding,
                                                  bias=child.bias,
                                                  tv_losses=losses))
                self.tv_layer_cnt+=1

            else:
                if self.num_tv_layers:
                    if self.tv_layer_cnt == self.num_tv_layers:
                        break
                self._add_tv_to_conv(child, losses)

    def _reset_losses(self):
        self.tv_losses.clear()

    def forward(self, x):
        self._reset_losses()
        x = self.model(x)
        return x

class BaseModel(LightningModule):

    # pull out resnet names from torchvision models
    MODEL_NAMES = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
    )
    def __init__(
        self,
        arch: str,
        pretrained: bool,
        lr: float,
        momentum: float,
        weight_decay: int,
        no_jsd: bool,
        tv: float,
        tv_min:float,
        num_classes:int,
        batch_size:int,
        num_tv_layers:int = 0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.arch = arch
        self.pretrained = pretrained
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.no_jsd = no_jsd
        self.tv = tv
        self.tv_min = tv_min
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.model = models.__dict__[self.arch](pretrained=self.pretrained, num_classes=self.num_classes)
        self.num_tv_layers = num_tv_layers
        if self.tv > 0:
            self.model = TVModel(self.model, self.num_tv_layers)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.no_jsd:
            return self._training_step_standard(batch, batch_idx)
        else:
            return self._training_step_jsd(batch, batch_idx)

    def _compute_tv_loss(self):
        if self.tv > 0:
            tv_losses = self.model.tv_losses
            if self.num_tv_layers:
                assert(len(tv_losses) == self.num_tv_layers)
            tv_loss = sum(tv_losses).mean(axis=0)/1000
            tv_loss = tv_loss*self.tv
            #weights = np.geomspace(self.tv_min, 1, num=len(tv_losses))[::-1]
            #tv_loss = sum([w*l for w,l in zip(weights,tv_losses)])*self.tv
        else:
            tv_loss = 0.
        return tv_loss

    def _get_train_metrics(self, logits, targets, train_loss, tv_loss, jsd_loss=0.):
        acc1, acc5 = self.__accuracy(logits, targets, topk=(1, 5))
        tqdm_dict = {'train_loss': train_loss, 'tv_loss': tv_loss, 'jsd_loss':jsd_loss}
        output = OrderedDict({
            'loss': train_loss,
            'acc1': acc1,
            'acc5': acc5,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def _training_step_standard(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)

        train_loss = F.cross_entropy(logits, targets)
        tv_loss = self._compute_tv_loss()
        train_loss += tv_loss

        output = self._get_train_metrics(logits, targets, train_loss, tv_loss)
        return output

    def _training_step_jsd(self, batch, batch_idx):
        images, targets = batch
        images_all = torch.cat(images)
        logits_all = self(images_all)

        logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all,
                                                             images[0].size(0))

        train_loss = F.cross_entropy(logits_clean, targets)
        p_clean, p_aug1, p_aug2 = F.softmax(logits_clean, dim=1),\
                                  F.softmax(logits_aug1, dim=1),\
                                  F.softmax(logits_aug2, dim=1)
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
        log_target = False
        jsd_loss = 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean', log_target=log_target) +
                          F.kl_div(p_mixture, p_aug1, reduction='batchmean', log_target=log_target) +
                          F.kl_div(p_mixture, p_aug2, reduction='batchmean', log_target=log_target)) / 3.

        train_loss += jsd_loss

        tv_loss = self._compute_tv_loss()
        train_loss += tv_loss
        output = self._get_train_metrics(logits_clean, targets, train_loss, tv_loss, jsd_loss)
        return output



    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_val = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc1': acc1,
            'val_acc5': acc5,
        })
        return output

    def validation_epoch_end(self, outputs):
        tqdm_dict = {}
        for metric_name in ["val_loss", "val_acc1", "val_acc5"]:
            tqdm_dict[metric_name] = torch.stack([output[metric_name] for output in outputs]).mean()

        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': tqdm_dict["val_loss"]}
        return result

    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        lr = self.lr*self.batch_size/256.
        optimizer = optim.SGD(
            self.parameters(),
            lr=lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: 0.1 ** (epoch // 30)
        )
        return [optimizer], [scheduler]

    def test_step(self, batch, batch_idx, loader_idx):
        images, targets = batch
        logits = self(images)
        pred = logits.data.max(1)[1]
        #total_correct = pred.eq(targets.data).sum().item()
        out_dict = { 'test_pred': pred,
                     'test_target': targets
                   }
        return out_dict


    def test_epoch_end(self, outputs):
        results = []
        loader_names = [dl.loader_name for dl in self.test_dataloader()]
        for output, loader_name in zip(outputs, loader_names):
            pred = torch.cat([o['test_pred'].cpu() for o in output])
            target = torch.cat([o['test_target'].cpu() for o in output])
            total_correct = pred.eq(target.data).sum().item()
            test_acc = float(total_correct)/target.size(0)
            metric_name = loader_name + "_" + 'test_acc'
            log_dict = { metric_name: test_acc}
            result = {'log': log_dict}
            results.append(result)
        return results


    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                            choices=BaseModel.MODEL_NAMES,
                            help=('model architecture: ' + ' | '.join(BaseModel.MODEL_NAMES)
                                  + ' (default: resnet18)'))
        parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('-b', '--batch-size', default=256, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
        parser.add_argument('--num-classes', default=100, type=int, metavar='NUM_CLASSES',
                            help='number of classes in the dataset')
        parser.add_argument('--tv', '--tv-weight', default=0.0, type=float,
                            metavar='TV', help='total variation regularization weight', dest='tv')
        parser.add_argument('--tv-min', default=0.001, type=float,
                            metavar='TVMIN', help='min weight for tv weight scale', dest='tv_min')

        parser.add_argument('--num-tv-layers', default=None, type=int,
                            metavar='NUMTVLAYERS', help='the number of tv layers')

        parser.add_argument('-nj','--no-jsd', dest='no_jsd', action='store_true',
                            help='turn off jsd loss')
        return parser
