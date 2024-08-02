import torch, os, cv2, wandb, signal, sys, csv, math, re
from tqdm import tqdm 
from torch.utils.data import DataLoader

from datasets import UnsupervisedDataset, BalancedBatchSampler, contrastive_collate, DistributedBatchSampler
from models import ResNet, BasicBlock
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import lightning as L
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

SAVE_DIR= "~/scratch/gapjnc/models/"

from torch.optim.optimizer import Optimizer, required
import torch.optim as optim
import torch
import argparse

from utils import lprint

DEBUG = False

# almost copy paste from https://github.com/noahgolmant/pytorch-lars/blob/master/lars.py
class LARS(Optimizer):
    r"""Implements LARS (Layer-wise Adaptive Rate Scaling).
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        eta (float, optional): LARS coefficient as used in the paper (default: 1e-3)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        epsilon (float, optional): epsilon to prevent zero division (default: 0)
    Example:
        >>> optimizer = torch.optim.LARS(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=required, momentum=0, eta=1e-3, dampening=0,
                 weight_decay=0, nesterov=False, epsilon=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, eta=eta, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, epsilon=epsilon)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(LARS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            dampening = group['dampening']
            nesterov = group['nesterov']
            epsilon = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                w_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)
                if w_norm * g_norm > 0:
                    local_lr = eta * w_norm / (g_norm +
                        weight_decay * w_norm + epsilon)
                else:
                    local_lr = 1
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=-local_lr * group['lr'])

        return loss


class ContrastiveModel(nn.Module):
    def __init__(self, base_model):
        super(ContrastiveModel, self).__init__()
        self.model = base_model
        self.final_non_linear = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256)
        )
    
    def forward(self, data):
        return self.final_non_linear((self.model(data)))

def contrastive_train(model, dataloader, T_max, epochs=1000, temperature=0.5,batch_size=500):
    lowest_loss = None

    #optimizer
    # base = optim.SGD(model.parameters(), lr=0.075*math.sqrt(batch_size), weight_decay=1e-6)
    # optimizer = LARS(optimizer=base, eps=1e-8, trust_coef=0.001)
    optimizer = LARS(model.parameters(), lr=0.075*math.sqrt(batch_size), weight_decay=1e-6, epsilon=1e-8)


    decay_lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    for i in tqdm(range(epochs)):
        for imgs, labels in dataloader:

            #get embeddings:
            embs = model(imgs)

            #compute similarity:
            sim_matrix = (embs @ embs.T)
            norms = torch.sqrt(torch.sum(embs**2)).unsqueeze(-1)
            norm_matrix = norms @ norms.T
            sim_matrix = sim_matrix/norm_matrix
            sim_matrix = torch.exp(sim_matrix/temperature)

            #zero the diagonal
            sim_matrix = sim_matrix * (torch.eye(sim_matrix.shape[0], sim_matrix.shape[1]) == 0)

            #compute loss per label:
            labels_sim = labels.unsqueeze(0).squeeze(-1).repeat(labels.shape[1], 1)
            labels_sim = labels_sim.T == labels_sim
            labels_sim = labels_sim * torch.eye(sim_matrix.shape[0], sim_matrix.shape[1]) == 0

            denoms = torch.sum(sim_matrix, axis=0)
            nums = sim_matrix * labels_sim
            loss = -torch.log(nums / denoms).sum(axis=0)
            loss = loss.mean()

            loss.backward()
            optimizer.step()
            if i >= 10: decay_lr_schedule.step()

class LightningContrastive(L.LightningModule):
    def __init__(self, model, temperature, T_max, dataset=None, n_classes=5, n_samples=100):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.T_max = T_max
        self.batch_size=500

        self.train_dataset = dataset
        self.n_classes = n_classes
        self.n_samples = n_samples

        self.automatic_optimization = False
    
    def training_step(self, batch, batch_idx):
            

        opt = self.optimizers()
        opt.zero_grad()

        sch = self.lr_schedulers()

        #get embeddings:
        imgs, labels = batch
        embs = self.model(imgs)

        all_outputs = self.all_gather(embs, sync_grads=True)
        all_labels = self.all_gather(labels)
        embs = all_outputs.view(-1, 256)
        labels = all_labels.view(-1)

        #compute similarity:

        sim_matrix = (embs @ embs.T)
        norms = torch.sqrt(torch.sum(embs**2, dim=-1)).unsqueeze(-1)
        norm_matrix = norms @ norms.T
        sim_matrix = sim_matrix/norm_matrix
        sim_matrix = torch.exp(sim_matrix/self.temperature)

        #zero the diagonal
        sim_matrix = sim_matrix * (torch.eye(sim_matrix.shape[0], sim_matrix.shape[1]) == 0).to(sim_matrix.device)

        #compute loss per label:
        labels_sim = labels.unsqueeze(0).repeat(labels.shape[0], 1)
        labels_sim = labels_sim.T == labels_sim
        labels_sim *= (torch.eye(sim_matrix.shape[0], sim_matrix.shape[1]) == 0).to(sim_matrix.device)

        denoms = torch.sum(sim_matrix, axis=0)
        nums = sim_matrix * labels_sim
        loss = -torch.log(nums / denoms).sum(axis=0)
        loss = loss.mean()

        loss.backward()
        opt.step()

        if self.trainer.current_epoch > 10: 
            sch.step()
            
        self.log("loss", loss, batch_size=embs.size(0), sync_dist=True, prog_bar=True)

    def configure_optimizers(self):
        # base = optim.SGD(self.model.parameters(), lr=0.075*math.sqrt(self.batch_size), weight_decay=1e-6)
        # optimizer = LARS(optimizer=base, eps=1e-8, trust_coef=0.001)

        optimizer = LARS(self.model.parameters(), lr=0.075*math.sqrt(self.batch_size), weight_decay=1e-6, epsilon=1e-8)

        decay_lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": decay_lr_schedule
            }
        }
    
    def train_dataloader(self):
        # balanced_batch_sampler = BalancedBatchSampler(train_dataset, n_classes, n_samples))
        limited_labels = list(set([int(re.findall("neuron_\d+", label)[0][len("neuron_"):]) for label in self.train_dataset.labels]))
        dist_balanced_batch_sampler = DistributedBatchSampler(limited_labels, shuffle=True, n_samples=self.n_samples, n_classes=self.n_classes, actual_dataset=self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, batch_sampler=dist_balanced_batch_sampler, collate_fn=contrastive_collate, num_workers=12)
        return train_dataloader
    
def setup_contrastive_learning(dataset_dir, epochs=1000, temperature=0.5, n_classes=100, n_samples=5, lightning=True):

    batch_size = n_classes * n_samples

    model = ContrastiveModel(ResNet(BasicBlock, [1, 4, 6, 3, 3], norm_layer = nn.BatchNorm2d))
    train_dataset = UnsupervisedDataset(dataset_dir)

    if lightning:
        model = LightningContrastive(model, temperature, len(train_dataset)//batch_size + 1, dataset=train_dataset, n_classes=n_classes, n_samples=n_samples)

        checkpoint_callback = ModelCheckpoint(dirpath=SAVE_DIR,
        filename='constrative_{epoch}',
        save_top_k=-1,
        every_n_epochs=20,
        save_on_train_epoch_end=True
        )
        # logger = WandbLogger(log_model="all", project="celegans", entity="mishaalkandapath")
        trainer = L.Trainer(callbacks=[checkpoint_callback], max_epochs=epochs, log_every_n_steps=100, num_sanity_val_steps=0, default_root_dir=SAVE_DIR, use_distributed_sampler=False)
        trainer.fit(model)

    else:
        limited_labels = list(set([int(re.findall("neuron_\d+", label)[0][len("neuron_"):]) for label in train_dataset.labels]))
        dist_balanced_batch_sampler = BalancedBatchSampler(limited_labels, train_dataset, n_classes, n_samples)
        train_dataloader = DataLoader(train_dataset, batch_sampler=dist_balanced_batch_sampler, collate_fn=contrastive_collate, num_workers=12)
        contrastive_train(model, train_dataloader, len(train_dataset)//batch_size + 1, epochs, temperature, batch_size)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.set_float32_matmul_precision('medium')
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="/home/mishaalk/scratch/gapjunc/train_datasets/unsupervised")
    parser.add_argument("--epochs", default=1000)
    parser.add_argument("--temperature", default=0.5)
    parser.add_argument("--n_classes", default=100, type=int)
    parser.add_argument("--n_samples", default=5, type=int)
    parser.add_argument("--lightning", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    os.environ["N_CLASSES_CUSTOM"] = str(args.n_classes)
    os.environ["N_SAMPLES_CUSTOM"] = str(args.n_samples)

    setup_contrastive_learning(args.dataset_dir, args.epochs, args.temperature, args.n_classes, args.n_samples, args.lightning)
