"""Train models module."""
from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


import logging
from pathlib import Path
import sys
import os

from tqdm import tqdm
import torch
import torch.nn.functional as F
from losses import loss_supervised, loss_unsupervised



from latent_trainer.config import LOGGING_FORMAT
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "....")))
from sc2_datasets.torch.sc2_egset_dataset import SC2EGSetDataset
from sc2_datasets.torch.datasets.sc2_dataset import SC2Dataset
from sc2_datasets.replay_data.sc2_replay_data import SC2ReplayData
from sc2_datasets.available_replaypacks import EXAMPLE_SYNTHETIC_REPLAYPACKS

from guided_vae import Classifier, suGuidedVAE

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets
from torchvision.utils import save_image
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from sc2_datasets.torch.datasets.sc2_replaypack_dataset import SC2ReplaypackDataset


def train_supervised(epoch, model, model_c, optimizer, optimizer_c, dataloader, w_cls, device):
    model.train()
    re_loss = 0
    cls_error = 0
    correct = 0

    cls1_error = 0
    cls2_error = 0

    correct1 = 0
    correct2 = 0
    for batch_idx, (data, label) in enumerate(tqdm(dataloader)):
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, re = model(data)
        loss_list = loss_supervised(recon_batch, data, mu, logvar)
        loss = loss_list[0]
        loss_cls = F.binary_cross_entropy(re, label, reduction='sum')
        cls_error += loss_cls
        loss += loss_cls * w_cls
        loss.backward()
        re_loss += loss_list[1].item()
        optimizer.step()

        optimizer_c.zero_grad()
        z = model.reparameterize(mu, logvar).detach()
        z = z[:, 1:]
        cls1 = model_c(z)
        loss = F.binary_cross_entropy(cls1, label, reduction='sum')
        cls1_error += loss.item()
        loss *= w_cls
        loss.backward()
        optimizer_c.step()

        optimizer.zero_grad()
        mu, logvar = model.encode(data)
        z = model.reparameterize(mu, logvar)
        z = z[:, 1:]
        cls2 = model_c(z)
        label1 = torch.empty_like(label).fill_(0.5)
        loss = F.binary_cross_entropy(cls2, label1, reduction='sum')
        cls2_error += loss.item()
        loss *= w_cls
        loss.backward()
        optimizer.step()

        pred = (re + 0.5).int()
        correct += pred.eq(label.int()).sum().item()

        pred = (cls1 + 0.5).int()
        correct1 += pred.eq(label.int()).sum().item()

        pred = (cls2 + 0.5).int()
        correct2 += pred.eq(label.int()).sum().item()

    cls_error = cls_error / len(dataloader.dataset)
    cls1_error = cls1_error / len(dataloader.dataset)
    cls2_error = cls2_error / len(dataloader.dataset)
    print('====> Epoch: {} reconstruction loss: {:.4f} Cls loss: {:.4f} acc: {:.2f} cls1 loss: {:.4f} cls2 loss: {:.4f} acc1: {:.2f} acc2: {:.2f}'.format(
          epoch, re_loss / len(dataloader.dataset), cls_error, 100. * correct / len(dataloader.dataset),
          cls1_error, cls2_error, 100. * correct1 / len(dataloader.dataset),
          100. * correct2 / len(dataloader.dataset)))

# Check this
def arg_parse():
    parser = argparse.ArgumentParser(description='Guided VAE')
    parser.add_argument('--batch-size', '-b', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=128, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu id to use')
    parser.add_argument('--nz', type=int, default=10,
                        help='bottleneck size')
    parser.add_argument('--output', default='output',
                        help='output directory for results')
    parser.add_argument('--dataroot', default='data',
                        help='root directory for dataset')
    parser.add_argument('--dataset', default='MNIST', choices=['MNIST', 'CelebA'],
                        help='dataset to train')
    parser.add_argument('--cls', default='200.0', type=float,
                        help='classification error weight for supervised Guided-VAE')
    parser.add_argument('--selected_attrs', default=['Smiling'], nargs='+',
                        help='selected attrs in CelebA training')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='number of workers for dataloader')
    parser.add_argument('--test_interval', default=1, type=int,
                        help='interval for testing')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='weight decay')
    parser.add_argument('--lr_c', default=1e-4, type=float,
                        help='classifier learning rate(in supervised version)')
    parser.add_argument('--weight_decay_c', default=1e-4, type=float,
                        help='classifier weight decay(in supervised version)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)

    download_path = Path("./data/download").resolve().as_posix()
    unpack_path = Path("./data/unpack").resolve().as_posix()

    logging.info(f"{download_path=}")
    logging.info(f"{unpack_path=}")

    dataset = SC2Dataset(names_urls=EXAMPLE_SYNTHETIC_REPLAYPACKS).__getitem__("result")
    args = arg_parse()
    device = torch.device('cpu')
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    torch.manual_seed(1024)

    model = suGuidedVAE().to(device)
    model_c = model_c = Classifier().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_c = optim.Adam(model_c.parameters(), lr=args.lr, weight_decay=args.weight_decay_c)
    transform = T.Compose([
            T.CenterCrop(178),
            T.Resize(64),
            T.ToTensor(),
        ])


    # data_path = Path(args.dataroot, 'images').resolve()
    # attr_path = Path(args.dataroot, 'list_attr_celeba.txt').resolve()

    train_dataset = dataset
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)        
    for epoch in range(1, args.epochs + 1):
            train_supervised(epoch, model, model_c, optimizer, optimizer_c, train_loader, args.cls, device)


