import sys

sys.argv = ['']
del sys

import numpy as np
import os
import math
from collections import defaultdict
import argparse
import argparse
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data as Data
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, TensorDataset, random_split
import copy
import random
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# from VIBImodels import ResNet, resnet18, resnet34, Unet

# from debug import debug
import torch.nn as nn
import torch.optim

import torch.nn.functional as F
#import pyhessian
#from pyhessian import hessian

class Normal(object):
    def __init__(self, mu, sigma, log_sigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
        self.logsigma = log_sigma
        dim = mu.get_shape()
        if v is None:
            v = torch.FloatTensor(*dim)
        if r is None:
            r = torch.FloatTensor(*dim)
        self.v = v
        self.r = r


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.tanh(self.linear2(x))


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.tanh(self.linear2(x))


class VAE(torch.nn.Module):
    latent_dim = 8

    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.Linear(100, 8)
        self._enc_log_sigma = torch.nn.Linear(100, 8)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc).to(args.device)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma).to(args.device)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().to(args.device)

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z)


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

class LinearModel(nn.Module):
    # 定义神经网络
    def __init__(self, n_feature=192, h_dim=3 * 32, n_output=10):
        # 初始化数组，参数分别是初始化信息，特征数，隐藏单元数，输出单元数
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(n_feature, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, n_output)  # output

    # 设置隐藏层到输出层的函数

    def forward(self, x):
        # 定义向前传播函数
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class VIBI(nn.Module):
    def __init__(self, explainer, approximator, forgetter, k=4, num_samples=4, temp=1):
        super().__init__()

        self.explainer = explainer
        self.approximator = approximator
        self.forgetter = forgetter
        # self.fc3 = nn.Linear(49, 400)
        # self.fc4 = nn.Linear(400, 784)
        self.k = k
        self.temp = temp
        self.num_samples = num_samples

        self.warmup = False

    def explain(self, x, mode='topk', num_samples=None):
        """Returns the relevance scores
        """
        double_logits_z = self.explainer(x)  # (B, C, h, w)
        if mode == 'distribution':  # return the distribution over explanation
            B, double_dimZ = double_logits_z.shape
            dimZ = int(double_dimZ / 2)
            mu = double_logits_z[:, :dimZ].cuda()
            logvar = torch.log(torch.nn.functional.softplus(double_logits_z[:, dimZ:]).pow(2)).cuda()
            logits_z = self.reparametrize(mu, logvar)
            return logits_z, mu, logvar
        elif mode == 'test':  # return top k pixels from input
            B, double_dimZ = double_logits_z.shape
            dimZ = int(double_dimZ / 2)
            mu = double_logits_z[:, :dimZ].cuda()
            logvar = torch.log(torch.nn.functional.softplus(double_logits_z[:, dimZ:]).pow(2)).cuda()
            logits_z = self.reparametrize(mu, logvar)
            return logits_z

    def forward(self, x, mode='topk'):
        B = x.size(0)
        #         print("B, C, H, W", B, C, H, W)
        if mode == 'distribution':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            return logits_z, logits_y, mu, logvar
        elif mode == 'cifar_distribution':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            # B, dimZ = logits_z.shape
            # logits_z = logits_z.reshape((B,8,8,8))
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            return logits_z, logits_y, mu, logvar
        elif mode == 'forgetting':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            #print("logits_z, mu, logvar", logits_z, mu, logvar)
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            x_hat = self.forget(logits_z)
            return logits_z, logits_y, x_hat, mu, logvar
        elif mode == 'cifar_forgetting':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            #print("logits_z, mu, logvar", logits_z, mu, logvar)
            # B, dimZ = logits_z.shape
            # logits_z = logits_z.reshape((B,8,8,8))
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            x_hat = self.cifar_forget(logits_z)
            return logits_z, logits_y, x_hat, mu, logvar
        elif mode == 'cifar_test':
            logits_z = self.explain(x, mode='test')  # (B, C, H, W)
            # B, dimZ = logits_z.shape
            # logits_z = logits_z.reshape((B,8,8,8))
            logits_y = self.approximator(logits_z)
            return logits_y
        elif mode == 'test':
            logits_z = self.explain(x, mode=mode)  # (B, C, H, W)
            logits_y = self.approximator(logits_z)
            return logits_y

    def forget(self, logits_z):
        output_x = self.forgetter(logits_z)
        return torch.sigmoid(output_x)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)



def init_vibi(dataset):
    k = args.k
    beta = args.beta
    num_samples = args.num_samples
    xpl_channels = args.xpl_channels
    explainer_type = args.explainer_type

    if dataset == 'MNIST':
        approximator = LinearModel(n_feature=49)
        forgetter = LinearModel(n_feature=49, n_output=28 * 28)
        explainer = LinearModel(n_feature=28 * 28, n_output=49 * 2)  # resnet18(1, 49*2) #
        lr = 0.001

    elif dataset == 'CIFAR10':
        approximator = LinearModel(n_feature=8*8*8, n_output=10) #resnet18(8,  10)
        explainer = resnet18(3,  8*8*8*2)  # resnet18(1, 49*2)
        forgetter = LinearModel(n_feature=8*8*8, n_output=3 * 32 * 32)
        lr = 0.005

    elif dataset == 'CIFAR100':
        approximator = LinearModel(n_feature=8*8*8, n_output=100)
        explainer = resnet18(3,  8*8*8*2)  # resnet18(1, 49*2)
        forgetter = LinearModel(n_feature=8*8*8, n_output=3 * 32 * 32)
        lr = 3e-4

    vibi = VIBI(explainer, approximator, forgetter, k=k, num_samples=args.num_samples)
    vibi.to(args.device)
    return vibi, lr

class PoisonedDataset(Dataset):

    def __init__(self, dataset, base_label, trigger_label, poison_samples, mode="train", device=torch.device("cuda"),
                 dataname="MNIST"):
        # self.class_num = len(dataset.classes)
        # self.classes = dataset.classes
        # self.class_to_idx = dataset.class_to_idx
        self.device = device
        self.dataname = dataname
        self.ori_dataset = dataset
        self.data, self.targets = self.add_trigger(self.reshape(dataset, dataname), dataset.targets, base_label,
                                                   trigger_label, poison_samples, mode)
        self.channels, self.width, self.height = self.__shape_info__()
        # self.data_test, self.targets_test = self.add_trigger_test(self.reshape(dataset.data, dataname), dataset.targets, base_label, trigger_label, portion, mode)

    def __getitem__(self, item):
        img = self.data[item]
        label_idx = self.targets[item]

        label = np.zeros(10)
        label[label_idx] = 1  # 把num型的label变成10维列表。
        label = torch.Tensor(label)

        img = img.to(self.device)
        label = label.to(self.device)

        return img, label

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
        return self.data.shape[1:]

    def reshape(self, dataset, dataname="MNIST"):
        if dataname == "MNIST":
            temp_img = dataset.data.reshape(len(dataset.data), 1, 28, 28).float()
        elif dataname == "CIFAR10":
            temp_img = torch.empty(0, 3, 32, 32).float().cuda()
            temp_label = torch.empty(0).long().cuda()
            for id in range(len(dataset)):
                image, label = dataset[id]
                image, label = image.cuda().reshape(1, 3, 32, 32), torch.tensor([label]).long().cuda()
                # print(label)
                # label = torch.tensor([label])
                temp_img = torch.cat([temp_img, image], dim=0)
                temp_label = torch.cat([temp_label, label], dim=0)
                # print(id)

        # x = torch.Tensor(image.cuda())
        # x = torch.tensor(image)
        # # print(x)

        return np.array(temp_img.to("cpu"))

    def norm(self, data):
        offset = np.mean(data, 0)
        scale = np.std(data, 0).clip(min=1)
        return (data - offset) / scale

    def add_trigger(self, data, targets, base_label, trigger_label, poison_samples, mode):
        print("## generate——test " + mode + " Bad Imgs")
        new_data = copy.deepcopy(data)
        new_targets = []
        new_data_re = []

        # total_poison_num = int(len(new_data) * portion/10)
        _, width, height = data.shape[1:]
        for i in range(len(data)):
            if targets[i] == base_label:
                new_targets.append(trigger_label)
                if trigger_label != base_label:
                    new_data[i, :, width - 3, height - 3] = 250
                    new_data[i, :, width - 3, height -4] = 250
                    new_data[i, :, width - 4, height - 3] = 250
                    new_data[i, :, width - 4, height - 4] = 250
                    # new_data[i, :, width - 23, height - 21] = 254
                    # new_data[i, :, width - 23, height - 22] = 254
                # new_data[i, :, width - 22, height - 21] = 254
                # new_data[i, :, width - 24, height - 21] = 254
                new_data[i] = new_data[i]/255
                new_data_re.append(new_data[i])
                #print("new_data[i]",new_data[i])
                poison_samples = poison_samples - 1
                if poison_samples <= 0:
                    break
                # x=torch.tensor(new_data[i])
                # x_cpu = x.cpu().data
                # x_cpu = x_cpu.clamp(0, 1)
                # x_cpu = x_cpu.view(1, 1, 28, 28)
                # grid = torchvision.utils.make_grid(x_cpu, nrow=1, cmap="gray")
                # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
                # plt.show()

        return torch.Tensor(new_data_re), torch.Tensor(new_targets).long()


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--erased_size', type=int, default=100, help="erased samples size")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=50, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--sampling', type=float, default=1.0, help="random sampling (default: 1.0)")
    parser.add_argument('--epsilon', type=float, default=1.0, help="DP epsilon (default: 1.0)")
    parser.add_argument('--poison_portion', type=float, default=0.0, help="poisoning data portion rate (default: 0.0)")
    parser.add_argument('--erased_portion', type=float, default=0.0, help="erased rate (default: 0.0)")


    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    # parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='need auxiliary data or not for non iid')
    parser.add_argument('--add_noise', action='store_true', default=False,  help='need add noise or not for non iid')


    parser.add_argument('--dataset', choices=['MNIST', 'CIFAR10'], default='CIFAR10')
    # parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs for VIBI.')
    parser.add_argument('--explainer_type', choices=['Unet', 'ResNet_2x', 'ResNet_4x', 'ResNet_8x'],
                        default='ResNet_4x')
    parser.add_argument('--xpl_channels', type=int, choices=[1, 3], default=3)
    parser.add_argument('--k', type=int, default=12, help='Number of chunks.')
    parser.add_argument('--beta', type=float, default=0.001, help='beta in objective J = I(y,t) - beta * I(x,t).')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples used for estimating expectation over p(t|x).')
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--save_best', action='store_true',
                        help='Save only the best models (measured in valid accuracy).')
    parser.add_argument('--save_images_every_epoch', action='store_true', help='Save explanation images every epoch.')
    parser.add_argument('--jump_start', action='store_true', default=False)

    args = parser.parse_args()

    return args

def conv_block(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=None):
        super().__init__()
        stride = stride or (1 if in_channels >= out_channels else 2)
        self.block = conv_block(in_channels, out_channels, stride)
        if stride == 1 and in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return F.relu(self.block(x) + self.skip(x))


class ResNet(nn.Module):
    def __init__(self, in_channels, block_features, num_classes=10, headless=False):
        super().__init__()
        block_features = [block_features[0]] + block_features + ([num_classes] if headless else [])
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, block_features[0], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(block_features[0]),
        )
        self.res_blocks = nn.ModuleList([
            ResBlock(block_features[i], block_features[i + 1])
            for i in range(len(block_features) - 1)
        ])
        self.linear_head = None if headless else nn.Linear(block_features[-1], num_classes)

    def forward(self, x):
        x = self.expand(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        if self.linear_head is not None:
            x = F.avg_pool2d(x, x.shape[-1])  # completely reduce spatial dimension
            x = self.linear_head(x.reshape(x.shape[0], -1))
        return x


def resnet18(in_channels, num_classes):
    block_features = [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2
    return ResNet(in_channels, block_features, num_classes)


def resnet34(in_channels, num_classes):
    block_features = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3
    return ResNet(in_channels, block_features, num_classes)


class Unet(nn.Module):
    def __init__(self, in_channels, down_features, num_classes, pooling=False):
        super().__init__()
        self.expand = conv_block(in_channels, down_features[0])

        self.pooling = pooling

        down_stride = 1 if pooling else 2
        self.downs = nn.ModuleList([
            conv_block(ins, outs, stride=down_stride) for ins, outs in zip(down_features, down_features[1:])])

        up_features = down_features[::-1]
        self.ups = nn.ModuleList([
            conv_block(ins + outs, outs) for ins, outs in zip(up_features, up_features[1:])])

        self.final_conv = nn.Conv2d(down_features[0], num_classes, kernel_size=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.expand(x)

        x_skips = []

        for down in self.downs:
            x_skips.append(x)
            x = down(x)
            if self.pooling:
                x = F.max_pool2d(x, 2)

        for up, x_skip in zip(self.ups, reversed(x_skips)):
            x = torch.cat([self.upsample(x), x_skip], dim=1)
            x = up(x)

        x = self.final_conv(x)

        return x



@torch.no_grad()
def test_accuracy(model, datatest, args, name='test'):
    num_total = 0
    num_correct = 0
    model.eval()
    data_loader = DataLoader(datatest, batch_size=args.bs)
    for x, y in data_loader:
        x, y = x.to(args.device), y.to(args.device)
        x = x.view(x.size(0), -1)
        out = model(x, mode='test')
        if y.ndim == 2:
            y = y.argmax(dim=1)
        num_correct += (out.argmax(dim=1) == y).sum().item()
        num_total += len(x)
    acc = num_correct / num_total
    print(f'{name} accuracy: {acc:.3f}')
    return acc

def num_params(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def sample_gumbel(size):
    return -torch.log(-torch.log(torch.rand(size)))


def gumbel_reparametrize(log_p, temp, num_samples):
    assert log_p.ndim == 2
    B, C = log_p.shape  # (B, C)
    shape = (B, num_samples, C)
    g = sample_gumbel(shape).to(log_p.device)  # (B, N, C)
    return F.softmax((log_p.unsqueeze(1) + g) / temp, dim=-1)  # (B, N, C)


# this is only a, at most k-hot relaxation
def k_hot_relaxed(log_p, k, temp, num_samples):
    assert log_p.ndim == 2
    B, C = log_p.shape  # (B, C)
    shape = (k, B, C)
    k_log_p = log_p.unsqueeze(0).expand(shape).reshape((k * B, C))  # (k* B, C)
    k_hot = gumbel_reparametrize(k_log_p, temp, num_samples)  # (k* B, N, C)
    k_hot = k_hot.reshape((k, B, num_samples, C))  # (k, B, N, C)
    k_hot, _ = k_hot.max(dim=0)  # (B, N, C)
    return k_hot  # (B, N, C)


# needed for when labels are not one-hot
def soft_cross_entropy_loss(logits, y):
    return -(y * F.log_softmax(logits, dim=-1)).sum(dim=1).mean()

class LinearCIFAR(nn.Module):
    # 定义神经网络
    def __init__(self, n_feature=192, h_dim=3*32, h_dim2=32, n_output=10):
        # 初始化数组，参数分别是初始化信息，特征数，隐藏单元数，输出单元数
        super(LinearCIFAR, self).__init__()
        self.fc0 = nn.Linear(n_feature, n_feature)
        self.fc1 = nn.Linear(n_feature, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, h_dim2)  # mu
        self.fc3 = nn.Linear(h_dim2, n_output)  # log_var

        self.W = [self.fc0.weight, self.fc1.weight, self.fc2.weight, self.fc3.weight]

    # 设置隐藏层到输出层的函数

    def forward(self, x):
        # 定义向前传播函数
        a0 = self.fc0(x)
        h0 = F.relu(a0)
        a1 = self.fc1(h0)
        h1 = F.relu(a1)
        a2 = self.fc2(h1)
        h2 = F.relu(a2)
        z = self.fc3(h2)

        # hid0 = F.relu(self.fc0(x))
        # hid1 = F.relu(self.fc1(hid0))
        # hid2 = F.relu(self.fc2(hid1))
        # # 给x加权成为a，用激励函数将a变成特征b
        # hid3 = F.softmax(self.fc3(hid2))
        cache = (a0, h0, a1, h1, a2, h2)
        z.retain_grad()
        for c in cache:
            c.retain_grad()

        return z, cache

        #return hid3



class Linear(nn.Module):
    # 定义神经网络
    def __init__(self, n_feature=192, h_dim=3*32, h_dim2=32, n_output=10):
        # 初始化数组，参数分别是初始化信息，特征数，隐藏单元数，输出单元数
        super(Linear, self).__init__()
        self.fc0 = nn.Linear(n_feature, n_feature)
        self.fc1 = nn.Linear(n_feature, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, h_dim2)  # mu
        self.fc3 = nn.Linear(h_dim2, n_output)  # log_var

        self.W = [self.fc0.weight, self.fc1.weight, self.fc2.weight, self.fc3.weight]

    # 设置隐藏层到输出层的函数

    def forward(self, x):
        # 定义向前传播函数
        a0 = self.fc0(x)
        h0 = F.relu(a0)
        a1 = self.fc1(h0)
        h1 = F.relu(a1)
        a2 = self.fc2(h1)
        h2 = F.relu(a2)
        z = self.fc3(h2)

        # hid0 = F.relu(self.fc0(x))
        # hid1 = F.relu(self.fc1(hid0))
        # hid2 = F.relu(self.fc2(hid1))
        # # 给x加权成为a，用激励函数将a变成特征b
        # hid3 = F.softmax(self.fc3(hid2))
        cache = (a0, h0, a1, h1, a2, h2)
        z.retain_grad()
        for c in cache:
            c.retain_grad()

        return z




def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    print('dict_users', len(dict_users))
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 2*num_users , int(len(dataset)/(2*num_users)) #2*
    # num_shards, num_imgs = 20, 3000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def auxiliary(dict_users, num_users):
    each_user_has = len(dict_users[0])
    each_put = int (each_user_has / num_users)
    dict_users[num_users] = []
    for i in range(num_users):
        rand_set = list(set(np.random.choice(dict_users[i], each_put, replace=False)))
        dict_users[num_users] = np.concatenate((dict_users[num_users], rand_set), axis=0).astype(int)
    return dict_users, num_users+1


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = num_users*2 , int(len(dataset)/(2*num_users))
    # num_shards, num_imgs = 20, 3000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.array( dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


class CNNMnist(nn.Module):
    def __init__(self , args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, args.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, args):
        self.index=0
        self.dataset = dataset
        #         self.idxs = list(idxs)
        #         self.idxs = random.sample(list(idxs), int(len(idxs)*sampling))
        if args.sampling == 1:
            self.idxs = list(idxs)
        else:
            self.idxs = np.random.choice(list(idxs), size=int(len(idxs) * args.sampling), replace=True)
            #self.idxs = random.sample(list(idxs), int(len(idxs) * sampling)) # without replacement
            # random.choice is with replacement
        # print('datasplite' , idxs, len(dataset))

        self.data, self.targets = self.get_image_label()

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        #print("item", item, self.index, self.idxs[item],label)
        self.index+=1
        #print("self.idxs", self.idxs)
        return image, label

    def get_image_label(self, ):
        if args.dataset=="MNIST":
            temp_img = torch.empty(0, 1, 28,28).float().to(args.device)
            temp_label = torch.empty(0).long().to(args.device)
            for id in self.idxs:
                image, label = self.dataset[id]
                image, label = image.reshape(1, 1, 28, 28).to(args.device), torch.tensor([label]).long().to(args.device)
                #print(image)
                # print(label)
                # label = torch.tensor([label])
                temp_img = torch.cat([temp_img, image], dim=0)
                temp_label = torch.cat([temp_label, label], dim=0)
        elif args.dataset=="CIFAR10":
            temp_img = torch.empty(0, 3, 32, 32).float().to(args.device)
            temp_label = torch.empty(0).long().to(args.device)
            for id in self.idxs:
                image, label = self.dataset[id]
                image, label = image.to(args.device).reshape(1, 3, 32, 32), torch.tensor([label]).long().to(args.device)
                # print(label)
                # label = torch.tensor([label])
                temp_img = torch.cat([temp_img, image], dim=0)
                temp_label = torch.cat([temp_label, label], dim=0)

        print(temp_label.shape, temp_img.shape)
        d = Data.TensorDataset(temp_img, temp_label)
        return temp_img, temp_label

def add_noise(data, epsilon, sensitivity, args):
    noise_tesnor = np.random.laplace(1, sensitivity/epsilon,data.shape) * args.lr
    # data = torch.add(data, torch.from_numpy(noise_tesnor))
    # for x in np.nditer(np_data, op_flags=['readwrite']):
    #     x[...] = x + np.random.laplace(1, sensitivity/epsilon,)
    if args.gpu == -1:
        return data.add(torch.from_numpy(noise_tesnor).float())
    else:
        return data.add(torch.from_numpy(noise_tesnor).float().to(args.device))


class AdaHessian(torch.optim.Optimizer):
    """
    Implements the AdaHessian algorithm from "ADAHESSIAN: An Adaptive Second OrderOptimizer for Machine Learning"

    Arguments:
        params (iterable) -- iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional) -- learning rate (default: 0.1)
        betas ((float, float), optional) -- coefficients used for computing running averages of gradient and the squared hessian trace (default: (0.9, 0.999))
        eps (float, optional) -- term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional) -- weight decay (L2 penalty) (default: 0.0)
        hessian_power (float, optional) -- exponent of the hessian trace (default: 1.0)
        update_each (int, optional) -- compute the hessian trace approximation only after *this* number of steps (to save time) (default: 1)
        n_samples (int, optional) -- how many times to sample `z` for the approximation of the hessian trace (default: 1)
    """

    def __init__(self, params, lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 hessian_power=1.0, update_each=1, n_samples=1, average_conv_kernel=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError(f"Invalid Hessian power value: {hessian_power}")

        self.n_samples = n_samples
        self.update_each = update_each
        self.average_conv_kernel = average_conv_kernel

        # use a separate generator that deterministically generates the same `z`s across all GPUs in case of distributed training
        self.generator = torch.Generator().manual_seed(2147483647)

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, hessian_power=hessian_power)
        super(AdaHessian, self).__init__(params, defaults)

        for p in self.get_params():
            p.requires_grad = True
            p.hess = 0.0
            self.state[p]["hessian step"] = 0

    def get_params(self):
        """
        Gets all parameters in all param_groups with gradients
        """

        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)

    def zero_hessian(self):
        """
        Zeros out the accumalated hessian traces.
        """

        for p in self.get_params():
            if not isinstance(p.hess, float) and self.state[p]["hessian step"] % self.update_each == 0:
                p.hess.zero_()

    @torch.no_grad()
    def set_hessian(self):
        """
        Computes the Hutchinson approximation of the hessian trace and accumulates it for each trainable parameter.
        """

        params = []
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            if self.state[p]["hessian step"] % self.update_each == 0:  # compute the trace only each `update_each` step
                params.append(p)
            self.state[p]["hessian step"] += 1

        if len(params) == 0:
            return

        if self.generator.device != params[0].device:  # hackish way of casting the generator to the right device
            self.generator = torch.Generator(params[0].device).manual_seed(2147483647)

        # for p in params:
        #     # p.grad.requires_grad=True
        #     print(p.shape)
        #     print(p.grad.shape)
        grads = [p.grad for p in params]
        # grads.requires_grad = True
        # print("grads", grads)

        for i in range(self.n_samples):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator, device=p.device) * 2.0 - 1.0 for p in params]  # Rademacher distribution {-1.0, 1.0}

            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True, retain_graph=i < self.n_samples - 1)
            for h_z, z, p in zip(h_zs, zs, params):
                # print(h_z, z)
                p.hess += h_z * z / self.n_samples  # approximate the expected values of z*(H@z)
                # print("p.hess", p.hess.shape)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional) -- a closure that reevaluates the model and returns the loss (default: None)
        """

        loss = None
        if closure is not None:
            loss = closure()

        self.zero_hessian()
        self.set_hessian()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None:
                    continue

                if self.average_conv_kernel and p.dim() == 4:
                    p.hess = torch.abs(p.hess).mean(dim=[2, 3], keepdim=True).expand_as(p.hess).clone()

                # Perform correct stepweight decay as in AdamW
                p.mul_(1 - group['lr'] * group['weight_decay'])

                state = self.state[p]

                # State initialization
                if len(state) == 1:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # Exponential moving average of gradient values
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)  # Exponential moving average of Hessian diagonal square values

                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(p.hess, p.hess, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                k = group['hessian_power']
                denom = (exp_hessian_diag_sq / bias_correction2).pow_(k / 2).add_(group['eps'])

                # make update
                step_size = group['lr'] / bias_correction1
                #p.addcdiv_(exp_avg, denom, value=-step_size)
                p = p.addcdiv_(exp_avg, denom, value=step_size)

        return self.get_params()

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, need_poison=None, poison_data=None,poison_targets=None, erased_perm=None, idx=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.erased_perm = erased_perm
        self.idx = idx
        # prepare the local original dataset
        self.train_splite = DatasetSplit(dataset, idxs, args)
        self.poison_data = poison_data
        self.poison_targets = poison_targets
        #print(self.train_splite.dataset)
        # self.train_set = self.train_splite.get_image_label()
        if need_poison:
            if self.args.dataset=='MNIST':
                data_reshape = self.train_splite.data.reshape(len(self.train_splite.data), 1, 28, 28)
            elif self.args.dataset=='CIFAR10':
                data_reshape = self.train_splite.data.reshape(len(self.train_splite.data), 3, 32, 32)
            data = torch.cat([poison_data, data_reshape], dim=0)
            targets = torch.cat([poison_targets, self.train_splite.targets], dim=0)
            self.poison_trainset = Data.TensorDataset(data, targets) #Data.TensorDataset(data, targets)
            self.pure_backdorred_set = Data.TensorDataset(poison_data, poison_targets)

            """in a backdoored medol, we need to unlearn the trigger, 
            so the remaining dataset is all the clean samples, and the erased dataset is the poisoned samples"""
            self.remaining_set = Data.TensorDataset(data_reshape, self.train_splite.targets)
            self.erasing_set = self.pure_backdorred_set
        else:
            if self.args.dataset=='MNIST':
                data_reshape = self.train_splite.data.reshape(len(self.train_splite.data), 1, 28, 28)
            elif self.args.dataset=='CIFAR10':
                data_reshape = self.train_splite.data.reshape(len(self.train_splite.data), 3, 32, 32)
            self.poison_trainset = Data.TensorDataset(data_reshape, self.train_splite.targets)
            self.remaining_set = self.poison_trainset #Data.TensorDataset(data_reshape, self.train_splite.targets)
            #self.erasing_set = self.pure_backdorred_set



    @staticmethod
    def add_noise(data, epsilon, sensitivity, args):
        noise_tesnor = np.random.laplace(1, sensitivity / epsilon, data.shape)
        # data = torch.add(data, torch.from_numpy(noise_tesnor))
        # for x in np.nditer(np_data, op_flags=['readwrite']):
        #     x[...] = x + np.random.laplace(1, sensitivity/epsilon,)
        if args.gpu == -1:
            return data.add(torch.from_numpy(noise_tesnor).float())
        else:
            return data.add(torch.from_numpy(noise_tesnor).float().to(args.device))

    @staticmethod
    def jacobian(y, x, create_graph=False):
        jac = []
        flat_y = y.reshape(-1)
        grad_y = torch.zeros_like(flat_y)
        for i in range(len(flat_y)):
            grad_y[i] = 1.
            grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
            jac.append(grad_x.reshape(x.shape))
            grad_y[i] = 0.
        return torch.stack(jac).reshape(y.shape + x.shape)

    @staticmethod
    def hessian(y, x):
        return LocalUpdate.jacobian(LocalUpdate.jacobian(y, x, create_graph=True), x)


    @staticmethod
    @torch.no_grad()
    def hessian_unl_update(p, hess, args, i):
        average_conv_kernel = False
        weight_decay = 0.0
        betas = (0.9, 0.999)
        hessian_power = 1.0
        eps = args.lr # 1e-8

        if average_conv_kernel and p.dim() == 4:
            hess = torch.abs(hess).mean(dim=[2, 3], keepdim=True).expand_as(hess).clone()

        # Perform correct stepweight decay as in AdamW
        # p = p.mul_(1 - args.lr * weight_decay)

        state = {}
        state["hessian"] = 1

        # State initialization
        if len(state) == 1:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p.data)  # Exponential moving average of gradient values
            state['exp_hessian_diag_sq'] = torch.zeros_like(
                p.data)  # Exponential moving average of Hessian diagonal square values

        exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']
        beta1, beta2 = betas
        state['step'] = i

        # Decay the first and second moment running average coefficient

        exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
        #exp_hessian_diag_sq.mul_(beta2).addcmul_(p_hs.hess, p_hs.hess, value=1 - beta2)
        exp_hessian_diag_sq.mul_(beta2).addcmul_(hess, hess, value=1 - beta2)

        bias_correction1 = 1 #- beta1 ** state['step']
        bias_correction2 = 1 #- beta2 ** state['step']

        k = hessian_power
        denom = (exp_hessian_diag_sq / bias_correction2).pow_(k / 2).add_(eps)

        # make update
        step_size = args.lr / bias_correction1
        # p.addcdiv_(exp_avg, denom, value=-step_size)
        p = p.addcdiv_(exp_avg, denom, value=step_size * 0.1)
        #p_hs.data = p_hs.data + args.lr * p_hs.grad.data * 10
        return exp_avg, denom, step_size


    def train(self, net, idx, args):

        self.ldr_train = DataLoader(self.poison_trainset, batch_size=self.args.local_bs, shuffle=True)
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                #images, labels = images.to(self.args.device), labels.to(self.args.device)
                images, labels = torch.tensor(images, requires_grad=True).to(self.args.device), torch.tensor(labels, requires_grad=False).to(self.args.device)
                B,c,h,w = images.shape
                #print(B,h,w)
                if args.dataset=='MNIST':
                    images = images.reshape((B, -1))
                net.zero_grad()
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                fl_acc = (log_probs.argmax(dim=1) == labels).float().mean().item()
                # print('batch_idx', batch_idx)
                if batch_idx % 1000 == 0 and iter==0:
                    print("fl_acc", fl_acc, "loss", loss.item(), "idx", idx)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def re_train(self, net, idx, args):
        self.remaining_loader = DataLoader(self.remaining_set, batch_size=self.args.local_bs, shuffle=True)
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            # print(iter)
            for batch_idx, (images, labels) in enumerate(self.remaining_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                B,c,h,w = images.shape
                #print(B,h,w)
                if args.dataset=='MNIST':
                    images = images.reshape((B, -1))

                net.zero_grad()
                log_probs = net(images)


                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                fl_acc = (log_probs.argmax(dim=1) == labels).float().mean().item()
                # print('batch_idx', batch_idx)
                if batch_idx % 1000 == 0 and iter==0:
                    print("fl_acc", fl_acc, "loss", loss.item(), "idx", idx)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def train_Bayes(self, net, idx, args):

        # train and update
        self.ldr_train = DataLoader(self.poison_trainset, batch_size=self.args.local_bs, shuffle=True)
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

        reconstruction_function = nn.MSELoss(size_average=True)

        for epoch in range(self.args.local_ep):
            step_start = epoch * len(self.ldr_train)
            net, optimizer = LocalUpdate.learning_train(self.ldr_train, net, step_start, self.loss_func, reconstruction_function,
                                             optimizer, args, epoch,idx)
            # net.eval()
        return net.state_dict()

    def retrain_Bayes(self, net, net_org, net_unl, idx, args):
        # train and update
        self.remaining_loader = DataLoader(self.remaining_set, batch_size=self.args.local_bs, shuffle=True)
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

        reconstruction_function = nn.MSELoss(size_average=True)
        KL_fr = []
        KL_er = []
        KL_nipsr = []
        KL_kl = []
        mu_list_f = []
        sigma_list_f = []
        mu_list_e = []
        sigma_list_e = []
        mu_list_r = []
        sigma_list_r = []
        for epoch in range(self.args.local_ep):
            step_start = epoch * len(self.ldr_train)
            net, optimizer, KL_fr, KL_nipsr = LocalUpdate.retraining_kld(net, optimizer, self.remaining_loader, self.loss_func, reconstruction_function, net_org, net_unl, args, epoch, idx, KL_fr, KL_nipsr)
            # net.eval()
        return net.state_dict(), KL_fr, KL_nipsr

    def unlearn_Bayes(self, net, net_temp, idx, args, train_type):
        # train and update
        #self.remaining_loader = DataLoader(self.remaining_set, batch_size=self.args.local_bs, shuffle=True)
        if idx not in self.erased_perm:
            return net.state_dict()

        temp_img = torch.empty(0, 1, 28, 28).float().to(args.device)
        temp_label = torch.empty(0).long().to(args.device)
        temp_img_e = torch.empty(0, 1, 28, 28).float().to(args.device)
        temp_label_e = torch.empty(0).long().to(args.device)
        temp_i = 0
        for image, label in self.remaining_set:
            image, label = image.reshape(1, 1, 28, 28).to(args.device), torch.tensor([label]).long().to(args.device)
            # print(label)
            # label = torch.tensor([label])
            temp_img = torch.cat([temp_img, image], dim=0)
            temp_label = torch.cat([temp_label, label], dim=0)

            temp_i = temp_i + 1
            # if temp_i >= self.args.erased_size*2:
            #     break
            # temp_img = torch.cat([temp_img, image], dim=0)
            # temp_label = torch.cat([temp_label, label], dim=0)

        for i in range(1):
            for image_e, label_e in self.erasing_set:
                image_e, label_e = image_e.reshape(1, 1, 28, 28).to(args.device), torch.tensor([label_e]).long().to(args.device)
                # print(label)
                # label = torch.tensor([label])
                temp_img_e = torch.cat([temp_img_e, image_e], dim=0)
                temp_label_e = torch.cat([temp_label_e, label_e], dim=0)

        print("sharp", temp_img.shape, temp_img_e.shape)
        remain_set = Data.TensorDataset(temp_img, temp_label)
        erased_set = Data.TensorDataset(temp_img_e, temp_label_e)
        #remaining_loader = DataLoader(self.remaining_set, batch_size=self.args.local_bs, shuffle=True)
        remaining_loader = DataLoader(remain_set, batch_size=self.args.local_bs, shuffle=True)
        self.erased_loader = DataLoader(erased_set, batch_size=self.args.local_bs, shuffle=True)

        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

        reconstruction_function = nn.MSELoss(size_average=True)

        user_acc_list = []
        user_unlearn_list=[]
        convergence = 0
        for epoch in range(self.args.local_ep+40):
            step_start = epoch * len(self.ldr_train)
            net, optimizer, acc_list, unlearned_acc_list = LocalUpdate.unlearning_nips(net, optimizer, self.erased_loader, remaining_loader, self.loss_func, reconstruction_function, net_temp, args ,epoch, idx, train_type)
            user_acc_list.append(np.mean(acc_list))
            user_unlearn_list.append(np.mean(unlearned_acc_list))
            if np.mean(unlearned_acc_list) < 0.1:
                convergence = convergence + 1
            if convergence >= args.unl_conver_r and train_type=='unlearn_nips': # if accuracy on unlearned dataset performs lower than random, we think it converges
                print("unlearn finish",epoch, np.mean(unlearned_acc_list))
                print("user_unlearn_list", user_unlearn_list)
                break
            if convergence >= args.unl_conver_r and train_type=='self-sharing': # if accuracy on unlearned dataset performs lower than random, we think it converges
                print("unlearn finish",epoch, np.mean(unlearned_acc_list))
                print("user_unlearn_list", user_unlearn_list)
                break
            # net.eval()
        if idx in self.erased_perm:
            print("user_acc_list", user_acc_list)
        return net, net.state_dict()

    @staticmethod
    def learning_train(train_loader, model, step_start, loss_fn, reconstruction_function, optimizer, args, epoch, idx):
        logs = defaultdict(list)
        mu_list = []
        sigma_list = []
        init_epoch = 0

        for step, (x, y) in enumerate(train_loader, start=step_start):
            x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)

            # print("x org",x)
            # break
            x = x.view(x.size(0), -1)
            logits_z, logits_y, x_hat, mu, logvar = model(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)
            H_p_q = loss_fn(logits_y, y)
            KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
            KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
            KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()

            x_hat = x_hat.view(x_hat.size(0), -1)
            x = x.view(x.size(0), -1)
            # x = torch.sigmoid(torch.relu(x))
            BCE = reconstruction_function(x_hat, x)  # mse loss
            loss = args.beta * KLD_mean + H_p_q #+ BCE #+ BCE / (args.local_bs * 28 * 28)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
            optimizer.step()

            acc = (logits_y.argmax(dim=1) == y).float().mean().item()
            sigma = torch.sqrt_(torch.exp(logvar)).mean().item()
            # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()
            metrics = {
                'idx': idx,
                'acc': acc,
                'loss': loss.item(),
                'BCE': BCE.item(),
                'H(p,q)': H_p_q.item(),
                # '1-JS(p,q)': JS_p_q,
                'mu': torch.mean(mu).item(),
                #'sigma': sigma,
                #'KLD': KLD.item(),
                'KLD_mean': KLD_mean.item(),
            }

            for m, v in metrics.items():
                logs[m].append(v)
            if epoch == args.num_epochs - 1:
                mu_list.append(torch.mean(mu).item())
                sigma_list.append(sigma)
            if step % len(train_loader) % 600 == 0 and epoch==2:
                print(f'[{epoch}/{init_epoch + args.num_epochs}:{step % len(train_loader):3d}] '
                      + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
        return model, optimizer

    @staticmethod
    def unlearning_nips(net, optimizer_nips, dataloader_erase, remaining_loader, loss_fn, reconstruction_function, net_temp, args, epoch, idx, train_type):
        logs = defaultdict(list)
        mu_list = []
        sigma_list = []
        init_epoch = 0
        net.train()
        step_start = epoch * len(dataloader_erase)
        step = 0
        acc_list = []
        index=0
        unlearned_acc_list = []
        for   (x, y), (x2,y2) in   zip(dataloader_erase, remaining_loader)  :
            index = index + 1
            #print("index",index)
            x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
            x = x.view(x.size(0), -1)

            x2, y2 = x2.to(args.device), y2.to(args.device)  # (B, C, H, W), (B, 10)
            x2 = x2.view(x2.size(0), -1)

            logits_z_e, logits_y_e, x_hat_e, mu_e, logvar_e = net(x, mode='forgetting')
            logits_z_e2, logits_y_e2, x_hat_e2, mu_e2, logvar_e2 = net(x2, mode='forgetting')
            logits_z_f, logits_y_f, x_hat_f, mu_f, logvar_f = net_temp(x, mode='forgetting')
            # logits_y_e = torch.softmax(logits_y_e, dim=1)
            logits_z_e_log_softmax = logits_z_e.log_softmax(dim=1)
            p_x_e = x.softmax(dim=1)
            B = x.size(0)
            H_p_q = loss_fn(logits_y_e, y)  # -log p(y|(x;\theta))
            H_p_q_f = loss_fn(logits_y_e, logits_y_f.argmax(dim=1))
            H_p_q2 = loss_fn(logits_y_e2, y2)
            KLD_element2 = mu_e2.pow(2).add_(logvar_e2.exp()).mul_(-1).add_(1).add_(logvar_e2).to(args.device)
            KLD_mean2 = torch.mean(KLD_element2).mul_(-0.5).to(args.device)

            KLD = 0.5 * torch.mean(
                logvar_f - logvar_e + (torch.exp(logvar_e) + (mu_e - mu_f).pow(2)) / torch.exp(logvar_f) - 1).cuda()

            KLD_mean = 0.5 * torch.mean(
                logvar_f - logvar_e + (torch.exp(logvar_e) + (mu_e - mu_f).pow(2)) / torch.exp(logvar_f) - 1).cuda()

            KLD_mean_r = 0.5 * torch.mean(
                logvar_e - logvar_f + (torch.exp(logvar_f) + (mu_f - mu_e).pow(2)) / torch.exp(logvar_e) - 1).cuda()

            KL_z_r = (torch.exp(logits_z_e_log_softmax) * logits_z_e_log_softmax).sum(dim=1).mean() + math.log(
                logits_z_e_log_softmax.shape[1])

            # x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)
            # x_hat_e = torch.sigmoid(reconstructor(logits_z_e))
            x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)

            #x_hat_f = torch.sigmoid(reconstructor(logits_z_f))
            #x_hat_f = x_hat_f.view(x_hat_f.size(0), -1)
            # x = torch.sigmoid(torch.relu(x))
            x = x.view(x.size(0), -1)
            # x = torch.sigmoid(x)
            BCE = reconstruction_function(x_hat_e, x)  # mse loss = - log p = log 1/p
            # BCE = torch.mean(x_hat_e.log_softmax(dim=1))
            e_log_p = torch.exp(BCE / (args.local_bs * 28 * 28))  # = 1/p
            e_log_py = torch.exp(-H_p_q)
            log_z = torch.mean(logits_z_e.log_softmax(dim=1))
            kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
            unlearn_learning_rate = args.unlearn_learning_rate
            self_sharing_rate = args.self_sharing_rate
            if train_type == 'unlearn_nips':
                loss = KLD_mean - unlearn_learning_rate * H_p_q
            elif train_type == 'self-sharing':
                loss = KLD_mean  - unlearn_learning_rate * H_p_q + self_sharing_rate * (args.beta * KLD_mean2 + H_p_q2)

            optimizer_nips.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
            optimizer_nips.step()
            acc = (logits_y_e.argmax(dim=1) == y).float().mean().item()
            acc2 = (logits_y_e2.argmax(dim=1) == y2).float().mean().item()
            # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()
            metrics = {
                'idx':idx,
                'acc': acc,
                'loss': loss.item(),
                'BCE': BCE.item(),
                'H_p_q':H_p_q.item(),
                # '1-JS(p,q)': JS_p_q,
                #'mu_e': torch.mean(mu_e).item(),
                #'sigma_e': torch.sqrt_(torch.exp(logvar_e)).mean().item(),
                #'KLD': KLD.item(),
                'e_log_py': e_log_py.item(),
                'log_z': log_z.item(),
                'KLD_mean': KLD_mean.item(),
            }

            for m, v in metrics.items():
                logs[m].append(v)
            if epoch == args.num_epochs - 1:
                mu_list.append(torch.mean(mu_e).item())
                sigma_list.append(torch.sqrt_(torch.exp(logvar_e)).mean().item())
            step=step+1
            unlearned_acc_list.append(acc)
            acc_list.append(acc2)
            if index % len(dataloader_erase) == 0 and epoch==2:
                print(f'[{epoch}/{init_epoch + args.num_epochs}:{step % len(dataloader_erase):3d}] '
                      + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))


        net.eval()
        return net, optimizer_nips, acc_list, unlearned_acc_list

    @staticmethod
    def retraining_kld(net, optimizer_retrain, dataloader_remain, loss_fn, reconstruction_function, net_org, net_unl, args, epoch, idx, KL_fr, KL_nipsr):
        logs = defaultdict(list)
        mu_list = []
        sigma_list = []
        init_epoch = 0
        net.train()
        step_start = epoch * len(dataloader_remain)

        for step, (x, y) in enumerate(dataloader_remain, start=step_start):

            x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
            x = x.view(x.size(0), -1)
            logits_z_r, logits_y_r, x_hat_r, mu_r, logvar_r = net(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)

            # logits_z_e, logits_y_e, x_hat_e, mu_e, logvar_e = vibi_f_frkl(x, mode='forgetting')
            # logits_z_e_kl, logits_y_e_kl, x_hat_e_kl, mu_e_kl, logvar_e_kl = vibi_f_kl(x, mode='forgetting')
            logits_z_e_nips, logits_y_e_nips, x_hat_e_nips, mu_e_nips, logvar_e_nips = net_unl(x, mode='forgetting')
            # print(x_hat_e)
            logits_z_f, logits_y_f, mu_f, logvar_f = net_org(x, mode='distribution')
            # logits_y_r = torch.softmax(logits_y_r, dim=1)
            logits_z_r_softmax = logits_z_r.log_softmax(dim=1)
            p_x_r = x.softmax(dim=1)

            KLD_element = mu_r.pow(2).add_(logvar_r.exp()).mul_(-1).add_(1).add_(logvar_r).cuda()
            KLD = torch.mean(KLD_element).mul_(-0.5).cuda()
            KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()
            # x_hat_r = torch.sigmoid(reconstructor(logits_z_r))
            x_hat_r = x_hat_r.view(x_hat_r.size(0), -1)
            x = x.view(x.size(0), -1)
            #x_hat_e = torch.sigmoid(reconstructor(logits_z_e))
            #x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)

            #x_hat_f = torch.sigmoid(reconstructor(logits_z_f))
            #x_hat_f = x_hat_f.view(x_hat_f.size(0), -1)
            BCE = reconstruction_function(x_hat_r, x)  # mse loss
            H_p_q = loss_fn(logits_y_r, y)
            loss_r = args.beta * KLD_mean + H_p_q #+ BCE #+ BCE / (args.local_bs * 28 * 28)

            optimizer_retrain.zero_grad()
            loss_r.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
            optimizer_retrain.step()

            KLD_fr = 0.5 * torch.mean(
                logvar_r - logvar_f + (torch.exp(logvar_f) + (mu_f - mu_r).pow(2)) / torch.exp(logvar_r) - 1)

            # KLD_er = 0.5 * torch.mean(
            #     logvar_r - logvar_e + (torch.exp(logvar_e) + (mu_e - mu_r).pow(2)) / torch.exp(logvar_r) - 1)

            # KLD_klr = 0.5 * torch.mean(
            #     logvar_r - logvar_e_kl + (torch.exp(logvar_e_kl) + (mu_e_kl - mu_r).pow(2)) / torch.exp(logvar_r) - 1)
            #
            KLD_nips = 0.5 * torch.mean(
                logvar_r - logvar_e_nips + (torch.exp(logvar_e_nips) + (mu_e_nips - mu_r).pow(2)) / torch.exp(logvar_r) - 1)
            acc = (logits_y_r.argmax(dim=1) == y).float().mean().item()
            # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()
            metrics = {
                'idx': idx,
                'acc': acc,
                'loss': loss_r.item(),
                'BCE': BCE.item(),
                'H(p,q)': H_p_q.item(),
                #'mu_r': torch.mean(mu_r).item(),
                #'sigma_r': torch.sqrt_(torch.exp(logvar_r)).mean().item(),
                'KLD_fr': KLD_fr.item(),
                'KLD_nips': KLD_nips.item(),
                'KLD_mean': KLD_mean.item(),
            }

            for m, v in metrics.items():
                logs[m].append(v)
            if epoch == args.num_epochs - 1:
                KL_fr.append(KLD_fr.item())
                # KL_er.append(KLD_er.item())
                # KL_kl.append(KLD_klr.item())
                KL_nipsr.append(KLD_nips.item())
            # if epoch == args.num_epochs - 1:
            #     mu_list_r.append(torch.mean(mu_r).item())
            #     sigma_list_r.append(torch.sqrt_(torch.exp(logvar_r)).mean().item())
            #     mu_list_f.append(torch.mean(mu_f).item())
            #     sigma_list_f.append(torch.sqrt_(torch.exp(logvar_f)).mean().item())
            #     mu_list_e.append(torch.mean(mu_e_nips).item())
            #     sigma_list_e.append(torch.sqrt_(torch.exp(logvar_e_nips)).mean().item())

            if step % len(dataloader_remain) % 6000 == 0 and epoch==2:
                print(f'[{epoch}/{init_epoch + args.num_epochs}:{step % len(dataloader_remain):3d}] '
                      + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))

        net.eval()
        return net, optimizer_retrain, KL_fr, KL_nipsr

    def unl_train(self, net, idx, args):

        self.remaining_loader = DataLoader(self.remaining_set, batch_size=self.remaining_set.__len__(), shuffle=True)
        self.erased_loader = DataLoader(self.erasing_set, batch_size=self.args.local_bs, shuffle=True)

        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        epoch_loss = []
        acc_list = []
        backdoor_list = []
        params_with_hs = None

        #prepare hessian
        for batch_idx, (images, labels) in enumerate(self.remaining_loader):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            B, c, h, w = images.shape
            # print(B,h,w)
            images = images.reshape((B, -1))
            net.zero_grad()

            logits_z, logits_y, x_hat, mu, logvar = net(images, mode='forgetting')  # (B, C* h* w), (B, N, 10)
            H_p_q = self.loss_func(logits_y, labels)
            KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
            KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
            KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()

            loss = args.beta * KLD_mean + H_p_q #+ BCE / (args.local_bs * 28 * 28)
            optimizer.zero_grad()

            # loss.backward()
            # log_probs = net(images)
            # loss = self.loss_func(log_probs, labels)

            loss.backward(create_graph=True)

            optimizer_hs = AdaHessian(net.parameters())
            # optimizer_hs.get_params()
            optimizer_hs.zero_hessian()
            optimizer_hs.set_hessian()

            params_with_hs = optimizer_hs.get_params()

            net.zero_grad()


        # unlearning
        convergence = 0
        for iter in range(args.local_ep): #self.args.local_ep
            batch_loss = []
            # print(iter)
            temp_acc = []
            temp_back = []
            for (images, labels), (images2, labels2) in zip(self.erased_loader, self.remaining_loader):
            # for batch_idx, (images, labels) in enumerate(self.erased_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                B,c,h,w = images.shape
                #print(B,h,w)
                images  = images.reshape((B, -1))

                images2, labels2 = images2.to(self.args.device), labels2.to(self.args.device)
                B, c, h, w = images2.shape

                images2 = images2.reshape((B, -1))

                net.zero_grad()
                logits_z, logits_y, x_hat, mu, logvar = net(images, mode='forgetting')  # (B, C* h* w), (B, N, 10)
                logits_z2, logits_y2, x_hat2, mu2, logvar2 = net(images2, mode='forgetting')
                H_p_q = self.loss_func(logits_y, labels)
                KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
                KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
                KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()

                loss = args.beta * KLD_mean + H_p_q  # + BCE / (args.local_bs * 28 * 28)
                optimizer.zero_grad()

                loss.backward()
                # log_probs = net(images)
                # loss = self.loss_func(log_probs, labels)

                #loss.backward(create_graph=True)
                #loss.backward()

                # new_p = optimizer_hs.step()
                # for name, p in net.named_parameters():
                #     print(name, p.grad.data)
                #     if p.grad.data==None:
                #         break

                i=0
                for p_hs, p in zip(params_with_hs, net.parameters()):
                    i = i + 1
                    # if i==1:
                    #     continue
                    # print(p_hs.hess)
                    # break
                    temp_hs = torch.tensor(p_hs.hess)
                    #temp_hs = temp_hs.__add__(args.lr)
                    #p.data = p.data.addcdiv_(exp_avg, denom, value=-step_size * 10000)

                    #print(p.data)
                    #p.data = p_hs.data.addcdiv_(exp_avg, denom, value=step_size * args.lr)
                    if p.grad!=None:
                        exp_avg, denom, step_size = LocalUpdate.hessian_unl_update(p, temp_hs, args, i)
                        p.data = p.data.addcdiv_(exp_avg, denom, value=step_size* args.hessian_rate)
                        #p.data =p.data + torch.div(p.grad.data, temp_hs) * args.lr #torch.mul(p_hs.hess, p.grad)*10
                        print(p.grad.data.shape)
                    else:
                        p.data =p.data


                #optimizer.step()
                net.zero_grad()

                fl_acc = (logits_y.argmax(dim=1) == labels).float().mean().item()
                fl_acc2 = (logits_y2.argmax(dim=1) == labels2).float().mean().item()
                temp_acc.append(fl_acc2)
                temp_back.append(fl_acc)

                if batch_idx % 1000 == 0 and iter==0:
                    print("fl_acc2", fl_acc2, 'backdoor',fl_acc, "loss", loss.item(), "idx", idx)
                batch_loss.append(loss.item())

            acc_list.append(np.mean(temp_acc))
            backdoor_list.append(np.mean(temp_back))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if np.mean(temp_back) < 0.1:
                convergence = convergence + 1
            if convergence >= args.unl_conver_r:
                break
        print("backdoor_list", backdoor_list)
        print("acc_list", acc_list)
        #net_local_w, loss_item = self.re_train(net, idx, args)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def unl_train_fisher(self, net, idx, args):
        #self.remaining_loader = DataLoader(self.remaining_set, batch_size=self.args.local_bs, shuffle=True)
        self.erased_loader = DataLoader(self.erasing_set, batch_size=self.args.local_bs, shuffle=True)

        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        epoch_loss = []
        A = []  # KFAC A
        G = []  # KFAC G
        for Wi in net.W:
            A.append(torch.zeros(Wi.size(1)))
            G.append(torch.zeros(Wi.size(0)))

        A_inv, G_inv = 4 * [0], 4 * [0]
        eps = 1e-1
        alpha = 0.02
        for iter in range(1): #self.args.local_ep
            batch_loss = []
            # print(iter)
            for batch_idx, (images, labels) in enumerate(self.erased_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                B,c,h,w = images.shape
                #print(B,h,w)
                images  = images.reshape((B, -1))
                # print(labels)
                # print('batch_idx', batch_idx, images.shape)
                #if self.args.add_noise:
                #   images =LocalUpdate.add_noise( data=images, epsilon=self.args.epsilon, sensitivity=1,args=self.args)
                net.zero_grad()
                log_probs, cache = net(images)
                a0, h0, a1, h1, a2, h2 = cache
                loss = self.loss_func(log_probs, labels)
                # loss = F.nll_loss(log_probs, labels)
                loss.backward()

                # KFAC matrices
                # for name, param in net.named_parameters():
                #     print(param.grad)
                #     break
                # print("new")
                # print(a0.grad)
                # break
                G0_ = 1 / args.local_bs * a0.grad.t() @ a0.grad
                A0_ = 1 / args.local_bs * images.t() @ images
                G1_ = 1 / args.local_bs * a1.grad.t() @ a1.grad
                A1_ = 1 / args.local_bs * h0.t() @ h0
                G2_ = 1 / args.local_bs * a2.grad.t() @ a2.grad
                A2_ = 1 / args.local_bs * h1.t() @ h1
                G3_ = 1 / args.local_bs * log_probs.grad.t() @ log_probs.grad
                A3_ = 1 / args.local_bs * h2.t() @ h2

                G_ = [G0_, G1_, G2_, G3_]
                A_ = [A0_, A1_, A2_, A3_]

                # Update running estimates of KFAC
                rho = 0.0

                for k in range(4):
                    A[k] = rho * A[k].to(self.args.device) + (1 - rho) * A_[k].to(self.args.device)
                    G[k] = rho * G[k].to(self.args.device) + (1 - rho) * G_[k].to(self.args.device)

                # Step
                for k in range(4):
                    # Amortize the inverse. Only update inverses every now and then

                    A_inv[k] = (A[k] + eps * torch.eye(A[k].shape[0]).to(self.args.device)).inverse()
                    G_inv[k] = (G[k] + eps * torch.eye(G[k].shape[0]).to(self.args.device)).inverse()

                    # A_inv[k] = torch.diag_embed(torch.diag(A[k].inverse()))
                    # G_inv[k] = torch.diag_embed(torch.diag(G[k].inverse()))
                    delta = G_inv[k] @ net.W[k].grad.data @ A_inv[k]
                    #print("delta", delta)
                    net.W[k].data += alpha * delta

                # PyTorch stuffs

                #optimizer.step()
                net.zero_grad()

                fl_acc = (log_probs.argmax(dim=1) == labels).float().mean().item()
                # print('batch_idx', batch_idx)
                if batch_idx % 1000 == 0 and iter==0:
                    print("fl_acc", fl_acc, "loss", loss.item(), "idx", idx)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        #net_local_w, loss_item = self.re_train(net, idx, args)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)




def signSGD_RLR_grad(grad_locals, glob_w, threshold, lr, iter):
    grad_avg = copy.deepcopy(grad_locals[0])
    data_len = 0
    for i in range(len(grad_locals)):
        data_len += 1  # + np.random.laplace(0, beta, 1)[0]
    print('len_w', len(grad_locals), data_len)
    #init zero
    for k in grad_avg.keys():
        grad_avg[k] = grad_avg[k]-grad_avg[k]

    for k in grad_avg.keys():
        print(k, grad_avg[k].shape)
        for i in range(0, len(grad_locals)):
            temp = torch.sign(grad_locals[i][k])#/torch.abs(grad_locals[i][k])
            #print(temp)
            #temp = torch.add(grad_locals[i][k], 0) # np.random.laplace(0, beta, 1)[0] / 100
            # w_avg[k] += w[i][k]
            grad_avg[k] += temp
        #grad_avg[k] = torch.div(grad_avg[k], data_len)

    for k in grad_avg.keys():
        print(k, grad_avg[k].shape)
        # print(grad_avg[k].long())
        #temp = torch.abs(grad_avg[k])
        grad_avg[k] = torch.ge(grad_avg[k], threshold)

    w_avg = copy.deepcopy(glob_w)
    if iter==10:
        lr=0.001
    if iter==20:
        lr=0.0005
    for k in grad_avg.keys():
        grad_avg[k] = grad_avg[k].long()*2-1
        #print(grad_avg[k])
        w_avg[k] = glob_w[k] - lr * grad_avg[k]
    return w_avg

def signSGD_RLR_grad_pro(grad_locals, glob_w, args):
    grad_avg = copy.deepcopy(grad_locals[0])
    grad_avg_sign = copy.deepcopy(grad_locals[0])
    data_len = 0
    for i in range(len(grad_locals)):
        data_len += 1  # + np.random.laplace(0, beta, 1)[0]
    print('len_w', len(grad_locals), data_len)
    #init zero
    for k in grad_avg.keys():
        grad_avg[k] = grad_avg[k] - grad_avg[k]
        grad_avg_sign[k] = grad_avg_sign[k] - grad_avg_sign[k]

    for k in grad_avg_sign.keys():
        print(k, grad_avg_sign[k].shape)
        for i in range(0, len(grad_locals)):
            temp = torch.sign(grad_locals[i][k])#/torch.abs(grad_locals[i][k])
            #print(temp)
            #temp = torch.add(grad_locals[i][k], 0) # np.random.laplace(0, beta, 1)[0] / 100
            # w_avg[k] += w[i][k]
            grad_avg_sign[k] += temp
            #grad_avg[k] += grad_locals[i][k]
        #grad_avg[k] = torch.div(grad_avg[k], data_len)

    w_avg = copy.deepcopy(glob_w)

    for k in grad_avg_sign.keys():
        #print(k, grad_avg_sign[k])
        grad_avg_sign[k] = torch.abs(grad_avg_sign[k])
        grad_avg_sign[k] = torch.sign( grad_avg_sign[k] - args.num_users*0.1)
        # temp_sign_abs = torch.abs(grad_avg_sign[k])
        # grad_avg_sign[k] = torch.sign(grad_avg_sign[k])
        # temp_local = torch.sign(grad_locals[i][k])
        # temp_sign = torch.eq(temp_local, grad_avg_sign[k])*2-1
        for i in range(0, len(grad_locals)):
            temp = grad_locals[i][k] #* temp_sign * temp_sign_abs
            grad_avg[k] += temp
        grad_avg[k] = torch.div(grad_avg[k], data_len)
        #grad_avg_sign[k] = grad_avg_sign[k].long()*2-1
        #print(grad_avg_sign[k])
        #print(grad_avg[k])
        #print(grad_avg[k])
        # temp_ge = torch.ge(temp_sign_abs, 5)*2 -1
        #print(temp_sign_abs)
        w_avg[k] = glob_w[k] - grad_avg[k]*grad_avg_sign[k]
    return w_avg

def mc(net_state, w_locals):
    connect_ratio=0.8
    net_temp = copy.deepcopy(net_state)
    for k in net_state.keys():
        net_temp[k] = net_state[k] - net_state[k]
    for k in net_state.keys():
        for i in range(0, len(w_locals)):
            net_temp[k] = net_temp[k] +  w_locals[i][k] + connect_ratio * (net_state[k] - w_locals[i][k])  # (1-0.2)*(1-0.2)*net1[k] + 2*(0.2)*(1-0.2)*net[k] + (0.2)*(0.2)*net2[k]
        net_temp[k] = net_temp[k] /  len(w_locals)

    return net_temp



def signSGD_RLR_grad_our(grad_locals, glob_w):
    grad_avg = copy.deepcopy(grad_locals[0])
    grad_avg_sign = copy.deepcopy(grad_locals[0])
    data_len = 0
    for i in range(len(grad_locals)):
        data_len += 1  # + np.random.laplace(0, beta, 1)[0]
    print('len_w', len(grad_locals), data_len)
    #init zero
    for k in grad_avg.keys():
        grad_avg[k] = grad_avg[k]-grad_avg[k]
        grad_avg_sign[k] = grad_avg_sign[k] - grad_avg_sign[k]

    for k in grad_avg_sign.keys():
        print(k, grad_avg_sign[k].shape)
        for i in range(0, len(grad_locals)):
            temp = torch.sign(grad_locals[i][k])#/torch.abs(grad_locals[i][k])
            #print(temp)
            #temp = torch.add(grad_locals[i][k], 0) # np.random.laplace(0, beta, 1)[0] / 100
            # w_avg[k] += w[i][k]
            grad_avg_sign[k] += temp
            #grad_avg[k] += grad_locals[i][k]
        #grad_avg[k] = torch.div(grad_avg[k], data_len)

    w_avg = copy.deepcopy(glob_w)

    for k in grad_avg_sign.keys():
        # grad_avg_sign[k] = torch.abs(grad_avg_sign[k])
        # grad_avg_sign[k] = torch.ge(grad_avg_sign[k], 4).long()*2-1
        temp_sign_abs = torch.abs(grad_avg_sign[k]) - 4 # more than 4 votes
        grad_avg_sign[k] = torch.sign(grad_avg_sign[k]) #torch.sign(grad_avg_sign[k])
        temp_local = torch.sign(grad_locals[i][k])
        temp_sign = torch.eq(temp_local, grad_avg_sign[k])*2-1
        for i in range(0, len(grad_locals)):
            temp = grad_locals[i][k] * temp_sign #* temp_sign_abs
            grad_avg[k] += temp
        grad_avg[k] = torch.div(grad_avg[k], data_len)
        #grad_avg_sign[k] = grad_avg_sign[k].long()*2-1
        #print(grad_avg_sign[k])
        #print(grad_avg[k])
        #print(grad_avg[k])
        # temp_ge = torch.ge(temp_sign_abs, 5)*2 -1
        #print(temp_sign_abs)
        w_avg[k] = glob_w[k] - grad_avg[k] #*grad_avg_sign[k]
    return w_avg







def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    epsilon = 1
    beta = 1 / epsilon
    data_len = 0
    for i in range(len(w)):
        data_len += 1  # + np.random.laplace(0, beta, 1)[0]
    print('len_w', len(w) , data_len)
    temp_fc0 = torch.empty(0, 10).float().to(args.device)
    for k in w_avg.keys():
        #print(k, w_avg[k].shape)
        for i in range(1, len(w)):
            # print('before',w[i][k][1])

            temp = torch.add(w[i][k], 0) # np.random.laplace(0, beta, 1)[0] / 100
            # w_avg[k] += w[i][k]
            w_avg[k] += temp
            # if k.__eq__("fc3.bias"):
            #     temp = temp.reshape((1,10))
            #     print(temp)
            #     temp_fc0 = torch.cat([temp_fc0, temp], dim=0)
            # print(temp.shape)
            # print(temp)

            # print('after', temp[1])
        # w_avg[k] = torch.div(w_avg[k], len(w))
        w_avg[k] = torch.div(w_avg[k], data_len)
    # b = Data.TensorDataset(temp_fc0, temp_fc0)
    # lr_train_dl = DataLoader(b, batch_size=1, shuffle=True)
    #w_avg["fc3.bias"] = train_model_vae(lr_train_dl, data_len)

    return w_avg


def create_backdoor_train_dataset(dataname, train_data, base_label, trigger_label, poison_samples, batch_size, device):
    train_data = PoisonedDataset(train_data, base_label, trigger_label, poison_samples=poison_samples, mode="train",
                                 device=device, dataname=dataname)
    b = Data.TensorDataset(train_data.data, train_data.targets)
    # x = test_data_tri.data_test[0]
    x=torch.tensor(train_data.data[0])
    # print(x)
    x = x.cpu().data
    x = x.clamp(0, 1)
    if args.dataset=="MNIST":
        x = x.view(x.size(0), 1, 28, 28)
    elif args.dataset=="CIFAR10":
        x = x.view(1, 3, 32, 32)
    print(x)
    grid = torchvision.utils.make_grid(x, nrow=1, cmap="gray")
    plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
    plt.show()
    return train_data.data, train_data.targets


"""
                # x=torch.tensor(new_data[i])
                # x_cpu = x.cpu().data
                # x_cpu = x_cpu.clamp(0, 1)
                # x_cpu = x_cpu.view(1, 3, 32, 32)
                # grid = torchvision.utils.make_grid(x_cpu, nrow=1, cmap="gray")
                # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
                # plt.show()
"""

def create_backdoor_test_dataset(dataname, test_data, base_label, trigger_label, poison_samples, batch_size, device):
    test_data_tri = PoisonedDataset(test_data, base_label, trigger_label, poison_samples=poison_samples, mode="test",
                                    device=device, dataname=dataname)
    b = Data.TensorDataset(test_data_tri.data, test_data_tri.targets)
    # x = test_data_tri.data_test[0]
    x=torch.tensor(test_data_tri.data[0])
    # print(x)
    x = x.cpu().data
    x = x.clamp(0, 1)
    if args.dataset=="MNIST":
        x = x.view(x.size(0), 1, 28, 28)
    elif args.dataset=="CIFAR10":
        x = x.view(1, 3, 32, 32)
    grid = torchvision.utils.make_grid(x, nrow=1, cmap="gray")
    plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
    plt.show()
    return b



def create_backdoor_data_loader(dataname, train_data, test_data, trigger_label, posioned_portion, batch_size, device):
    train_data    = PoisonedDataset(train_data, trigger_label, portion=posioned_portion, mode="train", device=device, dataname=dataname)
    test_data_ori = PoisonedDataset(test_data,  trigger_label, portion=0,                mode="test",  device=device, dataname=dataname)
    test_data_tri = PoisonedDataset(test_data,  trigger_label, portion=1,                mode="test",  device=device, dataname=dataname)

    train_data_loader       = DataLoader(dataset=train_data,    batch_size=batch_size, shuffle=True)
    test_data_ori_loader    = DataLoader(dataset=test_data_ori, batch_size=batch_size, shuffle=True)
    test_data_tri_loader    = DataLoader(dataset=test_data_tri, batch_size=batch_size, shuffle=True) # shuffle 随机化

    return train_data #train_data_loader, test_data_ori_loader, test_data_tri_loader

def acc_evaluation(net_glob, dataset_test, args, Zmodel):
    poison_testset = create_backdoor_test_dataset(dataname=args.dataset, test_data=dataset_test,base_label=1, trigger_label=7, posioned_portion=1, batch_size=args.local_bs, device=args.device)
    acc_test, loss_test = testZ_img(net_glob, dataset_test, args, Zmodel)
    poison_acc, poison_loss = testZ_img(net_glob,poison_testset, args,Zmodel)
    print("Testing accuracy: {:.2f}".format(acc_test))
    print("Poison acc", poison_acc)
    return acc_test, poison_acc

def acc_evaluation_org(net_glob, dataset_test, args):
    poison_testset = create_backdoor_test_dataset(dataname=args.dataset, test_data=dataset_test,base_label=1, trigger_label=7, posioned_portion=1, batch_size=args.local_bs, device=args.device)
    acc_test, loss_test = test_img(net_glob, dataset_test, args, 0)
    poison_acc, poison_loss = test_img(net_glob, poison_testset, args, 0)
    print("org_Testing accuracy: {:.2f}".format(acc_test))
    print("Poison acc", poison_acc)
    return acc_test, poison_acc

def testZ_img(net_g, datatest, args, Zmodel):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    Zmodel.eval()
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        # x_batch = data.reshape([len(data), 28 * 28])
        # logits, x_hat, z_mean = Zmodel.forward(data)
        # data = data.reshape((args.bs, 3,32,32))
        logits_z,logits_y2= Zmodel(data, mode='distribution')

        # print(z_mean2)
        log_probs = net_g(logits_z)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        # print(y_pred, target.data.view_as(y_pred))
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


def test_img(net_g, datatest, args, temp_v):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)

    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        target = target-temp_v
        B,c,h,w = data.shape

        if args.dataset=='MNIST':
            data = data.reshape((B, -1))
        # print(z_mean2)
        log_probs  = net_g(data)  # (B, C* h* w), (B, N, 10), (B,N,C, h, w)
        # print(log_probs)
        # print(target)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability

        y_pred = log_probs.data.max(1, keepdim=True)[1]
       # print(log_probs,y_pred)
        # print(y_pred, target.data.view_as(y_pred))
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


def init_cp_model(args, dimZ, dataset):
    k = args.k
    xpl_channels = args.xpl_channels
    explainer_type = args.explainer_type

    if args.dataset == 'MNIST':
        approximator = resnet18(1, 10)
        approximator2 = LinearCIFAR(n_feature=dimZ)
        explainer = resnet18(1, dimZ)
        lr = 0.05
        temp_warmup = 200

    elif dataset == 'CIFAR10':
        approximator = LinearCIFAR(n_feature=dimZ)
        approximator2 =LinearCIFAR(n_feature=dimZ)
        explainer = resnet18(3, dimZ)

        lr = 0.005

    vibi = VIBI(explainer, approximator, approximator2, k=k, num_samples=args.num_samples)
    vibi.to(args.device)
 #   optimizer = torch.optim.Adam(vibi.parameters(), lr=lr)

    return vibi

def get_weights(model):
    for p in model.parameters():
        print("p", p)
    return np.concatenate([p.data.cpu().numpy().ravel() for p in model.parameters()])


def unlearning_net_global(unlearning_temp_net, idxs_local_dict, args, dataset_test, erased_perm,poison_testset):
    unlearning_temp_net.train()
    # org_resnet.train()

    # copy weights
    w_glob_re = unlearning_temp_net.state_dict()

    # training
    acc_test = []
    backdoor_acc_list = []
    poison_acc = []
    idxs_users = range(args.num_users)

    """3. federated unlearning"""

    for iter in range(10):
        idxs_users = range(args.num_users)
        w_locals = []
        grad_locals = []
        #1. unlearning
        for idx in idxs_users:
            if idx not in erased_perm:
                local = idxs_local_dict[idx]
                net_local_w = local.train_Bayes(copy.deepcopy(unlearning_temp_net).to(args.device), idx, args)

                glob_w = unlearning_temp_net.state_dict()
                new_lcoal_grad = unlearning_temp_net.state_dict()
                for k in glob_w.keys():
                    new_lcoal_grad[k] = glob_w[k] - net_local_w[k]

                w_locals.append(copy.deepcopy(net_local_w))
                grad_locals.append(copy.deepcopy(new_lcoal_grad))
            else:
                local = idxs_local_dict[idx]
                net_local_w, loss_item = local.unl_train(copy.deepcopy(unlearning_temp_net).to(args.device), idx, args)

                glob_w = unlearning_temp_net.state_dict()
                new_lcoal_grad = unlearning_temp_net.state_dict()
                for k in glob_w.keys():
                    new_lcoal_grad[k] = glob_w[k] - net_local_w[k]

                w_locals.append(copy.deepcopy(net_local_w))
                grad_locals.append(copy.deepcopy(new_lcoal_grad))

        #dict_usersZ = []

        for idx in erased_perm:
            local  = idxs_local_dict[idx]


            #w_locals_org.append(copy.deepcopy(org_net_glob_w))



        #threshold = 4 #
        #w_server = mcSGD(dataloader_bend, w_locals, args) # signSGD_RLR_grad_pro(grad_locals, glob_w, args)#  mcSGD(dataloader_bend, w_locals, args) #   signSGD_RLR_grad_our(grad_locals, glob_w)#    signSGD_RLR_grad_our(grad_locals, glob_w)## signSGD_RLR_grad(grad_locals, glob_w, threshold, args.lr, iter)#  signSGD_RLR_grad_our(grad_locals, glob_w)##
        w_server = FedAvg(w_locals)
        unlearning_temp_net.load_state_dict(w_server)

        print("epoch: ", iter)
        #valid_acc_old = valid_acc
        valid_acc = test_accuracy(unlearning_temp_net, dataset_test, args, name='vibi valid top1')
        # interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
        print("test_acc", valid_acc)
        acc_test.append(valid_acc)
        backdoor_acc = test_accuracy(unlearning_temp_net, poison_testset, args, name='vibi valid top1')
        backdoor_acc_list.append(backdoor_acc)
        # interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
        print("backdoor_acc", backdoor_acc)
        if backdoor_acc < 0.1:
            break
        # print("epoch: ", iter)
        # acc_temp, poison_acc_temp = acc_evaluation_org(unlearning_temp_net, dataset_test, args)
        # acc_test.append(acc_temp)
        # poison_acc.append(poison_acc_temp)
        # print("org_acc ",acc_test)
        # print("poison_acc_list:", poison_acc)
        #unlearning_temp_net = retrianing_net_global(unlearning_temp_net, idxs_local_dict, args, dataset_test, erased_perm, 1)

    print('backdoor_acc_list', backdoor_acc_list)
    print('test_acc',acc_test)
    #print(acc_test_org)
    unlearning_temp_net.eval()
    return unlearning_temp_net

def FL_train(net_glob, args, dataset_train, dataset_test, dict_users, idxs_local_dict, poison_testset, train_type="train"):
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    acc_test = []
    poison_acc = []


    print("dataset_train", len(dataset_train))
    print("dict_users", len(dict_users))

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]


    """federated learning training"""
    for iter in range(args.epochs):
        # dict_usersZ = []
        idxs_users = range(args.num_users)
        w_locals = []
        grad_locals = []

        for idx in idxs_users:

            local = idxs_local_dict[idx]
            if train_type=='train':
                net_local_w = local.train_Bayes(copy.deepcopy(net_glob).to(args.device), idx, args)

            glob_w = net_glob.state_dict()
            new_lcoal_grad = net_glob.state_dict()
            for k in glob_w.keys():
                new_lcoal_grad[k] = glob_w[k] - net_local_w[k]

            w_locals.append(copy.deepcopy(net_local_w))
            grad_locals.append(copy.deepcopy(new_lcoal_grad))

        w_server = FedAvg(w_locals)
        net_glob.load_state_dict(w_server)

        print("epoch: ", iter)
        #valid_acc_old = valid_acc
        valid_acc = test_accuracy(net_glob, dataset_test, args, name='vibi valid top1')
        # interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
        print("test_acc", valid_acc)
        backdoor_acc = test_accuracy(net_glob, poison_testset, args, name='vibi valid top1')
        # interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
        print("backdoor_acc", backdoor_acc)
        acc_test.append(valid_acc)
        #acc_temp, poison_acc_temp = acc_evaluation_org(net_glob, dataset_test, args)
        # acc_test.append(acc_temp)
        # poison_acc.append(poison_acc_temp)
        # print("org_acc ", acc_test)
        # print("poison_acc_list:", poison_acc)

    print("global train acc",acc_test)
    net_glob.eval()
    return net_glob


def FL_retrain(net_glob, net_org, net_unl, args, dataset_train, dataset_test, dict_users, idxs_local_dict,poison_testset, retrain_epoch, train_type="retrain"):
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    acc_test = []
    poison_acc = []
    backdoor_acc_list = []

    print("dataset_train", len(dataset_train))
    print("dict_users", len(dict_users))

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]


    """federated learning training"""
    for iter in range(retrain_epoch):
        # dict_usersZ = []
        idxs_users = range(args.num_users)
        w_locals = []
        grad_locals = []

        for idx in idxs_users:

            local = idxs_local_dict[idx]
            if train_type=='retrain':
                net_local_w, KL_fr, KL_nipsr = local.retrain_Bayes(copy.deepcopy(net_glob).to(args.device), net_org, net_unl, idx, args)

            glob_w = net_glob.state_dict()
            new_lcoal_grad = net_glob.state_dict()
            for k in glob_w.keys():
                new_lcoal_grad[k] = glob_w[k] - net_local_w[k]

            w_locals.append(copy.deepcopy(net_local_w))
            grad_locals.append(copy.deepcopy(new_lcoal_grad))

        w_server = FedAvg(w_locals)
        net_glob.load_state_dict(w_server)

        print("epoch: ", iter)
        #valid_acc_old = valid_acc
        valid_acc = test_accuracy(net_glob, dataset_test, args, name='vibi valid top1')
        # interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
        print("test_acc", valid_acc)
        acc_test.append(valid_acc)
        backdoor_acc = test_accuracy(net_glob, poison_testset, args, name='vibi valid top1')
        # interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
        backdoor_acc_list.append(backdoor_acc)
        print("backdoor_acc", backdoor_acc_list)
        # acc_temp, poison_acc_temp = acc_evaluation_org(net_glob, dataset_test, args)
        # acc_test.append(acc_temp)
        # poison_acc.append(poison_acc_temp)
        # print("org_acc ", acc_test)
        # print("poison_acc_list:", poison_acc)

    print("global_retrain_acc",acc_test)
    net_glob.eval()
    return net_glob, KL_fr, KL_nipsr

def FL_unlearn_train(net_glob, net_temp, args, dataset_train, dataset_test, dict_users, idxs_local_dict, erased_perm,poison_testset, train_type="train"):
    net_glob.train()
    w_glob = net_glob.state_dict()
    acc_test = []
    backdoor_acc_list = []
    print("dataset_train", len(dataset_train))
    print("dict_users", len(dict_users))

    # if args.all_clients:
    #     print("Aggregation over all clients")
    #     w_locals = [w_glob for i in range(args.num_users)]


    """federated unlearning training"""
    for iter in range(10):
        # dict_usersZ = []
        idxs_users = range(args.num_users)
        w_locals = []
        grad_locals = []

        """unlearning training should different from normal training,
        We should first unlearn, then use the unlearned model retrain"""
        #1. unlearning
        for idx in idxs_users:
            if idx not in erased_perm:
                local = idxs_local_dict[idx]
                net_local_w = local.train_Bayes(copy.deepcopy(net_glob).to(args.device), idx,
                                                args)  # local.unlearn_Bayes(copy.deepcopy(net_glob).to(args.device), net_temp, idx, args, train_type)

                glob_w = net_glob.state_dict()
                new_lcoal_grad = net_glob.state_dict()
                for k in glob_w.keys():
                    new_lcoal_grad[k] = glob_w[k] - net_local_w[k]

                w_locals.append(copy.deepcopy(net_local_w))
                grad_locals.append(copy.deepcopy(new_lcoal_grad))
            else:
                local = idxs_local_dict[idx]
                net, net_local_w = local.unlearn_Bayes(copy.deepcopy(net_glob).to(args.device), net_temp, idx, args,
                                                       train_type)

                glob_w = net_glob.state_dict()
                new_lcoal_grad = net_glob.state_dict()
                for k in glob_w.keys():
                    new_lcoal_grad[k] = glob_w[k] - net_local_w[k]

                w_locals.append(copy.deepcopy(net_local_w))
                grad_locals.append(copy.deepcopy(new_lcoal_grad))
        # w_server = FedAvg(w_locals)
        # net_glob.load_state_dict(w_server)
        #
        # # 2. train others using unlearn
        # w_locals = []

        w_server = FedAvg(w_locals)
        net_glob.load_state_dict(w_server)

        print("epoch: ", iter)
        # valid_acc_old = valid_acc
        valid_acc = test_accuracy(net_glob, dataset_test, args, name='vibi valid top1')
        # interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
        print("test_acc", valid_acc)
        acc_test.append(valid_acc)
        backdoor_acc = test_accuracy(net_glob, poison_testset, args, name='vibi valid top1')
        backdoor_acc_list.append(backdoor_acc)
        print("backdoor_acc", backdoor_acc_list)
        print("global_unl_acc: ", acc_test)
        if backdoor_acc < 0.1:
            break
    print("epoch: ", iter)
    # valid_acc_old = valid_acc
    valid_acc = test_accuracy(net_glob, dataset_test, args, name='vibi valid top1')
    # interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
    print("test_acc", valid_acc)
    acc_test.append(valid_acc)

    backdoor_acc = test_accuracy(net_glob, poison_testset, args, name='vibi valid top1')
    # interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
    backdoor_acc_list.append(backdoor_acc)
    print("backdoor_acc", backdoor_acc_list)
    print("global_unl_acc: ", acc_test)
    net_glob.eval()
    return net_glob

def clean_FL(args):

    print("device", args.device)
    dataset = args.dataset
    erased_perm = np.random.permutation(args.num_users)[0: int(args.num_users * args.erased_portion)]

    poison_perm = erased_perm #np.random.permutation(args.num_users)[0: int(args.num_users * args.poison_portion)]

    #should be the same
    print("poison_perm",poison_perm)
    print("erased_perm",erased_perm)
    # load dataset and split users
    if args.dataset == 'MNIST':
        trans_mnist = transforms.Compose([transforms.ToTensor(), ])  # transforms.Normalize((0.1307,), (0.3081,))
        dataset_train = datasets.MNIST('../../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../../data/mnist/', train=False, download=True, transform=trans_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'CIFAR10':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), ])  # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        dataset_train = datasets.CIFAR10('../../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')

    print('img_size', len(dataset_train))
    print('dataset', args.dataset)

    length = len(dataset_train)
    bend_size, remain_size = 2000, length-2000
    bend_set, remain_set = torch.utils.data.random_split(dataset_train, [bend_size, remain_size])
    # dataloader_bend = DataLoader(bend_set, batch_size=args.local_bs, shuffle=True)
    poison_samples = int(length / args.num_users) * args.erased_local_r
    poison_data, poison_targets = create_backdoor_train_dataset(dataname=args.dataset, train_data=dataset_train,
                                                                base_label=1,
                                                                trigger_label=2, poison_samples=poison_samples,
                                                                batch_size=args.local_bs, device=args.device)

    poison_testset = create_backdoor_test_dataset(dataname=args.dataset, test_data=dataset_test, base_label=1,
                                                  trigger_label=2, poison_samples=10000, batch_size=args.local_bs,
                                                  device=args.device)

    poison_data, poison_targets = poison_data.to(args.device), poison_targets.to(args.device)
    # build model
    if args.model == 'cnn' and args.dataset == 'CIFAR10':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'MNIST':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'z_linear' and args.dataset == 'MNIST':
        net_glob, lr = init_vibi(args.dataset)
        retrian_net, lr = init_vibi(args.dataset)
        #net_glob = Linear(n_feature= 28 * 28, n_output=10).to(args.device) #LinearCIFAR(n_feature=dimZ).to(args.device)
        #retraining_net = Linear(n_feature= 28 * 28, n_output=10).to(args.device)
    elif args.model == 'z_linear' and args.dataset == 'CIFAR10':
        #net_glob = models.resnet18().to(args.device)
        net_glob, lr = init_vibi(args.dataset)
        retrian_net, lr = init_vibi(args.dataset)
        #dimZ=7*7
        #net_glob = LinearCIFAR(n_feature=dimZ).to(args.device)
        #org_net_glob = LinearCIFAR(n_feature= 3 * 32 * 32, n_output=10).to(args.device) #init_cp_model(args, dimZ) #
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    glob_w = net_glob.state_dict()
    diff_grad = net_glob.state_dict()
    retraining_net_w = retrian_net.state_dict()
    distance = 0
    for k in glob_w.keys():
        diff_grad[k] = glob_w[k] - retraining_net_w[k]
        distance += torch.norm(diff_grad[k].float(), p=2)
    print("distance",distance)

    # retraining_net.train()
    # org_resnet.train()
    idxs_users = range(args.num_users)
    """create a idxs_local_list"""
    idxs_local_dict = {}
    for idx in idxs_users:
        if idx in poison_perm:
            need_poison = True
        else:
            need_poison = False
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], need_poison=need_poison, poison_data=poison_data,
                            poison_targets=poison_targets, erased_perm=erased_perm, idx=idx)
        idxs_local_dict[idx] = local

    print()
    print("start train")
    net_glob = FL_train(net_glob, args, dataset_train, dataset_test, dict_users, idxs_local_dict, poison_testset, train_type='train')

    print()
    print("start unlearn")
    unlearn_nips, lr = init_vibi(args.dataset)
    unlearn_nips.to(args.device)
    unlearn_nips.load_state_dict(net_glob.state_dict())

    unlearn_nips = unlearning_net_global(unlearn_nips, idxs_local_dict, args, dataset_test, erased_perm,poison_testset)


    # print("start unlearn parameter self-sharing")
    # unlearn_self, lr = init_vibi(args.dataset)
    # unlearn_self.to(args.device)
    # unlearn_self.load_state_dict(net_glob.state_dict())
    #
    # unlearn_self = FL_unlearn_train(unlearn_self, net_glob, args, dataset_train, dataset_test, dict_users, idxs_local_dict, erased_perm,poison_testset, train_type='self-sharing')


    #print("FL unl retrain one round")

    #unlearn_nips, KL_fr, KL_nipsr = FL_retrain(unlearn_nips, net_glob, copy.deepcopy(unlearn_nips), args, dataset_train, dataset_test, dict_users, idxs_local_dict, retrain_epoch=1, train_type='retrain')

    # print("KL_fr", KL_fr)
    # print("KL_nipsr", KL_nipsr)

    print("start retrain")
    retrian_net, KL_fr, KL_nipsr = FL_retrain(retrian_net, net_glob, unlearn_nips, args, dataset_train, dataset_test, dict_users, idxs_local_dict, poison_testset, retrain_epoch= args.epochs, train_type='retrain')

    # print("start retrain2")
    # retrian_net2, lr = init_vibi(args.dataset)
    # retrian_net2.to(args.device)
    # retrian_net2, KL_fr, KL_self_r = FL_retrain(retrian_net2, net_glob, unlearn_self, args, dataset_train, dataset_test, dict_users, idxs_local_dict,poison_testset, retrain_epoch= args.epochs, train_type='retrain')


    glob_w = net_glob.state_dict()
    diff_grad = net_glob.state_dict()
    for k in diff_grad.keys():
        diff_grad[k] = diff_grad[k] - diff_grad[k]
    retrain_net_w = retrian_net.state_dict()
    distance = 0
    for k in glob_w.keys():
        diff_grad[k] = retrain_net_w[k] - glob_w[k]
        distance += torch.norm(diff_grad[k], p=2)
        print("retrain-learning_distance", distance)

    diff_grad = unlearn_nips.state_dict()
    for k in diff_grad.keys():
        diff_grad[k] = diff_grad[k] - diff_grad[k]
    unlearning_net_w = unlearn_nips.state_dict()
    distance = 0
    for k in glob_w.keys():
        diff_grad[k] = retrain_net_w[k] - unlearning_net_w[k]
        distance += torch.norm(diff_grad[k], p=2)
        print("retrain_unlearning_distance", distance)

    # diff_grad = unlearn_nips.state_dict()
    # for k in diff_grad.keys():
    #     diff_grad[k] = diff_grad[k] - diff_grad[k]
    # unlearning_net_self = unlearn_self.state_dict()
    # retrain_net_w = retrian_net2.state_dict()
    # distance = 0
    # for k in glob_w.keys():
    #     diff_grad[k] = retrain_net_w[k] - unlearning_net_self[k]
    #     distance += torch.norm(diff_grad[k], p=2)
    #     print("retrain_unlearning_self_distance", distance)

    print("KL_fr", np.mean(KL_fr), KL_fr)
    print("KL_nipsr", np.mean(KL_nipsr), KL_nipsr)
    # print("KL_self_r", np.mean(KL_self_r), KL_self_r)












if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.gpu = 0
    args.num_users = 10
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.iid = True
    args.model = 'z_linear'
    args.local_bs = 100
    args.local_ep = 10
    args.num_epochs = 1
    args.dataset = 'MNIST'
    args.xpl_channels = 1
    args.epochs = int(10)
    args.add_noise = False
    args.beta = 0.001
    args.lr = 0.001
    args.erased_size = 1500 #120
    args.poison_portion = 0.0
    args.erased_portion = 0.5
    args.erased_local_r = 0.1
    ## in unlearning, we should make the unlearned model first be backdoored and then forget the trigger effect
    args.unlearn_learning_rate = 1.5
    args.self_sharing_rate = 1.5
    args.unl_conver_r=2
    args.hessian_rate=10
    print('args.beta',args.beta, 'args.lr', args.lr)
    print('args.erased_portion', args.erased_portion, 'args.erased_local_r',args.erased_local_r)
    print('args.hessian_rate',args.hessian_rate, 'args.self_sharing_rate',args.self_sharing_rate)
    clean_FL(args)




