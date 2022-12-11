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
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# from VIBImodels import ResNet, resnet18, resnet34, Unet

# from debug import debug
import torch.nn as nn
import torch.optim

import torch.nn.functional as F



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
        mu = self._enc_mu(h_enc).cuda()
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma).cuda()
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().cuda()

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


class PoisonedDataset(Dataset):

    def __init__(self, dataset, base_label, trigger_label, portion, mode="train", device=torch.device("cuda"), dataname="mnist"):
        # self.class_num = len(dataset.classes)
        # self.classes = dataset.classes
        # self.class_to_idx = dataset.class_to_idx
        self.device = device
        self.dataname = dataname
        self.ori_dataset = dataset
        self.data, self.targets = self.add_trigger(self.reshape(dataset.data, dataname), dataset.targets, base_label, trigger_label, portion, mode)
        self.channels, self.width, self.height = self.__shape_info__()
        #self.data_test, self.targets_test = self.add_trigger_test(self.reshape(dataset.data, dataname), dataset.targets, base_label, trigger_label, portion, mode)

    def __getitem__(self, item):
        img = self.data[item]
        label_idx = self.targets[item]

        label = np.zeros(10)
        label[label_idx] = 1 # 把num型的label变成10维列表。
        label = torch.Tensor(label)

        img = img.to(self.device)
        label = label.to(self.device)

        return img, label

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
        return self.data.shape[1:]

    def reshape(self, data, dataname="MNIST"):
        if dataname == "MNIST":
            new_data = data.reshape(len(data),1,28,28)
        elif dataname == "CIFAR10":
            new_data = data.reshape(len(data),3,32,32)
        return np.array(new_data.to("cpu"))

    def norm(self, data):
        offset = np.mean(data, 0)
        scale  = np.std(data, 0).clip(min=1)
        return (data- offset) / scale

    def add_trigger(self, data, targets, base_label, trigger_label, portion, mode):
        print("## generate——test " + mode + " Bad Imgs")
        new_data = copy.deepcopy(data)
        new_targets = []
        new_data_re = []

        #total_poison_num = int(len(new_data) * portion/10)
        _, width, height = data.shape[1:]
        for i in range(len(data)):
            if targets[i]==base_label:
                new_targets.append(trigger_label)
                new_data[i, :, width - 23, height - 20] = 90
                # new_data[i, :, width - 23, height - 21] = 254
                # new_data[i, :, width - 23, height - 22] = 254
                # new_data[i, :, width - 22, height - 21] = 254
                # new_data[i, :, width - 24, height - 21] = 254

                new_data_re.append(new_data[i])
                # x = new_data[i]
                # x=torch.tensor(x)
                # x = x.cpu().data
                # x = x.clamp(0, 1)
                # x = x.view(x.size(0), 1, 28, 28)
                # grid = torchvision.utils.make_grid(x, nrow=1, cmap="gray")
                # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
                # plt.show()
                #total_poison_num-=1
                #if total_poison_num==0:break
       # print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(perm), len(new_data)-len(perm), portion))
        return torch.Tensor(new_data_re), torch.Tensor(new_targets).long()


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=50, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--sampling', type=float, default=1.0, help="random sampling (default: 1.0)")
    parser.add_argument('--epsilon', type=float, default=1.0, help="DP epsilon (default: 1.0)")
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
    parser.add_argument('--cuda', action='store_true', default=True)
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
def test_accuracy(model, data_loader, FL_model, net_global, org_net_glob, device, name='test'):
    num_total = 0
    num_correct = 0
    model.eval()
    FL_model.eval()
    num_correct2 = 0
    num_correct3 = 0
    num_correct_global = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits_z, outy_2 = model(x, mode='distribution')
        B, C, H, W = x.shape
        # pz2 = logits_z.reshape((B, 7,7,7))
        out2 = FL_model(logits_z.detach())
        out_res = net_global(logits_z.detach())
        B,C,H,W = x.shape
        x_for_fl_org = x.reshape((B, -1))
        out3  = org_net_glob(x_for_fl_org)
        if y.ndim == 2:
            y = y.argmax(dim=1)
        num_correct += (outy_2.argmax(dim=1) == y).sum().item()
        num_correct2 +=(out2.argmax(dim=1)==y).sum().item()
        num_correct3 +=(out3.argmax(dim=1)==y).sum().item()
        num_correct_global += (outy_2.argmax(dim=1) == y).sum().item()
        num_total += len(x)
    acc = num_correct / num_total
    acc2 = num_correct2/num_total
    acc3 = num_correct3 / num_total
    acc_global = num_correct_global/num_total
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

    # 设置隐藏层到输出层的函数

    def forward(self, x):
        # 定义向前传播函数
        hid0 = F.relu(self.fc0(x))
        hid1 = F.relu(self.fc1(hid0))
        hid2 = F.relu(self.fc2(hid1))
        # 给x加权成为a，用激励函数将a变成特征b
        hid3 = F.softmax(self.fc3(hid2))
        return hid3

class VIBI(nn.Module):
    def __init__(self, explainer, approximator, approximator2, k=4, num_samples=4, temp=1):
        super().__init__()

        self.explainer = explainer
        self.approximator = approximator
        self.approximator2 = approximator2
        self.k = k
        self.temp = temp
        self.num_samples = num_samples

        self.warmup = False

    def explain(self, x, mode='topk', num_samples=None):
        """Returns the relevance scores
        """

        k = self.k
        temp = self.temp
        N = num_samples or self.num_samples

        B, C, H, W = x.shape

        logits_z = self.explainer(x)  # (B, C, h, w)

        #         print("B, C, H, W",B, C, H, W)
        #logits_z = logits_z.reshape((B, -1))  # (B, C* h* w)

        if mode == 'distribution':  # return the distribution over explanation
            #p_z = F.softmax(logits_z, dim=1).reshape((B, C, h, w))  # (B, C, h, w)
            #p_z_upsampled = F.interpolate(p_z, (H, W), mode='nearest')  # (B, C, H, W)
            return logits_z
        elif mode == 'warmup':
            return logits_z

    def forward(self, x, mode='topk'):

        #         print("B, C, H, W", B, C, H, W)
        if mode == 'distribution':
            logits_z_flat = self.explain(x, mode=mode)  # (B, C, H, W), (B, C* h* w)
            #B, n = logits_z_flat.shape
            #logits_z_for_predict = logits_z_flat.reshape((B, 7, 7, 7))
            logits_y2 = self.approximator2(logits_z_flat)
            return logits_z_flat, logits_y2
        elif mode == 'test':
            z_upsampled = self.explain(x, mode=mode)  # (B, C, H, W)
            logits_y = self.approximator(z_upsampled)
            return logits_y



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
    def __init__(self, dataset, idxs, sampling):
        self.index=0
        self.dataset = dataset
        #         self.idxs = list(idxs)
        #         self.idxs = random.sample(list(idxs), int(len(idxs)*sampling))
        if sampling == 1:
            self.idxs = list(idxs)
        else:
            self.idxs = np.random.choice(list(idxs), size=int(len(idxs) * sampling), replace=True)
            #self.idxs = random.sample(list(idxs), int(len(idxs) * sampling)) # without replacement
            # random.choice is with replacement
        # print('datasplite' , idxs, len(dataset))

        self.data, self.targets = self.get_image_label()

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        print("item", item, self.index, self.idxs[item],label)
        self.index+=1
        #print("self.idxs", self.idxs)
        return image, label

    def get_image_label(self, ):
        temp_img = torch.empty(0, 28,28).float().cuda()
        temp_label = torch.empty(0).long().cuda()
        for id in self.idxs:
            image, label = self.dataset[id]
            image, label = image.cuda(), torch.tensor([label]).long().cuda()
            #print(label)
            #label = torch.tensor([label])
            temp_img = torch.cat([temp_img, image], dim=0)
            temp_label = torch.cat([temp_label, label], dim=0)
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
        return data.add(torch.from_numpy(noise_tesnor).float().cuda())

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, need_poison=None, poison_portion=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        # prepare the local original dataset
        self.train_splite = DatasetSplit(dataset, idxs, args.sampling)
        #print(self.train_splite.dataset)
        # self.train_set = self.train_splite.get_image_label()
        if need_poison:
            data, targets = create_backdoor_train_dataset(dataname="MNIST", train_data=dataset, base_label=1, trigger_label=7, posioned_portion=poison_portion, batch_size=self.args.local_bs, device=self.args.device)
            data, targets = data.cuda(), targets.cuda()
            data_reshape = self.train_splite.data.reshape(len(self.train_splite.data), 1, 28, 28)
            data = torch.cat([data, data_reshape], dim=0)
            targets = torch.cat([targets, self.train_splite.targets], dim=0)
            self.poison_trainset = Data.TensorDataset(data, targets)
        else:
            data_reshape = self.train_splite.data.reshape(len(self.train_splite.data),1,28,28)
            self.poison_trainset = Data.TensorDataset(data_reshape, self.train_splite.targets)

        self.ldr_train = DataLoader(self.poison_trainset, batch_size=self.args.local_bs, shuffle=True)
        # local data compressing

    def add_noise(data, epsilon, sensitivity, args):
        noise_tesnor = np.random.laplace(1, sensitivity / epsilon, data.shape)
        # data = torch.add(data, torch.from_numpy(noise_tesnor))
        # for x in np.nditer(np_data, op_flags=['readwrite']):
        #     x[...] = x + np.random.laplace(1, sensitivity/epsilon,)
        if args.gpu==-1:
            return data.add(torch.from_numpy(noise_tesnor).float())
        else:
            return data.add(torch.from_numpy(noise_tesnor).float().cuda())

    def train(self, net, idx):
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            # print(iter)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                B,c,h,w = images.shape
                #print(B,h,w)
                images = images.reshape((B, -1))
                # print(labels)
                # print('batch_idx', batch_idx, images.shape)
                #if self.args.add_noise:
                #   images =LocalUpdate.add_noise( data=images, epsilon=self.args.epsilon, sensitivity=1,args=self.args)
                net.zero_grad()
                log_probs = net(images)
                #print("log_probs",log_probs)
                #break
                loss = self.loss_func(log_probs, labels)
                # loss = F.nll_loss(log_probs, labels)
                loss.backward()
                optimizer.step()
                fl_acc = (log_probs.argmax(dim=1) == labels).float().mean().item()
                # print('batch_idx', batch_idx)
                if batch_idx % 1000 == 0 and iter==0:
                    print("fl_acc", fl_acc, "loss", loss.item(), "idx", idx)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def trainZ(self, net, idx, iter_glob, beta):
        net.train()
        # train and update
        # optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
        epoch_loss = []
        for epoch in range(self.args.local_ep):
            batch_loss = []
            # print(iter)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                net.zero_grad()
                logits_z, logits_y2  = net(images,
                                                                      mode='distribution')  # (B, C* h* w), (B, N, 10), (B,N,C, h, w)


                hid0_log = logits_z.log_softmax(dim=1)

                #H_p_q = self.loss_func(logits_y, labels)

                KL_z_r = (torch.exp(hid0_log) * hid0_log).sum(dim=1).mean() + math.log(hid0_log.shape[1])


                global_loss = self.loss_func(logits_y2, labels)
                loss = beta * KL_z_r + global_loss  #+ 2 / (hid2_similarity)

                loss.backward()
                optimizer.step()

                # print('batch_idx', batch_idx)
                fl_acc = (logits_y2.argmax(dim=1) == labels).float().mean().item()
                # print('idx=idx', idx, images)
                if batch_idx % 100 == 1:
                    print('fl_acc:', fl_acc, 'loss:', loss.item(),'user idx:', idx, 'glob_iter:', iter_glob)

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def train_resnet18(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            # print('iter', iter)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # print(batch_idx)
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # print('batch_idx', batch_idx, images.shape)
                if self.args.add_noise:
                    images =LocalUpdate.add_noise( data=images, epsilon=0.8, sensitivity=1, args=self.args)
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                optimizer.zero_grad()
                # loss = F.nll_loss(log_probs, labels)
                loss.backward()
                optimizer.step()
                # print('batch_idx', batch_idx)
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

def train_model_vae(dataloader, data_len):
    input_dim=10
    encoder = Encoder(input_dim, 100, 100).cuda()
    decoder = Decoder(8, 100, input_dim).cuda()
    vae = VAE(encoder, decoder).cuda()

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(vae.parameters(), lr=0.005)
    l = None
    temp_fc0 = torch.empty(10).float().cuda()
    for epoch in range(10):
        for i, data in enumerate(dataloader, 0):
            inputs, classes = data
            inputs, classes = inputs.cuda(), classes.cuda()
            optimizer.zero_grad()
            dec = vae(inputs)
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(dec, inputs) +  ll
            loss.backward()
            optimizer.step()
            # print(dec.shape)
            if epoch==9:
                dec = dec.reshape((10))
                temp_fc0 = temp_fc0+ dec
                print(dec)
         #   print(loss)
            l = loss.item()
        print(epoch, l)
    return torch.div(temp_fc0, data_len)

def FedAvg_grad(grad_locals, glob_w):
    grad_avg = copy.deepcopy(grad_locals[0])
    data_len = 0
    for i in range(len(grad_locals)):
        data_len += 1  # + np.random.laplace(0, beta, 1)[0]
    print('len_w', len(grad_locals) , data_len)
    temp_fc0 = torch.empty(0, 10).float().cuda()
    for k in grad_avg.keys():
        print(k, grad_avg[k].shape)
        for i in range(1, len(grad_locals)):
            temp = torch.add(grad_locals[i][k], 0) # np.random.laplace(0, beta, 1)[0] / 100
            # w_avg[k] += w[i][k]
            grad_avg[k] += temp
        grad_avg[k] = torch.div(grad_avg[k], data_len)

    w_avg = copy.deepcopy(glob_w)
    for k in grad_avg.keys():
        w_avg[k] = glob_w[k] - grad_avg[k]
    return w_avg


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    epsilon = 1
    beta = 1 / epsilon
    data_len = 0
    for i in range(len(w)):
        data_len += 1  # + np.random.laplace(0, beta, 1)[0]
    print('len_w', len(w) , data_len)
    temp_fc0 = torch.empty(0, 10).float().cuda()
    for k in w_avg.keys():
        print(k, w_avg[k].shape)
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


def AvgCP(dict_usersZmodel):

    w = []
    for i in range(len(dict_usersZmodel)):
        w.append(copy.deepcopy(dict_usersZmodel[i].state_dict()))

    w_avg = copy.deepcopy(w[0])
    epsilon = 1
    beta = 1 / epsilon
    data_len = 0
    for i in range(len(w)):
        data_len += 1  # + np.random.laplace(0, beta, 1)[0]
    print('len_w', len(w) , data_len)
    for k in w_avg.keys():
        for i in range(1, len(w)):
            # print('before',w[i][k][1])
            temp = torch.add(w[i][k], 0) # np.random.laplace(0, beta, 1)[0] / 100
            # w_avg[k] += w[i][k]
            # print(temp)
            w_avg[k] += temp
            # print('after', temp[1])

        # w_avg[k] = torch.div(w_avg[k], len(w))
        w_avg[k] = torch.div(w_avg[k], data_len)

    # print(dict_usersZmodel[0])
    return w_avg


def vibi_train(args, ldr_train, z_dim, user_idx, vibi, optimizer, FL_model, net_glob, org_net_glob, user0_zmodel):
    device = 'cuda' if args.cuda else 'cpu'
    # print("device", device)
    dataset = args.dataset
    beta = args.beta

    init_epoch = 0

    logs = defaultdict(list)
    valid_acc = 0.8

    if dataset == 'MNIST':
        lr = 0.05
    elif dataset == 'CIFAR10':
        lr =0.005

    FL_optimizer_original = torch.optim.Adam(org_net_glob.parameters(), lr=lr)
    fl_org_loss_fn = nn.CrossEntropyLoss()


    temp_img = torch.empty(0, z_dim).float().cuda()
    temp_label = torch.empty(0).long().cuda()

    global_fn = nn.CrossEntropyLoss()
    #training
    print('Training VIBI')
    print(f'explainer params:\t{num_params(vibi.explainer) / 1000:.2f} K')
    print(f'{type(vibi.approximator).__name__:>10} approximator params:\t{num_params(vibi.approximator) / 1000:.2f} K')

    # inspect_explanations()
    reconstruction_function = nn.MSELoss(size_average=False)

    for epoch in range(init_epoch, args.num_epochs):
        vibi.train()
        step_start = epoch * len(ldr_train)
        for step, (x, y) in enumerate(ldr_train, start=step_start):
            # print(user_idx, x.shape)
            # warmup = t < 1
            x, y = x.to(device), y.to(device)  # (B, C, H, W), (B, 10)
            # if user_idx == 0:
            #     y = y+1
            #     y = y%10
            logits_z, logits_y2 = vibi(x,  mode='distribution')  # (B, C* h* w), (B, N, 10), (B,N,C, h, w)

            logits_z_temp, logits_y2_temp = user0_zmodel(x, mode='distribution')
            B, n = logits_z.shape
            #logits_z_for_pre = logits_z.reshape((B,7 ,7,7))
            fl_y = FL_model(logits_z)

            hid2_similarity = torch.sum(torch.cosine_similarity(logits_z, logits_z_temp, dim=1, eps=1e-8))
            hid0_log = logits_z.log_softmax(dim=1)

            KL_z_r = (torch.exp(hid0_log) * hid0_log).sum(dim=1).mean() + math.log(hid0_log.shape[1])


            global_loss = global_fn(logits_y2, y)
            loss =  beta * KL_z_r + global_loss #+ (100-hid2_similarity)*(100-hid2_similarity)*0.01

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch == args.num_epochs - 1:
                temp_img = torch.cat([temp_img, logits_z], dim=0)
                temp_label = torch.cat([temp_label, y], dim=0)

            B, C, H, W = x.shape
            x_for_fl_org = x.reshape((B, -1))

            fl_org_y = org_net_glob(x_for_fl_org)
            fl_org_loss = fl_org_loss_fn(fl_org_y, y)
            FL_optimizer_original.zero_grad()
            fl_org_loss.backward()
            FL_optimizer_original.step()


            #acc = (logits_y.argmax(dim=1) == y).float().mean().item()
            # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()
            fl_acc = (fl_y.argmax(dim=1) == y).float().mean().item()
            fl_org_acc = (fl_org_y.argmax(dim=1) == y).float().mean().item()
            # fl_res_acc = (fl_res_Y.argmax(dim=1) == y).float().mean().item()
            global_acc = (logits_y2.argmax(dim=1) == y).float().mean().item()
            metrics = {
                'user_idx': user_idx,
                # 'fl_org_res': fl_res_acc,
                'fl_org_acc': fl_org_acc,
                'global_y': global_acc,
                'fl_acc': fl_acc,
                #'acc': acc,
                'loss': loss.item(),
                'temp': vibi.temp,
                #'H(p,q)': H_p_q.item(),
                # '1-JS(p,q)': JS_p_q,
                'KL(z||r)': KL_z_r.item(),
                #'I_ZX_bound': I_ZX_bound,
                'hid2_similarity': hid2_similarity.item(),
            }

            for m, v in metrics.items():
                logs[m].append(v)

            if step % len(ldr_train) % 50 == 2:
                print(f'[{epoch}/{init_epoch + args.num_epochs}:{step % len(ldr_train):3d}] '
                      + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
        vibi.eval()
        valid_acc_old = valid_acc
        valid_acc = test_accuracy(vibi, ldr_train, FL_model, net_glob, org_net_glob, device, name='vibi valid top1')
        interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(ldr_train)).tolist()
        logs['val_acc'].extend(interpolate_valid_acc)
        # print("test_acc", valid_acc)
    b = Data.TensorDataset(temp_img, temp_label)
    lr_train_dl = DataLoader(b, batch_size=args.local_bs, shuffle=True)

    return lr_train_dl, vibi, vibi.approximator2.state_dict(), org_net_glob.state_dict(), org_net_glob #, org_resnet.state_dict()

def create_backdoor_train_dataset(dataname, train_data, base_label, trigger_label, posioned_portion, batch_size, device):
    train_data = PoisonedDataset(train_data, base_label, trigger_label, portion=posioned_portion, mode="train", device=device, dataname=dataname)
    #b = Data.TensorDataset(train_data.data, train_data.targets)
    return train_data.data, train_data.targets

def create_backdoor_test_dataset(dataname, test_data, base_label, trigger_label, posioned_portion, batch_size, device):
    test_data_tri = PoisonedDataset(test_data, base_label,  trigger_label, portion=1,                mode="test",  device=device, dataname=dataname)
    b = Data.TensorDataset(test_data_tri.data, test_data_tri.targets)
    # x = test_data_tri.data_test[0]
    # x=torch.tensor(x)
    # # print(x)
    # x = x.cpu().data
    # x = x.clamp(0, 1)
    # x = x.view(x.size(0), 1, 28, 28)
    # grid = torchvision.utils.make_grid(x, nrow=1, cmap="gray")
    # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
    # plt.show()
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
    poison_testset = create_backdoor_test_dataset(dataname="MNIST", test_data=dataset_test,base_label=1, trigger_label=7, posioned_portion=1, batch_size=args.local_bs, device=args.device)
    acc_test, loss_test = testZ_img(net_glob, dataset_test, args, Zmodel)
    poison_acc, poison_loss = testZ_img(net_glob,poison_testset, args,Zmodel)
    print("Testing accuracy: {:.2f}".format(acc_test))
    print("Poison acc", poison_acc)
    return acc_test, poison_acc

def acc_evaluation_org(net_glob, dataset_test, args):
    poison_testset = create_backdoor_test_dataset(dataname="MNIST", test_data=dataset_test,base_label=1, trigger_label=7, posioned_portion=1, batch_size=args.local_bs, device=args.device)
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
        data, target = data.cuda(), target.cuda()
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


def test_img(net_g, datatest, args,temp_v):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)

    for idx, (data, target) in enumerate(data_loader):
        data, target = data.cuda(), target.cuda()
        target = target-temp_v
        B,c,h,w = data.shape
        data2 = data.reshape((B, -1))
        # print(z_mean2)
        log_probs = net_g(data2)  # (B, C* h* w), (B, N, 10), (B,N,C, h, w)
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


def init_cp_model(args, dimZ):
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
    vibi.to(device)
 #   optimizer = torch.optim.Adam(vibi.parameters(), lr=lr)

    return vibi


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.gpu = 0
    args.num_users = 10
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.iid = True
    args.model = 'z_linear'
    args.local_bs = 100
    args.local_ep = 2
    args.num_epochs = 1
    args.dataset = 'MNIST'
    args.xpl_channels = 1
    args.epochs = int(50)
    args.add_noise = False
    args.beta=0.01
    poison_portion = 0.3
    dimZ = 7*7 #49
    device = 'cuda' if args.cuda else 'cpu'
    print("device", device)
    dataset = args.dataset
    poison_perm = np.random.permutation(args.num_users)[0: int(args.num_users * poison_portion)]
    print("poison_perm",poison_perm)
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
    img_size = dataset_train[0][0].shape
    print('img_size', len(dataset_train))
    print('data set', args.dataset)

    #init need poison
    need_poison=False
    # build model
    if args.model == 'cnn' and args.dataset == 'CIFAR10':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'MNIST':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'z_linear' and args.dataset == 'MNIST':
        net_glob = LinearCIFAR(n_feature= 28 * 28, n_output=10).to(args.device) #LinearCIFAR(n_feature=dimZ).to(args.device)
        org_net_glob = LinearCIFAR(n_feature= 28 * 28, n_output=10).to(args.device)
    elif args.model == 'z_linear' and args.dataset == 'CIFAR10':
        net_glob = LinearCIFAR(n_feature=dimZ).to(args.device)
        org_net_glob = LinearCIFAR(n_feature= 3 * 32 * 32, n_output=10).to(args.device) #init_cp_model(args, dimZ) #
        # org_resnet = resnet18(3, 10).to(device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()
    #org_net_glob.train()
    # org_resnet.train()

    # copy weights
    w_glob = net_glob.state_dict()
    #org_w_glob = org_net_glob.state_dict()
    # w_glob_org_res = org_resnet.state_dict()

    # training
    loss_train = []
    acc_test = []
    poison_acc = []
    acc_test_org = []

    print("dataset_train", len(dataset_train))
    print("dict_users", len(dict_users))

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    loss_avg = 1
    # beta = args.lr
    batch_size = args.bs
    samples_amount = 30
    idxs_users = range(args.num_users)
    # dict_usersZmodel = []
    # dict_userFLmodel =[]


    for iter in range(args.epochs):
        #dict_usersZ = []
        idxs_users = range(args.num_users)

        w_locals = []
        grad_locals = []
        w_locals_org = []
        w_locals_org_res = []
        #zmodel_locals = []
        for idx in idxs_users:
            if idx in poison_perm:
                need_poison = True
            else:
                need_poison = False

            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], need_poison=need_poison, poison_portion=poison_portion)
            #Z_model = dict_usersZmodel[idx]

            #user0_zmodel = dict_usersZmodel[0]

            #FL_model = copy.deepcopy(net_glob).to(args.device) #dict_userFLmodel[idx]
            #vibi_temp = VIBI(copy.deepcopy(Z_model.explainer), Z_model.approximator, copy.deepcopy(net_glob).to(args.device), k=args.k, num_samples=args.num_samples)
            #vibi_temp.to(device)


            # lr_train_dl, Z_model, net_glob_w, org_net_glob_w, org_net_local_update = vibi_train(args=args, ldr_train=local.ldr_train, z_dim=dimZ, user_idx=idx,
            #                                 vibi=vibi_temp, optimizer=optimizer, FL_model=FL_model, net_glob=net_glob, \
            #                                 org_net_glob=copy.deepcopy(org_net_glob).to(args.device), user0_zmodel= user0_zmodel)

            net_local_w, loss_item = local.train(copy.deepcopy(net_glob).to(args.device), idx)
            glob_w = net_glob.state_dict()
            new_lcoal_grad = net_glob.state_dict()
            for k in glob_w.keys():
                new_lcoal_grad[k] = glob_w[k] - net_local_w[k]


            #dict_usersZ.append(lr_train_dl)
            #dict_usersZmodel[idx] = Z_model
            #zmodel_locals.append(copy.deepcopy(Z_model.state_dict()))
            w_locals.append(copy.deepcopy(net_local_w))
            grad_locals.append(copy.deepcopy(new_lcoal_grad))
            #w_locals_org.append(copy.deepcopy(org_net_glob_w))


        #w_glob = FedAvg(w_locals)
        w_glob = FedAvg_grad(grad_locals, glob_w)

        #org_w_glob = FedAvg(w_locals_org)

        net_glob.load_state_dict(w_glob)
        #org_net_glob.load_state_dict(org_w_glob)


        #vibi_glob =AvgCP(dict_usersZmodel)
        #vibi = init_cp_model(args, dimZ)
        #vibi.load_state_dict(vibi_glob)
        #dict_usersZmodel[0].load_state_dict(cp_z_model_w)
        print("epoch: ", iter)
        acc_temp, poison_acc_temp = acc_evaluation_org(net_glob, dataset_test, args)
        acc_test.append(acc_temp)
        poison_acc.append(poison_acc_temp)
        print("org_acc ",acc_test)
        print("poison_acc_list:", poison_acc)
        #acc_temp = acc_evaluation_org(org_net_glob, dataset_test, args)
        #acc_test_org.append(acc_temp)



    print(acc_test)
    #print(acc_test_org)
    net_glob.eval()





