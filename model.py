from layer import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()





    def forward(self, x):
        x, x_skip_1 = self.downs[0](x)
        x, x_skip_2 = self.downs[1](x)
        x = self.ups[0](x)
        x = self.ups[1](torch.cat((x_skip_2, x), 1))
        x = self.last_layer(torch.cat((x_skip_1, x), 1))  # cat in channel dim
        return x


class AutoEncoder1d(nn.Module):
    def __init__(self, nch_in, nch_out):
        super(AutoEncoder1d, self).__init__()

        # self.nch_in = args.nch_in
        # self.nch_out = args.nch_out
        self.nch_in = nch_in
        self.nch_out = nch_out

        self.efc1 = nn.Linear(self.nch_in, 400)
        self.efc2 = nn.Linear(400, 200)
        self.efc3 = nn.Linear(200, 100)
        self.efc4 = nn.Linear(100, 50)

        self.dfc4 = nn.Linear(50, 100)
        self.dfc3 = nn.Linear(100, 200)
        self.dfc2 = nn.Linear(200, 400)
        self.dfc1 = nn.Linear(400, self.nch_out)

    def forward(self, x):
        x = F.relu(self.efc1(x))
        x = F.relu(self.efc2(x))
        x = F.relu(self.efc3(x))
        x = F.relu(self.efc4(x))

        x = F.relu(self.dfc4(x))
        x = F.relu(self.dfc3(x))
        x = F.relu(self.dfc2(x))
        x = self.dfc1(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, nch_in, nch_out):
        super(Discriminator, self).__init__()

        # # [nbatch, 2, 400] => [nbatch, 1 * 64, 200]
        # self.conv1 = nn.Conv1d(nch_in, 1 * nch_out, kernel_size=4, stride=2, padding=1)
        #
        # # [nbatch, 1 * 64, 200] => [nbatch, 2 * 64, 100]
        # self.conv2 = nn.Conv1d(1 * nch_out, 2 * nch_out, kernel_size=4, stride=2, padding=1)
        #
        # # [nbatch, 2 * 64, 100] => [nbatch, 4 * 64, 50]
        # self.conv3 = nn.Conv1d(2 * nch_out, 4 * nch_out, kernel_size=4, stride=2, padding=1)
        #
        # # [nbatch, 4 * 64, 50] => [nbatch, 8 * 64, 49]
        # self.conv4 = nn.Conv1d(4 * nch_out, 8 * nch_out, kernel_size=4, stride=1, padding=1)
        #
        # # [nbatch, 8 * 64, 49] => [nbatch, 1, 48]
        # self.conv5 = nn.Conv1d(8 * nch_out, 1, kernel_size=4, stride=1, padding=1)

        sequence = [
            nn.Conv1d(nch_in, 1 * nch_out, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(1 * nch_out, 2 * nch_out, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(2 * nch_out, 4 * nch_out, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(4 * nch_out, 8 * nch_out, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(8 * nch_out, 1, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*sequence)

    # def forward(self, input, target):
        # x = torch.cat([torch.reshape(input, (input.shape[0], 1, input.shape[1])),
        #                torch.reshape(target, (target.shape[0], 1, target.shape[1]))], dim=1)
        #
        # x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        # x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        # x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        # x = F.leaky_relu(self.conv4(x), negative_slope=0.2)
        #
        # x = torch.sigmoid(self.conv5(x)).squeeze()
        #
        # return x
    def forward(self, x):
        return self.model(x)

"""
class UNetPiece(nn.Module):
    def __init__(self, is_up, num_layers=2, in_channels=32, out_channels=32, kernel_size=3):
        super().__init__()
        padding = int((kernel_size-1)/2)
        self.is_up = is_up

        self.convs = nn.ModuleList()
        self.convs.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding))

        for i in range(num_layers-1):
            self.convs.append(
                nn.Conv2d(in_channels=out_channels, out_channels=num_channels, kernel_size=kernel_size, padding=padding))
            
        self.ReLU = nn.ReLU()

        if is_up:
            self.pool = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2)            
        else:
            self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
            x = self.ReLU(x)
        x_pooled = self.pool(x)

        if self.is_up:
            return x_pooled
        else:
            return x_pooled, x            
    

class UNet(nn.Module):
    def __init__(self, num_channels=16, kernel_size=3):
        super().__init__()

        padding = int((kernel_size-1)/2)
        self.ReLU = nn.ReLU()


        self.downs = nn.ModuleList([
            UNetPiece(
                is_up=False, num_layers=1,
                in_channels=1, out_channels=num_channels, kernel_size=kernel_size),
            UNetPiece(
                is_up=False, num_layers=1,
                in_channels=num_channels, out_channels=2*num_channels, kernel_size=kernel_size)])
                                   
        self.ups = nn.ModuleList([
            UNetPiece(
                is_up=True, num_layers=1,
                in_channels=2*num_channels, out_channels=2*num_channels, kernel_size=kernel_size),
            UNetPiece(
                is_up=True, num_layers=1,
                in_channels=4*num_channels, out_channels=2*num_channels, kernel_size=kernel_size),
            ])
        

        self.last_layer = nn.Conv2d(in_channels=3*num_channels, out_channels=1,
                                    kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x, x_skip_1 = self.downs[0](x)
        x, x_skip_2 = self.downs[1](x)
        x = self.ups[0](x)
        x = self.ups[1](torch.cat((x_skip_2, x), 1))
        x = self.last_layer(torch.cat((x_skip_1, x), 1))  # cat in channel dim
        return x

class Fully(nn.Module):
    def __init__(self, size, num_layers, num_channels):
        super().__init__()

        
        self.layers = nn.ModuleList([
            nn.Linear(in_features=num_channels, out_features=num_channels)
            for i in range(num_layers)])

        self.layers.insert(
            0, nn.Linear(in_features=size, out_features=num_channels))

        self.layers.append(
            nn.Linear(in_features=num_channels, out_features=size))

        self.nonlin = nn.ReLU()

    def forward(self, x):
        for l in self.layers:
            x = l(x)
            x = self.nonlin(x)
        return x
"""


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad