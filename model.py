from layer import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64):
        super(UNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker

        self.enc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, stride=2, bnorm=False, brelu=0.2, bdrop=False)
        self.enc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, stride=2, bnorm=True, brelu=0.2, bdrop=False)
        self.enc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, stride=2, bnorm=True, brelu=0.2, bdrop=False)
        self.enc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, stride=2, bnorm=True, brelu=0.2, bdrop=False)
        self.enc5 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, stride=2, bnorm=True, brelu=0.2, bdrop=False)
        self.enc6 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, stride=2, bnorm=True, brelu=0.2, bdrop=False)
        self.enc7 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, stride=2, bnorm=True, brelu=0.2, bdrop=False)
        self.enc8 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, stride=2, bnorm=True, brelu=0.0, bdrop=False)

        self.dec8 = DECNR2d(1 * 8 * self.nch_ker, 8 * self.nch_ker, stride=2, bnorm=True, brelu=0.0, bdrop=0.5)
        self.dec7 = DECNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, stride=2, bnorm=True, brelu=0.0, bdrop=0.5)
        self.dec6 = DECNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, stride=2, bnorm=True, brelu=0.0, bdrop=0.5)
        self.dec5 = DECNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, stride=2, bnorm=True, brelu=0.0, bdrop=False)
        self.dec4 = DECNR2d(2 * 8 * self.nch_ker, 4 * self.nch_ker, stride=2, bnorm=True, brelu=0.0, bdrop=False)
        self.dec3 = DECNR2d(2 * 4 * self.nch_ker, 2 * self.nch_ker, stride=2, bnorm=True, brelu=0.0, bdrop=False)
        self.dec2 = DECNR2d(2 * 2 * self.nch_ker, 1 * self.nch_ker, stride=2, bnorm=True, brelu=0.0, bdrop=False)
        self.dec1 = Deconv2d(2 * 1 * self.nch_ker, 1 * self.nch_out, stride=2)
        # self.dec1 = DECNR2d(2 * 1 * self.nch_ker, 1 * self.nch_out, bnorm=False, brelu=0.0, bdrop=False)

    def forward(self, x):

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        enc7 = self.enc7(enc6)
        enc8 = self.enc8(enc7)

        dec8 = self.dec8(enc8)
        dec7 = self.dec7(torch.cat([enc7, dec8], dim=1))
        dec6 = self.dec6(torch.cat([enc6, dec7], dim=1))
        dec5 = self.dec5(torch.cat([enc5, dec6], dim=1))
        dec4 = self.dec4(torch.cat([enc4, dec5], dim=1))
        dec3 = self.dec3(torch.cat([enc3, dec4], dim=1))
        dec2 = self.dec2(torch.cat([enc2, dec3], dim=1))
        dec1 = self.dec1(torch.cat([enc1, dec2], dim=1))

        x = torch.tanh(dec1)

        return x


class Discriminator(nn.Module):
    def __init__(self, nch_in, nch_ker=64):
        super(Discriminator, self).__init__()

        self.nch_in = nch_in
        self.nch_ker = nch_ker

        self.dsc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, bnorm=False, brelu=0.2)
        self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, bnorm=True, brelu=0.2)
        self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, bnorm=True, brelu=0.2)
        self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, bnorm=True, brelu=0.2, stride=1)
        self.dsc5 = CNR2d(8 * self.nch_ker, 1,                bnorm=True, brelu=0.2, stride=1)

    def forward(self, x):

        x = self.dsc1(x)
        x = self.dsc2(x)
        x = self.dsc3(x)
        x = self.dsc4(x)
        x = self.dsc5(x)

        x = torch.sigmoid(x)

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
#
# class Discriminator(nn.Module):
#     def __init__(self, nch_in, nch_out):
#         super(Discriminator, self).__init__()
#
#         # # [nbatch, 2, 400] => [nbatch, 1 * 64, 200]
#         # self.conv1 = nn.Conv1d(nch_in, 1 * nch_out, kernel_size=4, stride=2, padding=1)
#         #
#         # # [nbatch, 1 * 64, 200] => [nbatch, 2 * 64, 100]
#         # self.conv2 = nn.Conv1d(1 * nch_out, 2 * nch_out, kernel_size=4, stride=2, padding=1)
#         #
#         # # [nbatch, 2 * 64, 100] => [nbatch, 4 * 64, 50]
#         # self.conv3 = nn.Conv1d(2 * nch_out, 4 * nch_out, kernel_size=4, stride=2, padding=1)
#         #
#         # # [nbatch, 4 * 64, 50] => [nbatch, 8 * 64, 49]
#         # self.conv4 = nn.Conv1d(4 * nch_out, 8 * nch_out, kernel_size=4, stride=1, padding=1)
#         #
#         # # [nbatch, 8 * 64, 49] => [nbatch, 1, 48]
#         # self.conv5 = nn.Conv1d(8 * nch_out, 1, kernel_size=4, stride=1, padding=1)
#
#         sequence = [
#             nn.Conv1d(nch_in, 1 * nch_out, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv1d(1 * nch_out, 2 * nch_out, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv1d(2 * nch_out, 4 * nch_out, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv1d(4 * nch_out, 8 * nch_out, kernel_size=4, stride=1, padding=1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv1d(8 * nch_out, 1, kernel_size=4, stride=1, padding=1),
#             nn.LeakyReLU(0.2, True),
#         ]
#
#         sequence += [
#             nn.Sigmoid()
#         ]
#
#         self.model = nn.Sequential(*sequence)
#
#     # def forward(self, input, target):
#         # x = torch.cat([torch.reshape(input, (input.shape[0], 1, input.shape[1])),
#         #                torch.reshape(target, (target.shape[0], 1, target.shape[1]))], dim=1)
#         #
#         # x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
#         # x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
#         # x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
#         # x = F.leaky_relu(self.conv4(x), negative_slope=0.2)
#         #
#         # x = torch.sigmoid(self.conv5(x)).squeeze()
#         #
#         # return x
#     def forward(self, x):
#         return self.model(x)

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