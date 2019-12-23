from model import *
from dataset import *

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class Train:
    def __init__(self, args):
        self.train_continue = args.train_continue

        self.scope = args.scope
        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log

        self.dir_data = args.dir_data
        self.dir_result = args.dir_result

        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size

        self.learning_rate = args.learning_rate

        self.mu = args.mu
        self.wgt_l1 = args.wgt_l1
        self.wgt_gan = args.wgt_gan

        self.optim = args.optim

        self.ny_in = args.ny_in
        self.nx_in = args.nx_in
        self.nch_in = args.nch_in

        self.ny_out = args.ny_out
        self.nx_out = args.nx_out
        self.nch_out = args.nch_out

        self.data_type = args.data_type

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # def save(self, net, epoch):
    #     dir_checkpoint = os.path.join(self.dir_checkpoint, self.scope)
    #
    #     if not os.path.exists(dir_checkpoint):
    #         os.makedirs(dir_checkpoint)
    #
    #     torch.save(net.state_dict(),  '%s/model_epoch%04d.pth' % (dir_checkpoint, epoch))
    #
    # def load(self, net, epoch=[]):
    #     dir_checkpoint = os.path.join(self.dir_checkpoint, self.scope)
    #
    #     if not epoch:
    #         ckpt = os.listdir(dir_checkpoint)
    #         ckpt.sort()
    #         epoch = int(ckpt[-1].split('epoch')[1].split('.pt')[0])
    #
    #     net.load_state_dict(torch.load('%s/model_epoch%04d.pth' % (dir_checkpoint, epoch)))
    #
    #     return net, epoch

    def save(self, netG, netD, epoch):
        dir_checkpoint = os.path.join(self.dir_checkpoint, self.scope)

        if not os.path.exists(dir_checkpoint):
            os.makedirs(dir_checkpoint)

        torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict()},
                   '%s/model_epoch%04d.pth' % (dir_checkpoint, epoch))

    def load(self, netG, netD, epoch=[]):
        dir_checkpoint = os.path.join(self.dir_checkpoint, self.scope)

        if not epoch:
            ckpt = os.listdir(dir_checkpoint)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pt')[0])

        nets = torch.load('%s/model_epoch%04d.pth' % (dir_checkpoint, epoch))
        netG.load_state_dict(nets['netG'])
        netD.load_state_dict(nets['netD'])

        return netG, netD, epoch

    def train(self):
        log_dir_train = os.path.join(self.dir_log, self.scope, 'train')
        log_dir_val = os.path.join(self.dir_log, self.scope, 'val')

        train_continue = self.train_continue
        num_epoch = self.num_epoch
        learning_rate = self.learning_rate
        mu = self.mu
        wgt_l1 = self.wgt_l1
        wgt_gan = self.wgt_gan

        batch_size = self.batch_size
        device = self.device

        nch_in = self.nch_in
        nch_out = self.nch_out

        num_train = 8000
        num_val = 1000
        num_test = 1000

        num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))
        num_batch_val = int((num_val / batch_size) + ((num_val % batch_size) != 0))
        num_batch_test = int((num_test / batch_size) + ((num_test % batch_size) != 0))

        ## setup dataset
        # dataset_train = PtDataset('Data', slice(num_train), transform=transforms.Compose([ToTensor()]))
        # dataset_val = PtDataset('Data', slice(num_train, num_train + num_val),transform=transforms.Compose([ToTensor()]))

        dataset_train = PtDataset('Data', slice(num_train), transform=[])
        dataset_val = PtDataset('Data', slice(num_train, num_train + num_val), transform=[])
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

        ## setup network
        netG = nn.Linear(nch_in, nch_out).to(device)
        netD = Discriminator(2, 64).to(device)
        # net = AutoEncoder1d(nch_in, nch_out)
        st_epoch = 0

        if train_continue == 'on':
            netG, netD, st_epoch = self.load(netG, netD)

        ## setup loss & optimization
        l1_fn = nn.L1Loss().to(device) # L1
        gan_fn = nn.BCELoss().to(device)
        # gan_fn = nn.BCEWithLogitsLoss()

        # fid_fn = nn.MSELoss()  # L2
        # reg_fn = TV1dLoss()  # TV

        paramsG = netG.parameters()
        paramsD = netD.parameters()
        optimG = torch.optim.Adam(paramsG, lr=learning_rate)
        optimD = torch.optim.Adam(paramsD, lr=learning_rate)

        schedG = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimG, 'min', factor=0.5, patience=20, verbose=True)
        schedD = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimD, 'min', factor=0.5, patience=20, verbose=True)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.995)

        ## setup tensorboard
        writer_train = SummaryWriter(log_dir=log_dir_train)
        writer_val = SummaryWriter(log_dir=log_dir_val)

        # writer_train.add_graph(net, torch.randn(1, 400))

        ## run train
        # global_step = 0
        for epoch in range(st_epoch + 1, num_epoch + 1):
            ## training phase
            netG.train()
            netD.train()

            gen_loss_l1_train = 0
            gen_loss_gan_train = 0
            discrim_loss_train = 0

            for i, data in enumerate(loader_train, 1):
                input = data['input'].to(device)
                label = data['label'].to(device)

                output = netG(input)

                fake = torch.cat([torch.reshape(input, (input.shape[0], 1, input.shape[1])),
                                  torch.reshape(output, (output.shape[0], 1, output.shape[1]))], dim=1)
                real = torch.cat([torch.reshape(input, (input.shape[0], 1, input.shape[1])),
                                  torch.reshape(label, (label.shape[0], 1, label.shape[1]))], dim=1)

                pred_fake = netD(fake)
                pred_real = netD(real)

                # gen_loss_gan = (gan_fn(pred_fake, True)).mean()
                gen_loss_gan = gan_fn(pred_fake, torch.ones_like(pred_fake))
                gen_loss_l1 = l1_fn(output, label)
                gen_loss = wgt_l1 * gen_loss_l1 + wgt_gan * gen_loss_gan

                discrim_loss = 0.5 * (gan_fn(pred_fake, torch.zeros_like(pred_fake)) + gan_fn(pred_real, torch.ones_like(pred_real)))

                gen_loss_l1_train += gen_loss_l1.item()
                gen_loss_gan_train += gen_loss_gan.item()
                discrim_loss_train += discrim_loss.item()

                print('TRAIN: EPOCH %d: BATCH %04d/%04d: GEN L1: %.6f GEN GAN: %.6f DISCRIM: %.6f'
                      % (epoch, i, num_batch_train, gen_loss_l1_train / i, gen_loss_gan_train / i, discrim_loss_train / i))

                # update netD
                set_requires_grad(netD, True)
                optimD.zero_grad()
                discrim_loss.backward(retain_graph=True)
                optimD.step()

                # update netG
                set_requires_grad(netD, False)
                optimG.zero_grad()
                gen_loss.backward(retain_graph=False)
                optimG.step()

            writer_train.add_scalar('gen_loss_L1', gen_loss_l1_train / num_batch_train, epoch)
            writer_train.add_scalar('gen_loss_GAN', gen_loss_gan_train / num_batch_train, epoch)
            writer_train.add_scalar('discrim_loss', discrim_loss_train / num_batch_train, epoch)

            ## show output
            add_figure(output, label, writer_train, epoch=epoch, ylabel='Density', xlabel='Radius', namescope='train/gen')

            ## show predict
            add_figure(pred_fake, pred_real, writer_train, epoch=epoch, ylabel='Probability', xlabel='Radius', namescope='train/discrim')

            ## validation phase
            # with torch.no_grad():
            netG.eval()
            netD.eval()

            gen_loss_l1_val = 0
            gen_loss_gan_val = 0
            discrim_loss_val = 0

            for i, data in enumerate(loader_val, 1):
                input = data['input'].to(device)
                label = data['label'].to(device)

                output = netG(input)

                pred_real = netD(input, label)
                pred_fake = netD(input, output)

                gen_loss_gan = gan_fn(pred_fake, torch.ones_like(pred_fake))
                gen_loss_l1 = l1_fn(output, label)
                gen_loss = gen_loss_l1 + mu * gen_loss_gan
                discrim_loss = (gan_fn(pred_fake, torch.zeros_like(pred_fake)) + gan_fn(pred_real, torch.ones_like(pred_real))).mean()

                gen_loss_l1_val += gen_loss_l1.item()
                gen_loss_gan_val += gen_loss_gan.item()
                discrim_loss_val += discrim_loss.item()

                print('VALID: EPOCH %d: BATCH %04d/%04d: GEN L1: %.6f GEN GAN: %.6f DISCRIM: %.6f'
                      % (epoch, i, num_batch_val, gen_loss_l1_val / i, gen_loss_gan_val / i, discrim_loss_val / i))

            writer_train.add_scalar('gen_loss_L1', gen_loss_l1_val / num_batch_val, epoch)
            writer_train.add_scalar('gen_loss_GAN', gen_loss_gan_val / num_batch_val, epoch)
            writer_train.add_scalar('discrim_loss', discrim_loss_val / num_batch_val, epoch)

            ## show output
            add_figure(output, label, writer_val, epoch=epoch, ylabel='Density', xlabel='Radius', namescope='valid/gen')

            ## show predict
            add_figure(pred_fake, pred_real, writer_val, epoch=epoch, ylabel='Probability', xlabel='Radius', namescope='valid/discrim')

            ## update schduler
            schedG.step(gen_loss_l1_val)
            schedD.step(gen_loss_l1_val)

            ## save
            if (epoch % 10) == 0:
                self.save(netG, netD, epoch)
                # torch.save(net.state_dict(), 'Checkpoints/model_epoch_%d.pt' % epoch)

        writer_train.close()
        writer_val.close()


    def test(self, epoch=[]):
        dir_result = os.path.join(self.dir_result, self.scope)

        if not os.path.exists(dir_result):
            os.makedirs(dir_result)

        batch_size = 1
        device = self.device

        nch_in = self.nch_in
        nch_out = self.nch_out

        num_train = 8000
        num_val = 1000
        num_test = 1000

        num_batch_train = (num_train / batch_size) + ((num_train % batch_size) != 0)
        num_batch_val = (num_val / batch_size) + ((num_val % batch_size) != 0)
        num_batch_test = (num_test / batch_size) + ((num_test % batch_size) != 0)

        ## setup dataset
        # dataset_test = PtDataset('Data', slice(num_train + num_val, num_train + num_val + num_test),
        #                                           transform=transforms.Compose([ToTensor()]))
        dataset_test = PtDataset('Data', slice(num_train + num_val, num_train + num_val + num_test), transform=[])
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)

        ## setup network
        net = nn.Linear(nch_in, nch_out)
        # net = AutoEncoder1d(nch_in, nch_out)
        net, st_epoch = self.load(net)

        ## setup loss & optimization
        # loss_fn = nn.L1Loss() # L1
        loss_fn = nn.MSELoss()  # L2

        ## test phase
        # with torch.no_grad():
        net.eval()
        loss_test = 0
        for i, data in enumerate(loader_test, 1):
            input = data['input'].to(device)
            label = data['label'].to(device)

            output = net(input)
            loss = loss_fn(output, label)
            # loss_test += loss.item()
            loss_test += loss.item()

            np.save(os.path.join(dir_result, "output_%05d_1d.npy" % (i - 1)), np.float32(np.squeeze(output.detach().numpy())))

            print('TEST: %d/%d: LOSS: %.6f' % (i, num_batch_test, loss.item()))
        print('TEST: AVERAGE LOSS: %.6f' % (loss_test / num_batch_test))


def add_figure(output, label, writer, epoch=[], ylabel='Density', xlabel='Radius', namescope=[]):
    fig, ax = plt.subplots()

    ax.plot(output.transpose(1, 0).detach().numpy(), '-')
    ax.plot(label.transpose(1, 0).detach().numpy(), '--')

    ax.set_xlim(0, 400)

    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    writer.add_figure(namescope, fig, epoch)