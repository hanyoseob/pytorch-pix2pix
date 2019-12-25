from model import *
from dataset import *

import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class Train:
    def __init__(self, args):
        self.opts = args

        self.train_continue = args.train_continue

        self.scope = args.scope
        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log

        self.dir_data = args.dir_data
        self.dir_result = args.dir_result

        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size

        self.lr_G = args.lr_G
        self.lr_D = args.lr_D

        self.wgt_l1 = args.wgt_l1
        self.wgt_gan = args.wgt_gan

        self.optim = args.optim
        self.beta1 = args.beta1

        self.ny_in = args.ny_in
        self.nx_in = args.nx_in
        self.nch_in = args.nch_in

        self.ny_load = args.ny_load
        self.nx_load = args.nx_load
        self.nch_load = args.nch_load

        self.ny_out = args.ny_out
        self.nx_out = args.nx_out
        self.nch_out = args.nch_out

        self.nch_ker = args.nch_ker

        self.data_type = args.data_type

        self.gpu_ids = args.gpu_ids
        self.num_freq = args.num_freq

        if self.gpu_ids and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % self.gpu_ids[0])
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.device = torch.device("cpu")

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

    def preprocess(self, data):
        nomalize = Nomalize()
        randflip = RandomFlip()
        rescale = Rescale((self.ny_load, self.nx_load))
        randomcrop = RandomCrop((self.ny_in, self.nx_in))
        totensor = ToTensor()
        return totensor(randomcrop(rescale(randflip(nomalize(data)))))
        # return  transforms.Compose([Nomalize(), RandomFlip(), Rescale(286), RandomCrop(256), ToTensor()])

    def deprocess(self, data):
        tonumpy = ToNumpy()
        denomalize = Denomalize()
        return denomalize(tonumpy(data))

    def train(self):
        dir_data_train = os.path.join(self.dir_data, 'facades', 'train')
        dir_data_val = os.path.join(self.dir_data, 'facades', 'val')

        log_dir_train = os.path.join(self.dir_log, self.scope, 'train')
        log_dir_val = os.path.join(self.dir_log, self.scope, 'val')

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr_G = self.lr_G
        lr_D = self.lr_D

        wgt_l1 = self.wgt_l1
        wgt_gan = self.wgt_gan

        batch_size = self.batch_size
        device = self.device

        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        num_freq = self.num_freq

        ## setup dataset
        dataset_train = PtDataset(dir_data_train, transform=self.preprocess)
        dataset_val = PtDataset(dir_data_val, transform=transforms.Compose([Nomalize(), ToTensor()]))

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

        num_train = len(dataset_train)
        num_val = len(dataset_val)

        num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))
        num_batch_val = int((num_val / batch_size) + ((num_val % batch_size) != 0))

        ## setup network
        netG = UNet(nch_in, nch_out, nch_ker)
        netD = Discriminator(2*nch_in, nch_ker)

        init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netD, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        st_epoch = 0

        if train_continue == 'on':
            netG, netD, st_epoch = self.load(netG, netD)

        ## setup loss & optimization
        fn_L1 = nn.L1Loss().to(device) # L1
        fn_GAN = nn.BCELoss().to(device)

        paramsG = netG.parameters()
        paramsD = netD.parameters()

        optimG = torch.optim.Adam(paramsG, lr=lr_G, betas=(self.beta1, 0.999))
        optimD = torch.optim.Adam(paramsD, lr=lr_D, betas=(self.beta1, 0.999))

        # schedG = get_scheduler(optimG, self.opts)
        # schedD = get_scheduler(optimD, self.opts)

        # schedG = torch.optim.lr_scheduler.ReduceLROnPlateau(optimG, 'min', factor=0.5, patience=20, verbose=True)
        # schedD = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimD, 'min', factor=0.5, patience=20, verbose=True)

        schedG = torch.optim.lr_scheduler.ExponentialLR(optimG, gamma=0.9)
        schedD = torch.optim.lr_scheduler.ExponentialLR(optimD, gamma=0.9)

        ## setup tensorboard
        writer_train = SummaryWriter(log_dir=log_dir_train)
        writer_val = SummaryWriter(log_dir=log_dir_val)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            ## training phase
            netG.train()
            netD.train()

            gen_loss_l1_train = 0
            gen_loss_gan_train = 0
            disc_loss_real_train = 0
            disc_loss_fake_train = 0

            for i, data in enumerate(loader_train, 1):
                def should(freq):
                    return freq > 0 and (i % freq == 0 or i == num_batch_train)

                input = data['input'].to(device)
                label = data['label'].to(device)

                # forward netG
                output = netG(input)

                # backward netD
                fake = torch.cat([input, output], dim=1)
                real = torch.cat([input, label], dim=1)

                set_requires_grad(netD, True)
                optimD.zero_grad()

                pred_fake = netD(fake.detach())
                pred_real = netD(real)

                disc_loss_real = fn_GAN(pred_real, torch.ones_like(pred_real))
                disc_loss_fake = fn_GAN(pred_fake, torch.zeros_like(pred_fake))
                disc_loss = 0.5 * (disc_loss_real + disc_loss_fake)

                disc_loss.backward()
                optimD.step()

                # backward netG
                fake = torch.cat([input, output], dim=1)

                set_requires_grad(netD, False)
                optimG.zero_grad()

                pred_fake = netD(fake)

                gen_loss_gan = fn_GAN(pred_fake, torch.ones_like(pred_fake))
                gen_loss_l1 = fn_L1(output, label)
                gen_loss = (wgt_l1 * gen_loss_l1) + (wgt_gan * gen_loss_gan)

                gen_loss.backward()
                optimG.step()

                # get losses
                gen_loss_l1_train += gen_loss_l1.item()
                gen_loss_gan_train += gen_loss_gan.item()
                disc_loss_fake_train += disc_loss_fake.item()
                disc_loss_real_train += disc_loss_real.item()

                print('TRAIN: EPOCH %d: BATCH %04d/%04d: '
                      'GEN L1: %.4f GEN GAN: %.4f DISC FAKE: %.4f DISC REAL: %.4f'
                      % (epoch, i, num_batch_train,
                         gen_loss_l1_train / i, gen_loss_gan_train / i, disc_loss_fake_train / i, disc_loss_real_train / i))

                if should(num_freq):
                    ## show output
                    input = self.deprocess(input)
                    output = self.deprocess(output)
                    label = self.deprocess(label)

                    writer_train.add_images('input', input, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('ouput', output, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('label', label, num_batch_train * (epoch - 1) + i, dataformats='NHWC')

                    ## show predict
                    pred_fake = self.deprocess(pred_fake)
                    pred_real = self.deprocess(pred_real)

                    writer_train.add_images('pred_fake', pred_fake, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('pred_real', pred_real, num_batch_train * (epoch - 1) + i, dataformats='NHWC')

            writer_train.add_scalar('gen_loss_L1', gen_loss_l1_train / num_batch_train, epoch)
            writer_train.add_scalar('gen_loss_GAN', gen_loss_gan_train / num_batch_train, epoch)
            writer_train.add_scalar('disc_loss_fake', disc_loss_fake_train / num_batch_train, epoch)
            writer_train.add_scalar('disc_loss_real', disc_loss_real_train / num_batch_train, epoch)

            ## validation phase
            with torch.no_grad():
                netG.eval()
                netD.eval()

                gen_loss_l1_val = 0
                gen_loss_gan_val = 0
                disc_loss_real_val = 0
                disc_loss_fake_val = 0

                for i, data in enumerate(loader_val, 1):
                    input = data['input'].to(device)
                    label = data['label'].to(device)

                    output = netG(input)

                    fake = torch.cat([input, output], dim=1)
                    real = torch.cat([input, label], dim=1)

                    pred_fake = netD(fake)
                    pred_real = netD(real)

                    disc_loss_real = fn_GAN(pred_real, torch.ones_like(pred_real))
                    disc_loss_fake = fn_GAN(pred_fake, torch.zeros_like(pred_fake))
                    disc_loss = 0.5 * (disc_loss_real + disc_loss_fake)

                    gen_loss_gan = fn_GAN(pred_fake, torch.ones_like(pred_fake))
                    gen_loss_l1 = fn_L1(output, label)
                    gen_loss = (wgt_l1 * gen_loss_l1) + (wgt_gan * gen_loss_gan)

                    gen_loss_l1_val += gen_loss_l1.item()
                    gen_loss_gan_val += gen_loss_gan.item()
                    disc_loss_real_val += disc_loss_real.item()
                    disc_loss_fake_val += disc_loss_fake.item()

                    print('VALID: EPOCH %d: BATCH %04d/%04d: '
                          'GEN L1: %.4f GEN GAN: %.4f DISC FAKE: %.4f DISC REAL: %.4f'
                          % (epoch, i, num_batch_val,
                             gen_loss_l1_val / i, gen_loss_gan_val / i, disc_loss_fake_val / i, disc_loss_real_val / i))

                    if should(num_freq):
                        ## show output
                        input = self.deprocess(input)
                        output = self.deprocess(output)
                        label = self.deprocess(label)

                        writer_val.add_images('input', input, num_batch_val * (epoch - 1) + i, dataformats='NHWC')
                        writer_val.add_images('ouput', output, num_batch_val * (epoch - 1) + i, dataformats='NHWC')
                        writer_val.add_images('label', label, num_batch_val * (epoch - 1) + i, dataformats='NHWC')
                        # add_figure(output, label, writer_val, epoch=epoch, ylabel='Density', xlabel='Radius', namescope='train/gen')

                        ## show predict
                        pred_fake = self.deprocess(pred_fake)
                        pred_real = self.deprocess(pred_real)

                        writer_val.add_images('pred_fake', pred_fake, num_batch_val * (epoch - 1) + i, dataformats='NHWC')
                        writer_val.add_images('pred_real', pred_real, num_batch_val * (epoch - 1) + i, dataformats='NHWC')
                        # add_figure(pred_fake, pred_real, writer_val, epoch=epoch, ylabel='Probability', xlabel='Radius', namescope='train/discrim')

                writer_val.add_scalar('gen_loss_L1', gen_loss_l1_val / num_batch_val, epoch)
                writer_val.add_scalar('gen_loss_GAN', gen_loss_gan_val / num_batch_val, epoch)
                writer_val.add_scalar('disc_loss_fake', disc_loss_fake_val / num_batch_val, epoch)
                writer_val.add_scalar('disc_loss_real', disc_loss_real_val / num_batch_val, epoch)

            # update schduler
            schedG.step()
            schedD.step()

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


def add_plot(output, label, writer, epoch=[], ylabel='Density', xlabel='Radius', namescope=[]):
    fig, ax = plt.subplots()

    ax.plot(output.transpose(1, 0).detach().numpy(), '-')
    ax.plot(label.transpose(1, 0).detach().numpy(), '--')

    ax.set_xlim(0, 400)

    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    writer.add_figure(namescope, fig, epoch)


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


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler