from utils import *
# from train import *
from train_w_discrim import *

import argparse

## setup parse
parser = argparse.ArgumentParser(description='Train the network for descattering related with LANL project',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--mode', default='train', choices=['train', 'test'], dest='mode')
parser.add_argument('--train_continue', default='off', choices=['on', 'off'], dest='train_continue')

parser.add_argument('--scope', default='trans2dens_1linear_l1_w_discrim', dest='scope')
# parser.add_argument('--scope', default='trans2dens_1linear_l1', dest='scope')
# parser.add_argument('--scope', default='trans2dens_1linear_l2', dest='scope')
# parser.add_argument('--scope', default='trans2dens_autoencoder', dest='scope')

parser.add_argument('--dir_checkpoint', default='./checkpoints', dest='dir_checkpoint')
parser.add_argument('--dir_log', default='./log', dest='dir_log')

parser.add_argument('--dir_data', default='./data', dest='dir_data')
parser.add_argument('--dir_result', default='./result', dest='dir_result')

parser.add_argument('--num_epoch', type=int,  default=1000, dest='num_epoch')
parser.add_argument('--batch_size', type=int, default=10, dest='batch_size')

parser.add_argument('--learning_rate', type=float, default=1e-2, dest='learning_rate')

parser.add_argument('--mu', type=float, default=1e-1, dest='mu')
parser.add_argument('--wgt_l1', type=float, default=1e2, dest='wgt_l1')
parser.add_argument('--wgt_gan', type=float, default=1e0, dest='wgt_gan')

parser.add_argument('--optim', default='adam', choices=['sgd', 'adam', 'rmsprop'], dest='optim')

parser.add_argument('--ny_in', type=int, default=1, dest='ny_in')
parser.add_argument('--nx_in', type=int, default=1, dest='nx_in')
parser.add_argument('--nch_in', type=int, default=400, dest='nch_in')

parser.add_argument('--ny_out', type=int, default=1, dest='ny_out')
parser.add_argument('--nx_out', type=int, default=1, dest='nx_out')
parser.add_argument('--nch_out', type=int, default=400, dest='nch_out')

parser.add_argument('--data_type', default='float32', dest='data_type')

PARSER = Parser(parser)

def main():
    ARGS = PARSER.get_arguments()
    PARSER.write_args()
    PARSER.print_args()

    TRAINER = Train(ARGS)

    if ARGS.mode == 'train':
        TRAINER.train()
    elif ARGS.mode == 'test':
        TRAINER.test(epoch=[])

if __name__ == '__main__':
    main()