# config.py
import argparse
import sys
import os
import yaml
import torch

# Load defaults from YAML config file
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path, 'r') as f:
	config = yaml.safe_load(f)

L = config.get('L', 20)
s = config.get('s', 3)
noise_sdmult = config.get('noise_sdmult', 0)
num_samps = config.get('num_samps', 10000)
num_test = config.get('num_test', 1000)
enable_cuda = config.get('enable_cuda', True)
max_epoch = config.get('max_epoch', 20)
batch_size = config.get('batch_size', 64)
show_iter = config.get('show_iter', 100)
lr = config.get('lr', 1e-4)


# Create argument parser for optional overrides
parser = argparse.ArgumentParser(description='Configuration for the project.')
parser.add_argument('--L', type=int, default=L, help='Length of the signal (default: %(default)s)')
parser.add_argument('--s', type=int, default=s, help='Shift parameter (default: %(default)s)')
parser.add_argument('--noise_sdmult', type=float, default=noise_sdmult, help='Noise standard deviation multiplier (default: %(default)s)')
parser.add_argument('--num_samps', type=int, default=num_samps, help='Number of samples (default: %(default)s)')
parser.add_argument('--num_test', type=int, default=num_test, help='Number of samples for test (default: %(default)s)')
parser.add_argument('--enable_cuda', type=bool, default=enable_cuda, help='Enable CUDA if available (default: %(default)s)')
parser.add_argument('--max_epoch', type=bool, default=max_epoch, help='Maximum number of epochs (default: %(default)s)')
parser.add_argument('--batch_size', type=bool, default=batch_size, help='Batch size (default: %(default)s)')
parser.add_argument('--show_iter', type=bool, default=show_iter, help='Number of epochs are which to display results (default: %(default)s)')
parser.add_argument('--lr', type=bool, default=lr, help='Learning rate (default: %(default)s)')


args, _ = parser.parse_known_args(sys.argv[1:])
L = args.L
s = args.s
noise_sdmult = args.noise_sdmult
num_samps = args.num_samps
num_test = args.num_test
num_train = num_samps - num_test
max_epoch = args.max_epoch
batch_size = args.batch_size
show_iter = args.show_iter
lr = args.lr
weight_decay = lr
enable_cuda = args.enable_cuda
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
