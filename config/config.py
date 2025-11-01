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
enable_cuda = config.get('enable_cuda', True)

# Create argument parser for optional overrides
parser = argparse.ArgumentParser(description='Configuration for the project.')
parser.add_argument('--L', type=int, default=L, help='Length of the signal (default: %(default)s)')
parser.add_argument('--s', type=int, default=s, help='Shift parameter (default: %(default)s)')
parser.add_argument('--noise_sdmult', type=float, default=noise_sdmult, help='Noise standard deviation multiplier (default: %(default)s)')
parser.add_argument('--num_samps', type=int, default=num_samps, help='Number of samples (default: %(default)s)')
parser.add_argument('--enable_cuda', type=bool, default=enable_cuda, help='Enable CUDA if available (default: %(default)s)')

args, _ = parser.parse_known_args(sys.argv[1:])
L = args.L
s = args.s
noise_sdmult = args.noise_sdmult
num_samps = args.num_samps
enable_cuda = args.enable_cuda
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
