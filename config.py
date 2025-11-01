# config.py
import argparse


# Define default values for shared variables
L = 20
s = 3
noise_sdmult = 0
num_samps = 10000
enable_cuda = True

import torch
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

# Create argument parser

parser = argparse.ArgumentParser(description='Configuration for the project.')
parser.add_argument('--L', type=int, default=L, help='Length of the signal (default: %(default)s)')
parser.add_argument('--s', type=int, default=s, help='Shift parameter (default: %(default)s)')
parser.add_argument('--noise_sdmult', type=float, default=noise_sdmult, help='Noise standard deviation multiplier (default: %(default)s)')
parser.add_argument('--num_samps', type=int, default=num_samps, help='Number of samples (default: %(default)s)')
parser.add_argument('--enable_cuda', type=bool, default=enable_cuda, help='Enable CUDA if available (default: %(default)s)')

# Only parse arguments and update variables if this file is run as a script
if __name__ == '__main__':
    args = parser.parse_args()
    L = args.L
    s = args.s
    noise_sdmult = args.noise_sdmult
    num_samps = args.num_samps
    enable_cuda = args.enable_cuda
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
