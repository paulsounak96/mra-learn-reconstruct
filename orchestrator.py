# Import basic libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# For plotting
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [5, 3]

# For saving and loading
import os
import pickle
from tqdm import tqdm

# Important directory for saving data
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)

# Import variables,helper functions and neural networks from other files
from config import L, s, noise_sdmult, num_samps, device
from utils import circ, mix_gauss, circular_trans_fourier, circular_trans, diff_min, create_moments, create_data, best_registration, moment_registration, multiplier
from models import rhonet, signet


def main():
    signal = torch.cat((torch.zeros(6, device=device), torch.ones(4, device=device), -torch.ones(4, device=device), torch.zeros(6, device=device)), 0)
    # signal = mix_gauss(torch.linspace(0, 1, steps=L)).to(device)  #, amp=torch.tensor([1.]), mu=torch.tensor([1/2]), sd=torch.tensor([1/3])
    #signal = torch.rand(L)
    signal *= 1/torch.norm(signal)
    fft_signal = torch.fft.ifftshift(torch.fft.fft(torch.fft.fftshift(signal)))

    # Plot signal in both real and fourier domain
    plt.figure()
    plt.plot(signal.cpu())
    plt.title("Signal (real domain)")
    plt.savefig(os.path.join(data_dir, 'signal_real.png'))
    plt.close()

    plt.figure()
    plt.plot(fft_signal.cpu().real, 'g', label='Real')
    plt.plot(fft_signal.cpu().imag, 'r--', label='Imag')
    plt.title("Fourier series for square wave")
    plt.legend()
    plt.savefig(os.path.join(data_dir, 'signal_fourier.png'))
    plt.close()

    # Generate shifts according to some distribution
    rho = torch.rand(L)
    rho = (rho/sum(rho)).to(device)
    s_list = torch.multinomial(rho, num_samps, replacement=True).to(device)
    #s_list = s_list.float() + torch.normal(mean=0.0, std=torch.ones(num_samps, device=device))
    #s_list[0] = 0.0


    plt.figure()
    plt.hist(s_list.cpu().numpy(), bins=int(L))
    plt.title("Histogram of s_list")
    plt.savefig(os.path.join(data_dir, 's_list_histogram.png'))
    plt.close()

    noise_rvs = torch.normal(mean=0.0, std=torch.ones((num_samps, L), device=device) * noise_sdmult)
    noise_fft = torch.fft.ifftshift(torch.fft.fft(torch.fft.fftshift(noise_rvs)))

    v_ori = circular_trans_fourier(signal, s_list).to(device)
    v_fftori = torch.fft.ifftshift(torch.fft.fft(torch.fft.fftshift(v_ori)))

    v_translist = v_ori + noise_rvs
    v_fftlist = torch.fft.ifftshift(torch.fft.fft(torch.fft.fftshift(v_translist)))

    # print(s_list[0])
    # plt.figure()
    # plt.plot(v_translist[0].cpu())
    # plt.title("First v_translist sample")
    # plt.savefig(os.path.join(data_dir, 'v_translist0.png'))
    # plt.close()


if __name__ == '__main__':
    main()