import torch
import matplotlib.pyplot as plt
from config.config import L, device
from src.utils import create_data, create_moments

# For saving and loading
import os
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(data_dir, exist_ok=True)
stack_dir = os.path.join(data_dir, 'stacks')
os.makedirs(stack_dir, exist_ok=True)


def create_data_stacks(num_unique=200):
    # Initialize and check single data sample shapes
    m1real, m2real, rho, sig = create_data()
    print(rho.shape)
    print(m1real.shape)
    print(m2real.shape)

    # Create empty lists and stacks to hold all data
    rhos = []
    sigs = []
    m1_list = []
    m2_list = []
    rho_list = []
    sig_list = []

    # Generate families of rhos and sigs
    for i in range(num_unique):
        # Parameters of family of distribution
        mu_rho = torch.rand(2)
        sd_rho = torch.rand(2) * (1/10 - 1/20) + 1/20

        # Parameters of family of signal
        mu_sig = torch.rand(2)
        sd_sig = torch.rand(2) * (1/10 - 1/20) + 1/20
        amp_sig = torch.ones(2)
        amp_sig = amp_sig / amp_sig.sum()

        # Data creation
        _, _, rho, sig = create_data(
            mu_rho=mu_rho, sd_rho=sd_rho, amp_sig=amp_sig,
            mu_sig=mu_sig, sd_sig=sd_sig
        )

        rhos.append(rho)
        sigs.append(sig)

    rhos = torch.stack(rhos).to(device)
    sigs = torch.stack(sigs).to(device)

    # Create all combinations of moments
    for i in range(num_unique):
        for j in range(num_unique):
            m1real, m2real = create_moments(rhos[i], sigs[j])

            m1_list.append(m1real)
            m2_list.append(m2real)
            rho_list.append(rhos[i])
            sig_list.append(sigs[j])

        if ((i + 1) % 100 == 0):
            print(f'Finished {i}th run')

    print(len(m1_list))
    print(len(m2_list))
    print(len(rho_list))
    print(len(sig_list))

    # Stack all data
    m1_stack = torch.stack(m1_list).to(device)
    m2_stack = torch.stack(m2_list).to(device)
    rho_stack = torch.stack(rho_list).to(device)
    sig_stack = torch.stack(sig_list).to(device)

    print(m1_stack.shape)
    print(m2_stack.shape)
    print(rho_stack.shape)
    print(sig_stack.shape)

    # Save all data in a new subfolder inside data/
    torch.save(m1_stack, os.path.join(stack_dir, f'm1_stack_{num_unique}.pt'))
    torch.save(m2_stack, os.path.join(stack_dir, f'm2_stack_{num_unique}.pt'))
    torch.save(rho_stack, os.path.join(stack_dir, f'rho_stack_{num_unique}.pt'))
    torch.save(sig_stack, os.path.join(stack_dir, f'sig_stack_{num_unique}.pt'))

if __name__ == "__main__":
    create_data_stacks()
