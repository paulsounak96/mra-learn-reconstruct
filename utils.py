import torch
import torch.nn.functional as F
from config import L, s, noise_sdmult, num_samps
import numpy as np

# Important global variables used across the project
multiplier = torch.linspace(-L/2, L/2 -1, steps = L) * -2*torch.pi*1j/L
denom = torch.sqrt(torch.tensor(2*torch.pi))

def circ(tensor, dim):
    """get a circulant version of the tensor along the {dim} dimension.

    The additional axis is appended as the last dimension.
    E.g. tensor=[0,1,2], dim=0 --> [[0,1,2],[2,0,1],[1,2,0]]"""
    S = tensor.shape[dim]
    tmp = torch.cat([tensor.flip((dim,)), torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=S-1)], dim=dim)
    return tmp.unfold(dim, S, 1).flip((-1,)).T

def mix_gauss(x, amp=torch.tensor([0.3, 1, -0.5]), mu=torch.tensor([1/8, 1/2, 3/4]), sd=torch.tensor([1/15, 1/10, 1/20])):
    # sd = sd * torch.ones(len(amp))
    to_ret = amp[:, None] * torch.exp(-0.5 * ((x[None, :] - mu[:, None])/sd[:, None])**2) / (denom * sd[:, None])

    return to_ret.sum(dim = 0)

def circular_trans_fourier(v, s_list):
    # multiplier = torch.linspace(-L/2, L/2 -1, steps = L) * -2*torch.pi*1j/L # Moved to global scope
    rot_mat = (s_list[:, None].type(torch.complex64) @ multiplier[None, :]).exp()
    fft_v = torch.fft.ifftshift(torch.fft.fft(torch.fft.fftshift(v)))
    v_fftlist = rot_mat @ torch.diag(fft_v)
    v_translist = torch.fft.ifftshift(torch.fft.ifft(torch.fft.fftshift(v_fftlist))).real

    return v_translist

def circular_trans(v, s_list):
    v_translist = torch.empty((0, L))
    #v_fftlist = torch.empty((0, L))
    for s in s_list:
        trans_signal = torch.cat((v[-s:], v[:-s])) #+ torch.normal(mean=0.0, std=torch.ones(L)*signal.max()*noise_sdmult)
        v_translist = torch.vstack((v_translist, trans_signal))
        #v_fftlist = torch.vstack((v_fftlist, torch.fft.ifftshift(torch.fft.fft(torch.fft.fftshift(trans_signal)))))

    return v_translist

def diff_min(mat, rho = 200, dim = -1):
    return -(1/rho) * torch.logsumexp(-rho * mat, dim)


def create_moments(rho, sig):
    m1real = (circ(sig, 0) @ rho[:, None]).flatten()
    m2real = circ(sig, 0) @ torch.diag(rho) @ circ(sig, 0).T

    return m1real, m2real


def create_data(L=L, mu_rho = torch.tensor([1/4, 3/4]), sd_rho = torch.tensor([1/11, 1/6]), amp_sig = torch.tensor([1, 1/2]), mu_sig = torch.tensor([1/8, 5/8]), sd_sig = torch.tensor([1/4, 1/6])):
    rho = 0
    for j in range(len(mu_rho)):
        normal_dist = torch.distributions.normal.Normal(L*mu_rho[j], (L*sd_rho[j]))
        rho += normal_dist.log_prob(torch.linspace(0, L-1, steps=L)).exp()
    rho = (rho/sum(rho))

    sig = 0
    for j in range(len(mu_sig)):
        normal_dist = torch.distributions.normal.Normal(L*mu_sig[j], (L*sd_sig[j]))
        sig += amp_sig[j] * normal_dist.log_prob(torch.linspace(0, L-1, steps=L)).exp()

    m1real, m2real = create_moments(rho = rho, sig = sig)

    return m1real, m2real, rho, sig


def best_registration(original, recon, search_space_mult = 100):
    rots = torch.linspace(0, L, steps=search_space_mult*L)
    rot_reconlist = circular_trans_fourier(recon, rots)

    errors = torch.norm(rot_reconlist - original, dim=1)
    min_idx = torch.argmin(errors)

    best_s = rots[min_idx]
    best_error = errors[min_idx]/torch.norm(original)
    best_recon = rot_reconlist[min_idx]

    return best_s, best_error, best_recon


def moment_registration(rho, sig, m1, m2, search_space_mult = 100):
    rots = torch.linspace(0, L, steps=search_space_mult*L)
    sig_rotlist = circular_trans_fourier(sig, rots)

    s_quad = torch.arange(L)[:, None].type(torch.complex64)
    # multiplier = torch.linspace(-L/2, L/2 -1, steps = L) * -2*torch.pi*1j/L # Moved to global scope
    loss_lst = []

    for sig_rot in sig_rotlist:
        recon_fft = torch.fft.ifftshift(torch.fft.fft(torch.fft.fftshift(sig_rot)))

        rot_mat = (s_quad @ multiplier[None, :]).exp()
        Rvlist = rot_mat @ torch.diag(recon_fft)

        m1hat = torch.sum((torch.diag(rho) @ Rvlist), 0)
        m2hat = (Rvlist.T @ torch.diag(rho) @ Rvlist)
        loss = torch.norm(m1 - m1hat)/torch.norm(m1) + torch.norm(m2 - m2hat)/torch.norm(m2)

        loss_lst.append(loss.to('cpu').data.numpy())

    min_idx = np.argmin(loss_lst)
    max_idx = np.argmax(loss_lst)
    # print([loss_lst[max_idx], loss_lst[min_idx], loss_lst[max_idx]/loss_lst[min_idx]])

    return rots[min_idx], sig_rotlist[min_idx]