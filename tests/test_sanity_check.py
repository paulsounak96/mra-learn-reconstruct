# moved from project root
import torch
import os
import pytest
from config.config import L, num_samps, device
from src.utils import fft

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def load_tensors():
    signal = torch.load(os.path.join(DATA_DIR, 'signal.pt'), map_location=device)
    rho = torch.load(os.path.join(DATA_DIR, 'rho.pt'), map_location=device)
    v_fftori = torch.load(os.path.join(DATA_DIR, 'v_fftori.pt'), map_location=device)
    v_fftlist = torch.load(os.path.join(DATA_DIR, 'v_fftlist.pt'), map_location=device)
    v_translist = torch.load(os.path.join(DATA_DIR, 'v_translist.pt'), map_location=device)
    return signal, rho, v_fftori, v_fftlist, v_translist

def test_sanity_check():
    signal, rho, v_fftori, v_fftlist, v_translist = load_tensors()

    m1_ground = torch.mean(v_fftori, 0)
    m1_trans = torch.mean(v_fftlist, 0)

    m2_ground = (v_fftori.T @ v_fftori) / num_samps
    dftmtx = fft(torch.eye(L, device=device))
    m2_trans = dftmtx.T @ ((v_translist.T @ v_translist).to(torch.cfloat) / num_samps) @ dftmtx

    rel_err_m1 = torch.norm(m1_trans - m1_ground) / torch.norm(m1_ground)
    rel_err_m2 = torch.norm(m2_trans - m2_ground) / torch.norm(m2_ground)

    # Set reasonable thresholds for sanity check
    assert rel_err_m1 < 1e-5, f"Relative error in first moment too high: {rel_err_m1}"
    assert rel_err_m2 < 1e-5, f"Relative error in second moment too high: {rel_err_m2}"
