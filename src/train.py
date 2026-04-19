import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from config.config import (
    L, num_test, num_train, max_epoch, batch_size, show_iter,
    lr, weight_decay, device
)
from src.models import rhonet
from src.utils import fft, diff_min, multiplier
import os


# =====================================================
# Checkpoints
# =====================================================
def save_checkpoint(model, optimizer, model_dir, epoch):
    directory = os.path.dirname(model_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, model_dir)
    print(f"Checkpoint saved to {model_dir}")


def resume_checkpoint(model, optimizer, model_dir, device_id):
    checkpoint = torch.load(
        model_dir,
        map_location=lambda storage, loc: storage.cuda(device=device_id)
    )
    model.load_state_dict(checkpoint['model_state'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    print(f"Resumed from checkpoint {model_dir} at epoch {checkpoint['epoch']}")
    return checkpoint['epoch']


# =====================================================
# Optimizer selection
# =====================================================
def use_optimizer(network, params):
    opt_type = params.get('optimizer', 'adam')
    if opt_type == 'sgd':
        return torch.optim.SGD(
            network.parameters(),
            lr=params.get('sgd_lr', 0.01),
            momentum=params.get('sgd_momentum', 0.9),
            weight_decay=params.get('l2_regularization', 0)
        )
    elif opt_type == 'rmsprop':
        return torch.optim.RMSprop(
            network.parameters(),
            lr=params.get('rmsprop_lr', 0.001),
            alpha=params.get('rmsprop_alpha', 0.99),
            momentum=params.get('rmsprop_momentum', 0),
            weight_decay=params.get('l2_regularization', 0)
        )
    else:  # default to Adam
        return torch.optim.Adam(
            network.parameters(),
            lr=params.get('adam_lr', lr),
            weight_decay=params.get('l2_regularization', weight_decay)
        )


# =====================================================
# Main training
# =====================================================
def train_data():
    # Load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'stacks')
    m1_stack = torch.load(os.path.join(data_dir, 'm1_stack_200.pt'), map_location=device)
    m2_stack = torch.load(os.path.join(data_dir, 'm2_stack_200.pt'), map_location=device)
    rho_stack = torch.load(os.path.join(data_dir, 'rho_stack_200.pt'), map_location=device)

    # Dataset and DataLoader
    dataset = TensorDataset(m1_stack[:num_train], m2_stack[:num_train], rho_stack[:num_train])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Model and optimizer
    model_rho = rhonet(act_parm=0.02).to(device)
    optimizer_params = {
        'optimizer': 'adam',
        'adam_lr': lr,
        'l2_regularization': weight_decay
    }
    optimizer_rho = use_optimizer(model_rho, optimizer_params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_rho,
                                                step_size=int(max_epoch/5),
                                                gamma=0.5)

    # Loss history and constants
    loss_hist = []
    s_list = torch.linspace(0, L - 1, steps=10 * L, device=device)
    rot_mat = (s_list[:, None].type(torch.complex64) @ multiplier[None, :]).exp()

    # Optionally resume from checkpoint
    checkpoint_path = os.path.join('checkpoints', 'rho_learn.pt')
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        start_epoch = resume_checkpoint(model_rho, optimizer_rho, checkpoint_path,
                                        device.index if device.type == 'cuda' else 0)

    # Training loop (epoch-based)
    for epoch in tqdm(range(start_epoch, max_epoch)):
        model_rho.train()
        epoch_loss = 0.0

        for m1_batch, m2_batch, rho_batch in dataloader:
            fft_rho_stack = fft(rho_batch)
            trans_rho_stack = rot_mat @ torch.diag_embed(fft_rho_stack)
            recon_rho = fft(model_rho(m1_batch, m2_batch))
            loss = diff_min(torch.norm(recon_rho[:, None, :] - trans_rho_stack, dim=2)).sum() / batch_size
            optimizer_rho.zero_grad()
            loss.backward()
            optimizer_rho.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_hist.append(avg_epoch_loss)

        scheduler.step()

        if ((epoch + 1) % show_iter) == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch + 1}/{max_epoch}, Loss: {avg_epoch_loss:.6f}, LR: {current_lr:.6e}")
            save_checkpoint(model_rho, optimizer_rho, checkpoint_path, epoch + 1)

    # Save final model
    save_checkpoint(model_rho, optimizer_rho, checkpoint_path, max_epoch)

    # Plot training curve
    plt.figure(figsize=(5, 5))
    plt.plot(loss_hist, label='loss')
    plt.legend()
    plt.title("Training Loss Curve")
    plt.savefig(os.path.join(data_dir, 'training_loss_curve.png'))
    plt.close()


def visualize_training():
    # Load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'stacks')
    m1_stack = torch.load(os.path.join(data_dir, 'm1_stack_200.pt'), map_location=device)
    m2_stack = torch.load(os.path.join(data_dir, 'm2_stack_200.pt'), map_location=device)
    rho_stack = torch.load(os.path.join(data_dir, 'rho_stack_200.pt'), map_location=device)

    # Model load
    model_rho = rhonet(act_parm=0.02).to(device)
    checkpoint_path = os.path.join('checkpoints', 'rho_learn.pt')
    resume_checkpoint(model_rho, None, checkpoint_path,
                                        device.index if device.type == 'cuda' else 0)

    # =====================================================
    # Post-training visualization on training set
    # =====================================================
    model_rho.eval()
    with torch.no_grad():
        # Randomly sample 10 examples from the training set
        idxs = torch.randint(num_train, (10,))
        recon_rho = model_rho(m1_stack[idxs], m2_stack[idxs])

        from src.utils import best_registration  # import here to avoid circular import

        for i, idx in enumerate(idxs):
            best_s, best_error, best_recon = best_registration(rho_stack[idx], recon_rho[i])

            fig, axes = plt.subplots(1, 2, figsize=(15, 4))

            # Left: reconstructed vs true rho
            axes[0].plot(best_recon.cpu().numpy(), label="Reconstructed")
            axes[0].plot(rho_stack[idx].cpu().numpy(), label="True")
            axes[0].set_title(f"Reconstructed shift distribution {i}")
            axes[0].legend()

            # Right: placeholder for underlying signal (if available)
            # sig_stack isn’t defined in your file, so this is skipped unless you add it.
            # Uncomment and modify if you have a signal tensor like 'sig_stack'
            # axes[1].plot(sig_stack[idx].cpu().numpy(), color='green')
            axes[1].set_title(f"Underlying signal (placeholder) {i}")

            fig.tight_layout()

            # Save figure
            save_path = os.path.join(data_dir, f"reconstruction_sample_{i}.png")
            plt.savefig(save_path)
            plt.close(fig)

        # Quantitative check on a larger subset
        indices = torch.randint(num_train, (5000,))
        fft_rho_stack = fft(rho_stack[indices])
        trans_rho_stack = rot_mat @ torch.diag_embed(fft_rho_stack)
        recon_rho = fft(model_rho(m1_stack[indices], m2_stack[indices]))
        loss_eval = diff_min(torch.norm(recon_rho[:, None, :] - trans_rho_stack, dim=2)).sum() / 5000

        print(f"\nEvaluation loss on 5000 random training samples: {loss_eval.item():.6f}")



if __name__ == "__main__":
    train_data()
