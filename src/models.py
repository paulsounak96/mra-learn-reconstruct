import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config.config import L

class rhonet(nn.Module):  # filters=32, kernel=15
    def __init__(self, input_dim=L, n_conv_filters=32, n_interm_channel=3, kernel_size=15, hidden_dim=256, output_dim=1, parm_limit=0.25, act_parm=0.02,):
        super(rhonet, self).__init__()

        # Activation function setting
        if (act_parm > 0) & (act_parm < 1):
            act_func = nn.LeakyReLU(act_parm)
        else:
            act_func = nn.Tanh()

        self.fc_inpdim = n_interm_channel * L
        self.pad = int((kernel_size - 1)/2)

        # CNN for m1
        self.cnn_m1 = nn.Sequential(
            nn.Conv1d(1, n_conv_filters, kernel_size=kernel_size, padding=self.pad, padding_mode='circular'),
            act_func,
            nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=kernel_size, padding=self.pad, padding_mode='circular'),
            act_func,
            nn.Conv1d(n_conv_filters, n_interm_channel, kernel_size=kernel_size, padding=self.pad, padding_mode='circular'),
            act_func,
            # nn.MaxPool1d(2),
        )

        # CNN for m2
        self.cnn_m2 = nn.Sequential(
            nn.Conv1d(input_dim, n_conv_filters, kernel_size=kernel_size, padding=self.pad, padding_mode='circular'),
            act_func,
            nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=kernel_size, padding=self.pad, padding_mode='circular'),
            act_func,
            nn.Conv1d(n_conv_filters, n_interm_channel, kernel_size=kernel_size, padding=self.pad, padding_mode='circular'),
            act_func,
            # nn.MaxPool1d(2),
        )

        # CNN for their combination
        self.cnn_combine =  nn.Sequential(
            nn.Conv1d(2*n_interm_channel, n_conv_filters, kernel_size=kernel_size, padding=self.pad, padding_mode='circular'),
            act_func,
            nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=kernel_size, padding=self.pad, padding_mode='circular'),
            act_func,
            nn.Conv1d(n_conv_filters, n_interm_channel, kernel_size=kernel_size, padding=self.pad, padding_mode='circular'),
            act_func,
            # nn.MaxPool1d(2),
        )

        # Fully connected layer in the end
        self.fc =  nn.Sequential(
            nn.Linear(self.fc_inpdim, hidden_dim),
            act_func,
            nn.Linear(hidden_dim, hidden_dim),
            act_func,
            nn.Linear(hidden_dim, hidden_dim),
            act_func,
            nn.Linear(hidden_dim, input_dim),
        )

        # Initialization of parameters
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.uniform_(m.weight, -parm_limit, parm_limit)
                nn.init.uniform_(m.bias, -parm_limit, parm_limit)

        self.cnn_m1.apply(init_weights)
        self.cnn_m2.apply(init_weights)
        self.cnn_combine.apply(init_weights)
        self.fc.apply(init_weights)

    def forward(self, first_moment, second_moment):
        out_m1 = self.cnn_m1(first_moment[:, None, :])
        # print(out_m1.shape)
        out_m2 = self.cnn_m2(second_moment)
        # print(out_m2.shape)
        out_combine = self.cnn_combine(torch.cat((out_m1, out_m2), dim=1))
        # print(out_combine.shape)
        out = self.fc(out_combine.reshape(-1, self.fc_inpdim))
        return F.softmax(out, dim=1)

class signet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, output_dim=1, act_parm=0.02):
        super(signet, self).__init__()

        if (act_parm > 0) & (act_parm < 1):
            act_func = nn.LeakyReLU(act_parm)
        else:
            act_func = nn.Tanh()

        self.r = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(act_parm),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(act_parm),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(act_parm),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.theta = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(act_parm),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(act_parm),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(act_parm),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.r(x) + self.theta(x)*1j # self.r(x) * torch.exp(self.theta(x)*torch.pi*1j) #