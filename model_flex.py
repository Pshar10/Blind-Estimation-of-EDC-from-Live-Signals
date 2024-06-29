import torch
import torch.nn as nn


"""This script has implementation of model architecture that has flexible layers, kernel size etc"""

class R2Dnet(nn.Module):

    def __init__(self, num_conv_layers=9):
        super(R2Dnet, self).__init__()

        self.device = torch.device("cuda")
        self.activation = nn.LeakyReLU(0.0)
        self.dropout = nn.Dropout(0.2)


        # Hyperparameters
        self.num_conv_layers = num_conv_layers
        self.kernel_sizes = [13, 13, 13, 7, 7, 7, 7, 7, 7]  # Default kernel sizes
        self.padding_sizes = [6, 6, 6, 3, 3, 3, 3, 3, 3]   # Default padding sizes
        self.pooling_factors = [5, 5, 5, 2, 2, 2, 2, 2, 1]  # Default pooling factors

        # Base Network - Convolutional and Pooling layers
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.bnorm = nn.ModuleList()
        in_channels = 1
        for i in range(self.num_conv_layers):
            out_channels = 64 * (i + 1)
            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_sizes[i], padding=self.padding_sizes[i])
            pool_layer = nn.MaxPool1d(self.pooling_factors[i])
            bnorm = nn.BatchNorm1d(out_channels)
            self.convs.append(conv_layer)
            self.pools.append(pool_layer)
            self.bnorm.append(bnorm)
            in_channels = out_channels

        # Calculate the size for the linear layer dynamically
        self.linear_input_size = self.calculate_linear_input_size()
        self.input = nn.Linear(self.linear_input_size, self.linear_input_size//16)

        self.linears = nn.ModuleList([nn.Linear(self.linear_input_size//16, self.linear_input_size//64)])

        # T_vals, A_vals, Noise, Sample length layers
        self.final1_t = nn.Linear(self.linear_input_size//64, 50)
        self.final2_t = nn.Linear(50, 1)
        self.final1_a = nn.Linear(self.linear_input_size//64, 50)
        self.final2_a = nn.Linear(50, 1)
        self.final1_n = nn.Linear(self.linear_input_size//64, 50)
        self.final2_n = nn.Linear(50, 1)
        self.final1_s = nn.Linear(self.linear_input_size//64, 50)
        self.final2_s = nn.Linear(50, 1)
    
    def calculate_linear_input_size(self, initial_input_size=16384):
        current_size = initial_input_size
        for i in range(self.num_conv_layers):
            # Apply convolution
            current_size = current_size + 2 * self.padding_sizes[i] - self.kernel_sizes[i] + 1
            # Apply pooling
            current_size = current_size // self.pooling_factors[i]
        # Multiply by the number of output channels from the last conv layer
        return current_size * self.convs[-1].out_channels
    

    def forward(self, Rs):
        x = Rs.unsqueeze(1)
        for conv, pool, norm in zip(self.convs, self.pools,self.bnorm):
            x = self.activation(norm(pool((conv(x)))))

        x = self.activation(self.input(self.dropout(x.view(Rs.shape[0], -1))))
        for layer in self.linears:
            x = self.activation(layer(x))

        # T_vals, A_vals, Noise, Sample length calculations
        t = torch.pow(self.activation(self.final2_t(self.activation(self.final1_t(x)))), 2.0) + 0.01
        a = torch.pow(self.activation(self.final2_a(self.activation(self.final1_a(x)))), 2.0) + 1e-16
        n_exponent = self.final2_n(self.activation(self.final1_n(x)))
        n_samples = self.final2_s(self.activation(self.final1_s(x)))

        return t, a, n_exponent, n_samples
