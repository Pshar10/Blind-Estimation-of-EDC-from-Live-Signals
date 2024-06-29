import torch
import torch.nn as nn

"""A simple implementation of model architecture of R2DNet CNN"""
class R2Dnet(nn.Module):

    def __init__(self):
        super(R2Dnet, self).__init__()

        self.device = torch.device("cuda")
        self.activation = nn.LeakyReLU(0.0)
        self.dropout = nn.Dropout(0.0)

        # Base Network
        self.conv1 = nn.Conv1d(1, 64, kernel_size=13, padding=6) #64,16384
        self.maxpool1 = nn.MaxPool1d(5) #64,3276
        self.conv2 = nn.Conv1d(64, 64*2, kernel_size=13, padding=6) #64*2,3276
        self.maxpool2 = nn.MaxPool1d(5) #64*2,655
        self.conv3 = nn.Conv1d(64*2, 64*3, kernel_size=13, padding=6) #64*3,655
        self.maxpool3 = nn.MaxPool1d(5) #64*3,131
        self.conv4 = nn.Conv1d(64*3, 64*4, kernel_size=7, padding=3) #64*4,131
        self.maxpool4 = nn.MaxPool1d(2) #64*4,65
        self.conv5 = nn.Conv1d(64*4, 64*4, kernel_size=7, padding=3) #64*4,65
        self.maxpool5 = nn.MaxPool1d(2) #64*4,32
        self.conv6 = nn.Conv1d(64*4, 64*4, kernel_size=7, padding=3) #64*4,32
        self.maxpool6 = nn.MaxPool1d(2) #64*4,16
        self.conv7 = nn.Conv1d(64*4, 64*4, kernel_size=7, padding=3) #64*4,16
        self.maxpool7 = nn.MaxPool1d(2) #64*4,8

        self.input = nn.Linear(256*8, 1024)
        

        self.linears = nn.ModuleList([nn.Linear(round(1024 * (1**i)),
                                                round(400 * (1**(i+1)))) for i in range(1)])

        # T_vals
        self.final1_t = nn.Linear(400, 50)
        self.final2_t = nn.Linear(50, 1)

        # A_vals
        self.final1_a = nn.Linear(400, 50)
        self.final2_a = nn.Linear(50, 1)

        # Noise
        self.final1_n = nn.Linear(400, 50)
        self.final2_n = nn.Linear(50, 1)


        # Sample length
        self.final1_s = nn.Linear(400, 50)
        self.final2_s = nn.Linear(50, 1)



    def forward(self, Rs):
        """
        Args:

        Returns:
        """

        # Base network
        x = self.maxpool1(self.activation(self.conv1(Rs.unsqueeze(1))))
        x = self.maxpool2(self.activation(self.conv2(x)))
        x = self.maxpool3(self.activation(self.conv3(x)))
        x = self.maxpool4(self.activation(self.conv4(x)))
        x = self.maxpool5(self.activation(self.conv5(x)))
        x = self.maxpool6(self.activation(self.conv6(x)))
        x = self.maxpool7(self.activation(self.conv7(x)))
        x = self.activation(self.input(self.dropout(x.view(Rs.shape[0], -1))))
        for layer in self.linears:
            x = layer(x)
            x = self.activation(x)

        # T_vals
        t = self.activation(self.final1_t(x))
        t = torch.pow(self.final2_t(t), 2.0) + 0.01

        # A_vals
        a = self.activation(self.final1_a(x))
        a = torch.pow(self.final2_a(a), 2.0) + 1e-16

        # Noise
        n_exponent = self.activation(self.final1_n(x))
        n_exponent = self.final2_n(n_exponent)

        # Samples
        n_samples = self.activation(self.final1_s(x))
        n_samples = self.final2_s(n_samples)

        # print("data output from  net : ",t.requires_grad, a.requires_grad, n_exponent.requires_grad)

        return t, a, n_exponent,n_samples


