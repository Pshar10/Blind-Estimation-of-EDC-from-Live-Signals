import torch
import torch.nn as nn

"""This script has implementation of model architecture that has fixed layers, kernel size etc"""


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(UpBlock, self).__init__()

        kernel_size = 41
        padding = 21 - stride // 2
        output_padding = 1

        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.BatchNorm1d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)
    

class R2Dnet(nn.Module):
    def __init__(self, num_conv_layers=9):
        super(R2Dnet, self).__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=512, kernel_size=8193, stride=256, padding=4096),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=41, stride=4, padding=20),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=41, stride=4, padding=20),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.upblock1 = UpBlock(1024, 512, 4)
        self.upblock2 = UpBlock(512, 256, 4)
        self.upblock3 = UpBlock(256, 128, 4)
        #self.upblock4 = UpBlock(128, 64, 2)
        #self.upblock5 = UpBlock(64, 64, 2)
        self.final = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=1, kernel_size=41, stride=1, padding=20),
            nn.LeakyReLU(negative_slope=0.2) ,nn.Tanh()
        )
        

        self.final1_t = nn.Linear(256, 50)
        self.final2_t = nn.Linear(50, 1)
        
        self.final1_a = nn.Linear(256, 50)
        self.final2_a = nn.Linear(50, 1)
        
        self.final1_n = nn.Linear(256, 50)
        self.final2_n = nn.Linear(50, 1)

        # Sample length
        self.final1_s = nn.Linear(256, 50)
        self.final2_s = nn.Linear(50, 1)
        

        
    def forward(self, Rs):
        # print(x.unsqueeze(1).shape)
        x = self.layer1(Rs.unsqueeze(1))
        x = self.layer2(x)
        x = self.layer3(x)        
        x = self.upblock1(x)
        x = self.upblock2(x)
        x = self.upblock3(x)
        x = self.final(x)
        x = x.view(x.shape[0], -1)
        
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
        
        return t,a,n_exponent,n_samples




    

# class R2Dnet(nn.Module):
#     def __init__(self, num_conv_layers=9):
#         super(R2Dnet, self).__init__()
#         self.activation = nn.LeakyReLU(negative_slope=0.5)

#         self.layer1 = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=512, kernel_size=8193, stride=256, padding=4096),
#             nn.LeakyReLU(negative_slope=0.5)
#         )

#         self.layer2 = nn.Sequential(
#             nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=41, stride=4, padding=20),
#             nn.BatchNorm1d(1024),
#             nn.LeakyReLU(negative_slope=0.5)
#         )
#         self.layer3 = nn.Sequential(
#             nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=41, stride=4, padding=20),
#             nn.BatchNorm1d(1024),
#             nn.LeakyReLU(negative_slope=0.5)
#         )
        
#         self.upblock1 = UpBlock(1024, 512, 4)
#         self.upblock2 = UpBlock(512, 256, 4)
#         self.upblock3 = UpBlock(256, 128, 4)
#         #self.upblock4 = UpBlock(128, 64, 2)
#         #self.upblock5 = UpBlock(64, 64, 2)
#         self.final = nn.Sequential(
#             nn.ConvTranspose1d(in_channels=128, out_channels=1, kernel_size=41, stride=1, padding=20),
#             nn.LeakyReLU(negative_slope=0.5) 
#         ) #,nn.Tanh()
        

#         #self.final1_t = nn.Linear(256, 50)
#         self.final1_t = nn.Linear(512, 1)
#         #self.final2_t = nn.Linear(50, 1)
        
#         #self.final1_a = nn.Linear(256, 50)
#         self.final1_a = nn.Linear(512, 1)
#         #self.final2_a = nn.Linear(50, 1)
        
#         #self.final1_n = nn.Linear(256, 50)
#         self.final1_n = nn.Linear(512, 1)
#         #self.final2_n = nn.Linear(50, 1)

#         # Sample length
#         #self.final1_s = nn.Linear(256, 50)
#         self.final1_s = nn.Linear(512, 1)
#         #self.final2_s = nn.Linear(50, 1)
        

        
#     def forward(self, Rs):
#         # print(x.unsqueeze(1).shape)
#         x = self.layer1(Rs.unsqueeze(1))
#         x = self.layer2(x)
#         x = self.layer3(x)        
#         x = self.upblock1(x)
#         x = self.upblock2(x)
#         x = self.upblock3(x)
#         x = self.final(x)
#         x = x.view(x.shape[0], -1)
        
#         # T_vals
#         t = self.activation(self.final1_t(x))
        
#         #t = torch.pow(self.final2_t(t), 2.0) + 0.01
#         t = torch.pow((t), 2.0) + 0.01
#         # A_vals
#         a = self.activation(self.final1_a(x))
        
#         #a = torch.pow(self.final2_a(a), 2.0) + 1e-16
#         a = torch.pow((a), 2.0) + 1e-16

#         # Noise
#         n_exponent = self.activation(self.final1_n(x))
        
#         #n_exponent = self.final2_n(n_exponent)

#         # Samples  
#         n_samples = self.activation(self.final1_s(x))
#         #n_samples = self.final2_s(n_samples)
        
#         return t,a,n_exponent,n_samples
    
