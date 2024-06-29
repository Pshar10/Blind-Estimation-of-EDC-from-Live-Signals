import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader_s2ir import R2DDataset
from torch.utils.tensorboard import SummaryWriter


"""Implementation of paper :  A. Ratnarajah, I. Ananthabhotla, V. K. Ithapu, P. Hoffmann, D. Manocha, and P. Calamia, “Towards Improved Room Impulse Response Estimation for Speech Recognition.” arXiv, Mar. 19, 2023. Accessed: Sep. 22, 2023. [Online]. Available: http://arxiv.org/abs/2211.04473
"""
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

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=512, kernel_size=8193, stride=256, padding=4096),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=41, stride=4, padding=20),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.upblock1 = UpBlock(1024, 512, 4)
        self.upblock2 = UpBlock(512, 256, 4)
        self.upblock3 = UpBlock(256, 128, 4)
        self.upblock4 = UpBlock(128, 64, 2)
        self.upblock5 = UpBlock(64, 64, 2)
        self.final = nn.Sequential(
            nn.ConvTranspose1d(64, 1, kernel_size=41, stride=1, padding=20),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.upblock1(x)
        x = self.upblock2(x)
        x = self.upblock3(x)
        x = self.upblock4(x)
        x = self.upblock5(x)
        x = self.upblock5(x)
        x = self.upblock5(x)
        x = self.final(x)
        return x

class COND_NET(nn.Module): 

    def __init__(self):
        super(COND_NET, self).__init__()
        self.fc = nn.Linear(512, 512, bias=True)
        self.relu = nn.PReLU()#nn.ReLU()

    def encode(self, Speech_samples):
        x = self.relu(self.fc(Speech_samples))
        return x


    def forward(self, embedding):
        c_code = self.encode(embedding)
        return c_code 


class Discriminator_RIR(nn.Module):
    def __init__(self, dis_dim=96):
        super(Discriminator_RIR, self).__init__()
        self.df_dim = dis_dim
        self.define_module()
        self.cond = COND_NET()

    def define_module(self):
        ndf = self.df_dim
        kernel_length = 41

        self.encode_EDC = nn.Sequential(
            nn.Conv1d(1, ndf, kernel_length, stride=2, padding=20, bias=False),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 96 x 512
            nn.Conv1d(ndf, ndf * 2, kernel_length, stride=2, padding=20, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (96*2) x 256
            nn.Conv1d(ndf * 2, ndf * 4, kernel_length, stride=2, padding=20, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (96*4) x 128
            nn.Conv1d(ndf * 4, ndf * 8, kernel_length, stride=2, padding=20, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (96*8) x 64
            nn.Conv1d(ndf * 8, ndf * 16, kernel_length, stride=2, padding=20, bias=False),
            nn.BatchNorm1d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (96*16) x 16
        )

        self.convd1d = nn.ConvTranspose1d(ndf * 16, ndf // 2, kernel_size=kernel_length, stride=1, padding=20)

        self.outlogits = nn.Sequential(
            nn.Conv1d(ndf // 2, 1, kernel_size=16, stride=16, padding=6),
            nn.Conv1d(1, 1, kernel_size=16, stride=8, padding=6),
            nn.Conv1d(1, 1, kernel_size=16, stride=4, padding=6),
            # nn.Linear(32,1),
            nn.Sigmoid()
        )
        
    def forward(self, EDC, speech):
        
        # print(speech.shape)
        # print(RIRs.shape)
        speech = self.cond(speech)
        concatenated_tensor = torch.cat((speech, EDC), dim=2)
        EDC_embedding = self.encode_EDC(concatenated_tensor)
        EDC_embedding = self.convd1d(EDC_embedding)
        output = self.outlogits(EDC_embedding)
        return output
    

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def save_model(model, filename):
    torch.save(model.state_dict(), filename)



###########END OF CLASSES#######
log_dir = f'/home/prsh7458/work/R2D/S2IR/'
tb_writer = SummaryWriter(log_dir=log_dir)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


generator = Generator().to(device)
discriminator = Discriminator_RIR().to(device)



generator_lr = 8e-5
discriminator_lr = 8e-5
lr_decay_step = 40
# optimizerD = \
#     optim.Adam(netD.parameters(),
#                lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))


optimizer_D = optim.RMSprop(discriminator.parameters(),
                lr=8e-5)
netG_para = []
for p in generator.parameters():
    if p.requires_grad:
        netG_para.append(p)


# optimizerG = optim.Adam(netG_para,
#                         lr=cfg.TRAIN.GENERATOR_LR,
#                         betas=(0.5, 0.999))



optimizer_G = optim.RMSprop(netG_para,
                        lr=8e-5)


# optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()

num_epochs = 100  # Adjust as needed
edc_len = 16384  # Adjust as per your data dimensions



dataset_synthdecays = R2DDataset()
trainloader = DataLoader(dataset_synthdecays, batch_size=128, shuffle=True) 

loss_fn = nn.L1Loss()
scheduler_d = optim.lr_scheduler.StepLR(optimizer_D, step_size=40, gamma=0.7)
scheduler_g = optim.lr_scheduler.StepLR(optimizer_G, step_size=40, gamma=0.7)
generator.train()
discriminator.train()
import time
for epoch in range(num_epochs):


    start_t = time.time()

    if epoch % lr_decay_step == 0 and epoch > 0:
        generator_lr *= 0.7#0.5
        for param_group in optimizer_G.param_groups:
            param_group['lr'] = generator_lr
        discriminator_lr *= 0.7#0.5
        for param_group in optimizer_D.param_groups:
            param_group['lr'] = discriminator_lr

            
    for i, data_frame in enumerate(trainloader):  # Replace real_data_loader with your data loader

        

        Reverb_speech,edcs,time_axis_interpolated,n_vals,n_samples = data_frame


        edcs = edcs.unsqueeze(1).to(device)
        Reverb_speech = Reverb_speech.unsqueeze(1).to(device)


        batch_size = Reverb_speech.size(0)
        # print(Reverb_speech.shape)

        # Prepare labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Train with real data
        real_data = edcs
        # print("real_data",real_data.shape)
        output_real = discriminator(real_data,Reverb_speech[:,:,:512])
        output_real = output_real
        loss_real = criterion(output_real.squeeze(1), real_labels)

        # Generate fake data (no gradient calculation needed here)
        fake_data = generator(Reverb_speech)

            # print("fake data",fake_data.shape)

        # Train with fake data
        output_fake = discriminator(fake_data,Reverb_speech[:,:,:512]).view(-1)

        # print(output_fake.shape)
        # Check if there's a need to squeeze and then apply BCELoss
        if output_fake.dim() > 1:
            output_fake = output_fake.squeeze(1)
        loss_fake = criterion(output_fake.unsqueeze(1), fake_labels)

        # Total discriminator loss
        loss_D = ((loss_real + loss_fake)/2)*5
        loss_D.backward()
        optimizer_D.step()
        scheduler_d.step()

        for p in range(2):
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate fake data again (with gradients for generator training)
            fake_data = generator(Reverb_speech)

            # Train generator (trying to fool discriminator)
            output = discriminator(fake_data,Reverb_speech[:,:,:512]).view(-1)
            loss_G = criterion(output.unsqueeze(1), real_labels)
            loss_l1  =loss_fn (real_data,fake_data)

            if not torch.is_tensor(loss_G) or loss_G.dim() != 0:
                raise ValueError("Generator BCE loss is not a scalar.")

            loss_l1 = loss_fn(real_data, fake_data)  # MSE loss

            # Ensure that loss_l1 is a scalar
            if not torch.is_tensor(loss_l1) or loss_l1.dim() != 0:
                raise ValueError("Generator MSE loss is not a scalar.")
            loss_overall_g = (loss_G+loss_l1)*5

            loss_overall_g.backward()
            optimizer_G.step()
            
            scheduler_g.step()

        # save_model(generator, '/home/prsh7458/work/R2D/S2IR/' + "S2EDC" + '.pth')
        # save_model(discriminator, '/home/prsh7458/work/R2D/S2IR/' + "S2EDC" + '.pth')

        print(f"Epoch [{epoch+1}/{num_epochs}] Batch {i}/{len(trainloader)} \
              Loss D: {loss_D.item()}, Loss G: {loss_G.item()}")
        
        
        
        tb_writer.add_scalar('Loss/loss_G', loss_G, (epoch - 1) * len(trainloader) + i)
        tb_writer.add_scalar('Loss/loss_D', loss_D, (epoch - 1) * len(trainloader) + i)
        tb_writer.add_scalar('Loss/MAE_step', loss_l1, (epoch - 1) * len(trainloader) + i)
        tb_writer.flush()