import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

""" Implementation of the paper Filtered Noise Shaping for Time Domain Room Impulse Response Estimation From Reverberant Speech. [Online]. Available: https://github.com/kyungyunlee/fins
"""

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super(EncoderBlock, self).__init__()
        if use_batchnorm:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(out_channels, track_running_stats=True),
                nn.PReLU(),
            )
            self.skip_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm1d(out_channels, track_running_stats=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=2, padding=7),
                nn.PReLU(),
            )
            self.skip_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
            )

    def forward(self, x):
        out = self.conv(x)
        skip_out = self.skip_conv(x)
        skip_out = out + skip_out
        return skip_out
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        block_list = []
        channels = [1, 32, 32, 64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512]

        for i in range(0, len(channels) - 1):
            if (i + 1) % 3 == 0:
                use_batchnorm = True
            else:
                use_batchnorm = False
            in_channels = channels[i]
            out_channels = channels[i + 1]
            curr_block = EncoderBlock(in_channels, out_channels, use_batchnorm)
            block_list.append(curr_block)

        self.encode = nn.Sequential(*block_list)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        b, c, l = x.size()
        out = self.encode(x)
        out = self.pooling(out)
        out = out.view(b, -1)
        out = self.fc(out)
        return out
class UpsampleNet(nn.Module):
    def __init__(self, input_size, output_size, upsample_factor):
        super(UpsampleNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.upsample_factor = upsample_factor

        layer = nn.ConvTranspose1d(
            input_size,
            output_size,
            upsample_factor * 2,
            upsample_factor,
            padding=upsample_factor // 2,
        )
        nn.init.orthogonal_(layer.weight)
        self.layer = spectral_norm(layer)

    def forward(self, inputs):
        outputs = self.layer(inputs)
        outputs = outputs[:, :, : inputs.size(-1) * self.upsample_factor]
        return outputs


class ConditionalBatchNorm1d(nn.Module):

    """Conditional Batch Normalization"""

    def __init__(self, num_features, condition_length):
        super().__init__()

        self.num_features = num_features
        self.condition_length = condition_length
        self.norm = nn.BatchNorm1d(num_features, affine=True, track_running_stats=True)

        self.layer = spectral_norm(nn.Linear(condition_length, num_features * 2))
        self.layer.weight.data.normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.layer.bias.data.zero_()  # Initialise bias at 0

    def forward(self, inputs, noise):
        outputs = self.norm(inputs)
        gamma, beta = self.layer(noise).chunk(2, 1)
        gamma = gamma.view(-1, self.num_features, 1)
        beta = beta.view(-1, self.num_features, 1)

        outputs = gamma * outputs + beta

        return outputs
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_factor, condition_length):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condition_length = condition_length
        self.upsample_factor = upsample_factor

        # Block A
        self.condition_batchnorm1 = ConditionalBatchNorm1d(in_channels, condition_length)

        self.first_stack = nn.Sequential(
            nn.PReLU(),
            UpsampleNet(in_channels, in_channels, upsample_factor),
            nn.Conv1d(in_channels, out_channels, kernel_size=15, dilation=1, padding=7),
        )

        self.condition_batchnorm2 = ConditionalBatchNorm1d(out_channels, condition_length)

        self.second_stack = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, dilation=1, padding=7),
        )

        self.residual1 = nn.Sequential(
            UpsampleNet(in_channels, in_channels, upsample_factor),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0),
        )
        # Block B
        self.condition_batchnorm3 = ConditionalBatchNorm1d(out_channels, condition_length)

        self.third_stack = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, dilation=4, padding=28),
        )

        self.condition_batchnorm4 = ConditionalBatchNorm1d(out_channels, condition_length)

        self.fourth_stack = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, dilation=8, padding=56),
        )

    def forward(self, enc_out, condition):
        inputs = enc_out

        outputs = self.condition_batchnorm1(inputs, condition)
        outputs = self.first_stack(outputs)
        outputs = self.condition_batchnorm2(outputs, condition)
        outputs = self.second_stack(outputs)

        residual_outputs = self.residual1(inputs) + outputs

        outputs = self.condition_batchnorm3(residual_outputs, condition)
        outputs = self.third_stack(outputs)
        outputs = self.condition_batchnorm4(outputs, condition)
        outputs = self.fourth_stack(outputs)

        outputs = outputs + residual_outputs

        return outputs

class Decoder(nn.Module):
    def __init__(self, num_filters, cond_length):
        super(Decoder, self).__init__()

        self.preprocess = nn.Conv1d(1, 512, kernel_size=15, padding=7)
        self.linear_layer = nn.Linear(400, 1)
        self.activation = nn.PReLU()
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(512, 512, 1, cond_length),
                DecoderBlock(512, 512, 1, cond_length),
                DecoderBlock(512, 256, 1, cond_length),
                DecoderBlock(256, 256, 1, cond_length),
                DecoderBlock(256, 256, 1, cond_length),
                DecoderBlock(256, 128, 1, cond_length),
                #DecoderBlock(128, 64, 5, cond_length),
            ]
        )

        self.postprocess = nn.Sequential(nn.Conv1d(128, num_filters, kernel_size=15, padding=7))

        self.sigmoid = nn.Sigmoid()

    def forward(self, v, condition):
        inputs = self.preprocess(v)
        outputs = inputs
        for i, layer in enumerate(self.blocks):
            outputs = layer(outputs, condition)
        outputs = self.postprocess(outputs)

        t = outputs[:, 0:1]
        t = self.linear_layer(t)
        t = torch.pow(self.activation(t), 2.0) + 0.01
        
        a = outputs[:, 1:2]
        a = self.linear_layer(a)
        a = torch.pow(self.activation(a), 2.0) + 1e-16
        
        n= outputs[:, 2:3]
        n = self.linear_layer(n)
        n = self.activation(n)

        return t, a, n

    def forward(self, enc_out, condition):
        inputs = enc_out

        outputs = self.condition_batchnorm1(inputs, condition)
        outputs = self.first_stack(outputs)
        outputs = self.condition_batchnorm2(outputs, condition)
        outputs = self.second_stack(outputs)

        residual_outputs = self.residual1(inputs) + outputs

        outputs = self.condition_batchnorm3(residual_outputs, condition)
        outputs = self.third_stack(outputs)
        outputs = self.condition_batchnorm4(outputs, condition)
        outputs = self.fourth_stack(outputs)

        outputs = outputs + residual_outputs

        return outputs

class Decoder(nn.Module):
    def __init__(self, num_filters, cond_length):
        super(Decoder, self).__init__()

        self.preprocess = nn.Conv1d(1, 512, kernel_size=15, padding=7)
        self.linear_layer = nn.Linear(400, 1)
        self.activation = nn.PReLU()
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(512, 512, 1, cond_length),
                DecoderBlock(512, 512, 1, cond_length),
                DecoderBlock(512, 256, 1, cond_length),
                DecoderBlock(256, 256, 1, cond_length),
                DecoderBlock(256, 256, 1, cond_length),
                DecoderBlock(256, 128, 1, cond_length),
            ]
        )

        self.postprocess = nn.Sequential(nn.Conv1d(128, num_filters, kernel_size=15, padding=7))

        self.sigmoid = nn.Sigmoid()

    def forward(self, v, condition):
        inputs = self.preprocess(v)
        outputs = inputs
        for i, layer in enumerate(self.blocks):
            outputs = layer(outputs, condition)
        outputs = self.postprocess(outputs)

        t = outputs[:, 0:1]
        t = self.linear_layer(t)
        t = torch.pow(self.activation(t), 2.0) + 0.01
        
        a = outputs[:, 1:2]
        a = self.linear_layer(a)
        a = torch.pow(self.activation(a), 2.0) + 1e-16
        
        n= outputs[:, 2:3]
        n = self.linear_layer(n)
        n = self.activation(n)

        return t, a, n
    

class R2Dnet(nn.Module):

    def __init__(self):
        super(R2Dnet, self).__init__()
        self.batch_size = 128
        self.model_enc = Encoder().to("cuda" )
        self.decoder = Decoder(3, 16 + 128).to("cuda")

    def forward(self,Rs):
        # Initialize the model
        x = Rs.unsqueeze(1)
        batch_size, _, _ = x.size()
        z = self.model_enc(x)
        noise_condition = torch.randn((batch_size, 16)).to("cuda" )
        condition = torch.cat([z, noise_condition], dim=-1).to("cuda")
        decoder_input = nn.Parameter(torch.randn((1, 1, 400))).to("cuda")
        decoder_input = decoder_input.repeat(batch_size, 1, 1)
        t, a,n = self.decoder(decoder_input, condition)
        return t.squeeze(1),a.squeeze(1),n.squeeze(1),1