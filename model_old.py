import torch
from torch import nn
import torch.nn.functional as F
import math


# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#         self.copies = []
#         self.encoder = nn.ModuleList(
#             [
#                 nn.Conv2d(3, 64, 3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(64, 64, 3, padding=1),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2),
#                 nn.Conv2d(64, 128, 3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(128, 128, 3, padding=1),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2),
#                 nn.Conv2d(128, 256, 3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(256, 256, 3, padding=1),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2),
#                 nn.Conv2d(256, 512, 3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(512, 512, 3, padding=1),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2),
#                 nn.Conv2d(512, 1024, 3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(1024, 1024, 3, padding=1),
#                 nn.ReLU(),
#             ]
#         )
#         self.decoder = nn.ModuleList(
#             [
#                 nn.ConvTranspose2d(1024, 512, 2, stride=2),
#                 nn.Conv2d(1024, 512, 3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(512, 512, 3, padding=1),
#                 nn.ReLU(),
#                 nn.ConvTranspose2d(512, 256, 2, stride=2),
#                 nn.Conv2d(512, 256, 3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(256, 256, 3, padding=1),
#                 nn.ReLU(),
#                 nn.ConvTranspose2d(256, 128, 2, stride=2),
#                 nn.Conv2d(256, 128, 3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(128, 128, 3, padding=1),
#                 nn.ReLU(),
#                 nn.ConvTranspose2d(128, 64, 2, stride=2),
#                 nn.Conv2d(128, 64, 3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(64, 64, 3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(64, 3, 1),
#                 nn.Sigmoid(),
#             ]
#         )

#     def forward(self, x):
#         for i, layer in enumerate(self.encoder):
#             x = layer(x)
#             if (
#                 i < len(self.encoder) - 1
#                 and layer.__class__ == nn.ReLU
#                 and self.encoder[i + 1].__class__ == nn.MaxPool2d
#             ):
#                 self.copies.append(x)
#         for i, layer in enumerate(self.decoder):
#             if layer.__class__ == nn.ConvTranspose2d:
#                 x = layer(x)
#                 x = torch.cat((x, self.copies.pop()), 1)
#             else:
#                 x = layer(x)
#         return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.encoder_outs = []
        
        self.encoder = nn.ModuleList(
            [
                self.double_conv(3, 64),
                nn.MaxPool2d(kernel_size=2, stride=2),
                self.double_conv(64, 128),
                nn.MaxPool2d(kernel_size=2, stride=2),
                self.double_conv(128, 256),
                nn.MaxPool2d(kernel_size=2, stride=2),
                self.double_conv(256, 512),
                nn.MaxPool2d(kernel_size=2, stride=2),
                self.double_conv(512, 1024),
            ]
        )
        
        self.decoder = nn.ModuleList(
            [
                nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                self.double_conv(1024, 512),
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                self.double_conv(512, 256),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                self.double_conv(256, 128),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                self.double_conv(128, 64),
                nn.Conv2d(64, 3, kernel_size=1),
                nn.Sigmoid()
            ]
        )
        
        
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
        
    def center_crop(self, x, target):
        target_size = target.size()[2]
        x_size = x.size()[2]
        delta = x_size - target_size
        delta = delta // 2
        return x[:, :, delta:x_size-delta, delta:x_size-delta]
        
    def forward(self, x):
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool2d):
                self.encoder_outs.append(x)
            x = layer(x)

        for layer in self.decoder:
            x = layer(x)
            if isinstance(layer, nn.ConvTranspose2d):
                enc_out = self.encoder_outs.pop()
                enc_out = self.center_crop(enc_out, x)
                x = torch.cat([x, enc_out], dim=1)
        x = 2 * x - 1
        return x
    



class GaussianNoiseScheduler():
    def __init__(self, T, start=0.0001, end=0.02):
        super(GaussianNoiseScheduler, self).__init__()
        
        self.T = T # number of timesteps
        self.start = start
        self.end = end
        self.betas = self.linear_beta_schedule(T)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    
    def linear_beta_schedule(self, timesteps):
        """
        timesteps: number of timesteps
        start: start value of beta
        end: end value of beta
        
        Returns a linear schedule of beta values
        """
        return torch.linspace(self.start, self.end, timesteps)
    
    def get_index_from_list(self, vals, t, x_shape):
        """
        vals: tensor of values
        t: tensor of indices
        x_shape: shape of the input tensor
        
        Returns the values from vals at the indices in t
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_sample(self, x_0, t):
        """
        x_0: input tensor (should usually be an image)
        t: desired timestep
        
        Returns a noisy version of x_0 at timestep t, along with the noise which can be used for reconstruction
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        return (sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise).squeeze(0), noise.squeeze(0)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


if __name__ == "__main__":
    # test the model
    model = UNet()
    print(model)
    x = torch.randn(1, 3, 768, 768)
    out = model(x)
    print(out.shape)
    
