import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        n_channels = opt.gan.n_channels
        assert opt.image_size[0] == opt.image_size[1] == 64
        self.condition = opt.gan.condition
        spec_norm = torch.nn.utils.spectral_norm if opt.gan.spec_norm else lambda x: x
        self.embedding = spec_norm(nn.Embedding(opt.data.num_classes, n_channels * 8))
        self.linear = spec_norm(nn.Linear(n_channels * 8, 1))
        self.layers = nn.Sequential(
            # [B,4,64,64]
            spec_norm(nn.Conv2d(4, n_channels, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # [B,C,32,32]
            spec_norm(nn.Conv2d(n_channels, n_channels * 2, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(n_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # [B,C*2,16,16]
            spec_norm(nn.Conv2d(n_channels * 2, n_channels * 4, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(n_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # [B,C*4,8,8]
            spec_norm(nn.Conv2d(n_channels * 4, n_channels * 8, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(n_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        if self.condition:
            torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, input, target):
        feature = self.layers(input)
        if self.condition:
            feature = torch.sum(feature, dim=(2,3))
            output = self.linear(feature)
            y = self.embedding(target)
            output += torch.sum(y * feature, dim=1, keepdim=True)
        else: 
            feature = torch.sum(feature, dim=(2,3))
            output = self.linear(feature)
        return output