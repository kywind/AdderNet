import torch
import torch.nn as nn
import torchvision.models as models

import sys

class Vgg16Deconv(nn.Module):
    """
    vgg16 transpose convolution network architecture
    """
    def __init__(self):
        super(Vgg16Deconv, self).__init__()

        self.features = nn.Sequential(
            # deconv1
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, padding=1),

            # deconv2
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, padding=1),
            
            # deconv3
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, padding=1),
            
            # deconv4
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            
            # deconv5
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, padding=1)    
        )

        self.conv2deconv_indices = {
                0:30, 2:28, 5:25, 7:23,
                10:20, 12:18, 14:16, 17:13,
                19:11, 21:9, 24:6, 26:4, 28:2
                }

        self.unpool2pool_indices = {
                26:4, 21:9, 14:16, 7:23, 0:30
                }

        self.init_weight()

    def init_weight(self):
        vgg16_pretrained = models.vgg16(pretrained=True)
        for idx, layer in enumerate(vgg16_pretrained.features):
            if isinstance(layer, nn.Conv2d):
                self.features[self.conv2deconv_indices[idx]].weight.data = layer.weight.data
                #self.features[self.conv2deconv_indices[idx]].bias.data\
                # = layer.bias.data
        
    def forward(self, x, layer, activation_idx, pool_locs):
        if layer in self.conv2deconv_indices:
            start_idx = self.conv2deconv_indices[layer]
        else:
            raise ValueError('layer is not a conv feature map')

        for idx in range(start_idx, len(self.features)):
            if isinstance(self.features[idx], nn.MaxUnpool2d):
                x = self.features[idx]\
                (x, pool_locs[self.unpool2pool_indices[idx]])
            else:
                x = self.features[idx](x)
        return x


class BNTranspose(nn.Module):
    def __init__(self, num_features):
        super(BNTranspose, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, input):
        return input * self.weight.reshape(-1,1,1) + self.bias.reshape(-1,1,1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.num_features) + ')'


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvTranspose, self).__init__()
        # self.num_features = num_features
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bias = nn.Parameter(torch.Tensor(in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, input):
        return self.conv(input - self.bias[:,None,None])



class Vgg16Deconv_bn(nn.Module):
    """
    vgg16 transpose convolution network architecture
    """
    def __init__(self):
        super(Vgg16Deconv_bn, self).__init__()

        self.features = nn.Sequential(
            # deconv1
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            BNTranspose(512),
            ConvTranspose(512, 512, 3, padding=1),
            nn.ReLU(),
            BNTranspose(512),
            ConvTranspose(512, 512, 3, padding=1),
            nn.ReLU(),
            BNTranspose(512),
            ConvTranspose(512, 512, 3, padding=1),

            # deconv2
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            BNTranspose(512),
            ConvTranspose(512, 512, 3, padding=1),
            nn.ReLU(),
            BNTranspose(512),
            ConvTranspose(512, 512, 3, padding=1),
            nn.ReLU(),
            BNTranspose(512),
            ConvTranspose(512, 256, 3, padding=1),
            
            # deconv3
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            BNTranspose(256),
            ConvTranspose(256, 256, 3, padding=1),
            nn.ReLU(),
            BNTranspose(256),
            ConvTranspose(256, 256, 3, padding=1),
            nn.ReLU(),
            BNTranspose(256),
            ConvTranspose(256, 128, 3, padding=1),
            
            # deconv4
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            BNTranspose(128),
            ConvTranspose(128, 128, 3, padding=1),
            nn.ReLU(),
            BNTranspose(128),
            ConvTranspose(128, 64, 3, padding=1),
            
            # deconv5
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            BNTranspose(64),
            ConvTranspose(64, 64, 3, padding=1),
            nn.ReLU(),
            BNTranspose(64),
            ConvTranspose(64, 3, 3, padding=1)    
        )
        # 43
        # self.conv2deconv_indices = {
        #         0:30, 2:28, 5:25, 7:23,
        #         10:20, 12:18, 14:16, 17:13,
        #         19:11, 21:9, 24:6, 26:4, 28:2
        #         }

        # self.unpool2pool_indices = {
        #         26:4, 21:9, 14:16, 7:23, 0:30
        #         }

        self.init_weight()

    def init_weight(self):
        vgg16_pretrained = models.vgg16_bn(pretrained=True)
        for idx, layer in enumerate(vgg16_pretrained.features):
            if isinstance(layer, nn.Conv2d):
                self.features[43 - idx].conv.weight.data = layer.weight.data
                self.features[43 - idx].bias.data = layer.bias.data
            if isinstance(layer, nn.BatchNorm2d):
                # self.features[43 - idx].weight.data = layer.weight.data
                # self.features[43 - idx].bias.data = layer.bias.data
                # self.features[43 - idx].running_mean.data = layer.running_mean.data
                # self.features[43 - idx].running_var.data = layer.running_var.data
                # self.features[43 - idx].eps = layer.eps
                t = (torch.sqrt(layer.running_var + layer.eps) / layer.weight).data
                self.features[43 - idx].weight.data = t
                self.features[43 - idx].bias.data = (-layer.bias * t + layer.running_mean).data
        
    def forward(self, x, layer, activation_idx, pool_locs):
        if layer in [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]:
            start_idx = 43 - layer
        else:
            raise ValueError('layer is not a conv feature map')

        for idx in range(start_idx, len(self.features)):
            if isinstance(self.features[idx], nn.MaxUnpool2d):
                x = self.features[idx](x, pool_locs[43 - idx])
            elif isinstance(self.features[idx], nn.ReLU):
                pass
            else:
                x = self.features[idx](x)
        return x

