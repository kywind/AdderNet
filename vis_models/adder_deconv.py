import torch
import torch.nn as nn
import torchvision.models as models

import sys
sys.path.append('../')
import adder

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision

from collections import OrderedDict
from functools import partial


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
        return input * self.weight[:,None,None] + self.bias[:,None,None]

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.num_features) + ')'


class AdderDeconv(nn.Module):  # NOTE: this is DEPRECATED

    def __init__(self):
        super(AdderDeconv, self).__init__()

        self.deblock1 = nn.Sequential(
            nn.MaxUnpool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            adder.adder2d_deconv(32, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            adder.adder2d_deconv(32, 3, kernel_size=3, padding=1)
        )

        self.deblock2 = nn.Sequential(
            nn.MaxUnpool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            adder.adder2d_deconv(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            adder.adder2d_deconv(64, 32, kernel_size=3, padding=1)
        )

        self.deblock3 = nn.Sequential(
            nn.MaxUnpool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            adder.adder2d_deconv(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            adder.adder2d_deconv(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            adder.adder2d_deconv(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            adder.adder2d_deconv(128, 64, kernel_size=3, padding=1)
        )

        self.conv2deconv_indices_1 = {
                0:6, 3:3
                }
        self.conv2deconv_indices_2 = {
                0:6, 3:3
                }
        self.conv2deconv_indices_3 = {
                0:12, 3:9, 6:6, 9:3
                }

        self.init_weight()
    
    def init_weight(self):
        adder_pretrained = torch.load("models_VGG_new/addernet_best.pt")

        for idx, layer in enumerate(adder_pretrained._modules.get('block1')):
            if isinstance(layer, adder.adder2d):
                self.deblock1[self.conv2deconv_indices_1[idx]].adder.data = layer.adder.data 

        for idx, layer in enumerate(adder_pretrained._modules.get('block2')):
            if isinstance(layer, adder.adder2d):
                self.deblock2[self.conv2deconv_indices_2[idx]].adder.data = layer.adder.data

        for idx, layer in enumerate(adder_pretrained._modules.get('block3')):
            if isinstance(layer, adder.adder2d):
                self.deblock3[self.conv2deconv_indices_3[idx]].adder.data = layer.adder.data
    
    def forward(self, x):
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
                self.pool_locs[idx] = location
            else:
                x = layer(x)
        
        # reshape to (1, 512 * 7 * 7)
        x = x.view(x.size()[0], -1)
        output = self.classifier(x)
        return output

        
    def forward(self, x, block, layer, activation_idx, pool_locs):
        try:
            if block == 1:
                if layer in self.conv2deconv_indices_1:
                    start_idx = self.conv2deconv_indices_1[layer]
            
                for idx in range(start_idx, len(self.deblock1)):
                    if idx == 0:
                        x = self.deblock1[idx](x, pool_locs[0])
                    else:
                        x = self.deblock1[idx](x)

            if block == 2:
                if layer in self.conv2deconv_indices_2:
                    start_idx = self.conv2deconv_indices_2[layer]

                for idx in range(start_idx, len(self.deblock2)):
                    if idx == 0:
                        x = self.deblock2[idx](x, pool_locs[1])
                    else:
                        x = self.deblock2[idx](x)

                for idx in range(len(self.deblock1)):
                    if idx == 0:
                        x = self.deblock1[idx](x, pool_locs[0])
                    else:
                        x = self.deblock1[idx](x)
            
            if block == 3:
                if layer in self.conv2deconv_indices_3:
                    start_idx = self.conv2deconv_indices_3[layer]

                for idx in range(start_idx, len(self.deblock3)):
                    if idx == 0:
                        x = self.deblock3[idx](x, pool_locs[2])
                    else:
                        x = self.deblock3[idx](x)
                
                for idx in range(len(self.deblock2)):
                    if idx == 0:
                        x = self.deblock2[idx](x, pool_locs[1])
                    else:
                        x = self.deblock2[idx](x)

                for idx in range(len(self.deblock1)):
                    if idx == 0:
                        x = self.deblock1[idx](x, pool_locs[0])
                    else:
                        x = self.deblock1[idx](x)
        except:
            # raise ValueError('block name is illegal or layer is not a conv feature map')
            raise Exception
        return x


class AdderDeconv_new(nn.Module):

    def __init__(self):
        super(AdderDeconv_new, self).__init__()

        self.feature = nn.Sequential(
            # deconv3
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            BNTranspose(128),
            adder.adder2d_deconv(128, 128, 3, padding=1),
            nn.ReLU(),
            BNTranspose(128),
            adder.adder2d_deconv(128, 128, 3, padding=1),
            nn.ReLU(),
            BNTranspose(128),
            adder.adder2d_deconv(128, 128, 3, padding=1),
            nn.ReLU(),
            BNTranspose(128),
            adder.adder2d_deconv(128, 64, 3, padding=1),
            
            # deconv4
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            BNTranspose(64),
            adder.adder2d_deconv(64, 64, 3, padding=1),
            nn.ReLU(),
            BNTranspose(64),
            adder.adder2d_deconv(64, 32, 3, padding=1),
            
            # deconv5
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            BNTranspose(32),
            adder.adder2d_deconv(32, 32, 3, padding=1),
            nn.ReLU(),
            BNTranspose(32),
            adder.adder2d_deconv(32, 3, 3, padding=1)    
        )

        self.conv2deconv_indices = {
                0:26, 3:23, 7:19, 10:16,
                14:12, 17:9, 20:6, 23:3
                }

        self.batchnorm_indices = {
                1:25, 4:22, 8:18, 11:15,
                15:11, 18:8, 21:5, 24:2
                }

        self.unpool2pool_indices = {
                20:6, 13:13, 0:26
                }

        self.init_weight()

    def init_weight(self):
        adder_pretrained = torch.load("models_VGG_new/addernet_best.pt")
        for idx, layer in enumerate(adder_pretrained.feature):
            if isinstance(layer, adder.adder2d):
                self.feature[self.conv2deconv_indices[idx]].adder.data = layer.adder.data 
            if isinstance(layer, nn.BatchNorm2d):
                t = (layer.weight / torch.sqrt(layer.running_var + layer.eps)).data
                self.feature[self.batchnorm_indices[idx]].weight.data = t
                self.feature[self.batchnorm_indices[idx]].bias.data = (-layer.bias * t + layer.running_mean).data
        
    def forward(self, x, layer, activation_idx, pool_locs):
        if layer in self.conv2deconv_indices:
            start_idx = self.conv2deconv_indices[layer]
        else:
            raise ValueError('layer is not a conv feature map')

        for idx in range(start_idx, len(self.feature)):
            if isinstance(self.feature[idx], nn.MaxUnpool2d):
                x = self.feature[idx]\
                (x, pool_locs[self.unpool2pool_indices[idx]])
            else:
                x = self.feature[idx](x)
        return x


if __name__ == '__main__':
    v = VGGSmallDeconv()