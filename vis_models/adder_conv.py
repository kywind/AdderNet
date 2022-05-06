import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
import sys

sys.path.append('../')
import adder

from collections import OrderedDict
from functools import partial


# class FeatureModel():
#     def __init__(self, model):
#         self.model = model
#         self.feature_maps = OrderedDict()
#         self.pool_locs = OrderedDict()
# 
#         def hook(module, input, output, key):
#             if isinstance(module, nn.MaxPool2d):
#                self.feature_maps[key] = output[0]
#                self.pool_locs[key] = output[1]
#             else:
#                self.feature_maps[key] = output
#         
#         for idx, layer in enumerate(self.model.modules()):  
#             # print(layer)
#             # _modules returns an OrderedDict
#             layer.register_forward_hook(partial(hook, key=idx))
#         # print(self.feature_maps)


class AdderConv(nn.Module):  # NOTE: this is DEPRECATED

    def __init__(self):
        super(AdderConv, self).__init__()

        self.block1 = nn.Sequential(
            adder.adder2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            adder.adder2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, return_indices=True)
        )

        self.block2 = nn.Sequential(
            adder.adder2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            adder.adder2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, return_indices=True)
        )

        self.block3 = nn.Sequential(
            adder.adder2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            adder.adder2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            adder.adder2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            adder.adder2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, return_indices=True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5, True),
            nn.Linear(2048, 10)
        )
        # index of conv
        self.conv_layer_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        # feature maps
        # self.feature_maps = OrderedDict()
        # switch
        self.pool_locs = OrderedDict()
        # initial weight
        self.init_weights()

    def init_weights(self):
        adder_pretrained = torch.load("models_VGG/addernet_best.pt")
        # adder_pretrained = FeatureModel(adder_pretrained)
        # adder_pretrained = SaveAllFeatures(list(model.modules()))
        # print(adder_pretrained.feature_maps.items())
        # raise Exception

        # fine-tune Conv2d
        for idx, layer in enumerate(adder_pretrained._modules.get('block1')):
            if isinstance(layer, adder.adder2d):
                self.block1[idx].adder.data = layer.adder.data 

        for idx, layer in enumerate(adder_pretrained._modules.get('block2')):
            if isinstance(layer, adder.adder2d):
                self.block2[idx].adder.data = layer.adder.data

        for idx, layer in enumerate(adder_pretrained._modules.get('block3')):
            if isinstance(layer, adder.adder2d):
                self.block3[idx].adder.data = layer.adder.data

        # fine-tune Linear
        for idx, layer in enumerate(adder_pretrained.classifier):
            if isinstance(layer, nn.Linear):
                self.classifier[idx].weight.data = layer.weight.data
                self.classifier[idx].bias.data = layer.bias.data
    
    def check(self):
        model = torch.load("../models_VGG/addernet_best.pt")
        return model
    
    def forward(self, x):
        for idx, layer in enumerate(self.block1):
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
                self.pool_locs[0] = location
            else:
                x = layer(x)

        for idx, layer in enumerate(self.block2):
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
                self.pool_locs[1] = location
            else:
                x = layer(x)

        for idx, layer in enumerate(self.block3):
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
                self.pool_locs[2] = location
            else:
                x = layer(x)
        
        # reshape to (1, 512 * 7 * 7)
        x = x.view(x.size()[0], -1)
        output = self.classifier(x)
        return output




class AdderConv_new(nn.Module):

    def __init__(self):
        super(AdderConv_new, self).__init__()

        self.feature = nn.Sequential(
            adder.adder2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            adder.adder2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, return_indices=True),

            adder.adder2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            adder.adder2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, return_indices=True),

            adder.adder2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            adder.adder2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            adder.adder2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            adder.adder2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, return_indices=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5, True),
            nn.Linear(2048, 10)
        )

        self.conv_layer_indices = [0, 3, 7, 10, 14, 17, 20, 23]
        self.feature_maps = OrderedDict()
        self.pool_locs = OrderedDict()
        self.init_weight()

    def init_weight(self):
        adder_pretrained = torch.load("models_VGG_new/addernet_best.pt")
        # print(list(adder_pretrained.children()))
        for idx, layer in enumerate(adder_pretrained.feature):
            if isinstance(layer, adder.adder2d):
                self.feature[idx].adder.data = layer.adder.data 
            if isinstance(layer, nn.BatchNorm2d):
                self.feature[idx].weight.data = layer.weight.data
                self.feature[idx].bias.data = layer.bias.data
                self.feature[idx].running_mean.data = layer.running_mean.data
                self.feature[idx].running_var.data = layer.running_var.data
                self.feature[idx].eps = layer.eps
        
        for idx, layer in enumerate(adder_pretrained.classifier):
            if isinstance(layer, nn.Linear):
                self.classifier[idx].weight.data = layer.weight.data
                self.classifier[idx].bias.data = layer.bias.data
        
    def forward(self, x):
        for idx, layer in enumerate(self.feature):
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
                self.pool_locs[0] = location
            else:
                x = layer(x)
        # x = x.view(x.size()[0], -1)
        # output = self.classifier(x)
        return x




if __name__ == '__main__':
    v = VGGSmall()

