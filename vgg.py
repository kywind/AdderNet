from torch.nn.modules.pooling import MaxUnpool1d
import adder
import torch.nn as nn

class VGGSmall(nn.Module):

    def __init__(self):
        super(VGGSmall, self).__init__()

        self.block1 = nn.Sequential(
            adder.adder2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            adder.adder2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        self.block2 = nn.Sequential(
            adder.adder2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            adder.adder2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
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
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5, True),
            nn.Linear(2048, 10)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


class VGGSmall_new(nn.Module):

    def __init__(self):
        super(VGGSmall_new, self).__init__()

        self.feature = nn.Sequential(
            # conv1
            adder.adder2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            adder.adder2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            
            # conv2
            adder.adder2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            adder.adder2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),

            # conv3
            adder.adder2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            adder.adder2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            adder.adder2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            adder.adder2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5, True),
            nn.Linear(2048, 10),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size()[0], -1)
        output = self.classifier(x)
        return output

