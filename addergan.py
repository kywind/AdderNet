import os
import adder
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader 

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.main = nn.Sequential(
			adder.adder2dTranspose(100, 128, kernel_size=4, stride=2, padding=1),
			nn.InstanceNorm2d(128),
			nn.ReLU(),
			adder.adder2dTranspose(128, 64, kernel_size=4, stride=2, padding=1),
			nn.InstanceNorm2d(64),
			nn.ReLU(),
			adder.adder2dTranspose(64, 32, kernel_size=3, stride=2, padding=1),
			nn.InstanceNorm2d(32),
			nn.ReLU(),
			adder.adder2dTranspose(32, 16, kernel_size=4, stride=2, padding=1),
			nn.InstanceNorm2d(16),
			nn.ReLU(),
			adder.adder2dTranspose(16, 1, kernel_size=4, stride=2, padding=1),
			nn.InstanceNorm2d(1),
			nn.Tanh()
		)
	
	def forward(self, x):
		return self.main(x)

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.main = nn.Sequential(
			adder.adder2d(1, 16, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(16),
			nn.LeakyReLU(0.2, True),
			adder.adder2d(16, 32, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.2, True),
			adder.adder2d(32, 64, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2, True),
			adder.adder2d(64, 128, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, True),
			adder.adder2d(128, 1, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(1),
			nn.Sigmoid()
		)
	
	def forward(self, x):
		return self.main(x)

def train(G, D, dataloader, number_epoch, lr, beta):
	criterion = nn.BCELoss()
	optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta, 0.999))
	optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta, 0.999))
	fixed_noise = torch.randn(64, 100, 1, 1)
	fixed_noise = fixed_noise.cuda()
	iters = 0
	print('Training started')
	for epoch in range(number_epoch):
		for i, data in enumerate(dataloader, 0):
			for _ in range(5):
				D.zero_grad()
				inputs = data[0]
				batch_size = inputs.size(0)
				labels = torch.full((batch_size,), 1.)
				inputs = inputs.cuda()
				labels = labels.cuda()
				outputs = D(inputs).view(-1)
				Dx = outputs.mean().item()
				loss_real = criterion(outputs, labels)
				loss_real.backward()

				noises = torch.randn(batch_size, 100, 1, 1)
				noises = noises.cuda()
				inputs = G(noises)
				labels = torch.full((batch_size,), 0.)
				inputs = inputs.cuda()
				labels = labels.cuda()
				outputs = D(inputs.detach()).view(-1)
				DGz1 = outputs.mean().item()
				loss_fake = criterion(outputs, labels)
				loss_fake.backward()
				optimizerD.step()

			G.zero_grad()
			labels = torch.full((batch_size,), 1.)
			inputs = inputs.cuda()
			labels = labels.cuda()
			outputs = D(inputs).view(-1)
			DGz2 = outputs.mean().item()
			loss = criterion(outputs, labels)
			loss.backward()
			optimizerG.step()

			if i % 10 == 0:
				print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
					  % (epoch, number_epoch, i, len(dataloader),
						 (loss_real + loss_fake).item(), loss.item(), Dx, DGz1, DGz2))

		inputs = G(fixed_noise)
		vutils.save_image(inputs.detach(),
				'gan_img/img%03d.png' % (epoch),
				normalize=True)
	
	

if __name__ == "__main__":
	os.environ['CUDA_VISIBLE_DEVICES'] = '3'
	number_epoch = 1000
	lr = 0.02
	beta = 0.5
	dataroot = "data"
	batch_size = 64

	G = Generator().cuda()
	D = Discriminator().cuda()

	transform_test = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize((.5,), (.5,)),
	])
	data_test = MNIST('MNIST/',
                  train=False,
                  transform=transform_test,
                  download=True)
	dataloader = DataLoader(data_test, batch_size=batch_size,
		shuffle=True, num_workers=4)

	train(G, D, dataloader, number_epoch, lr, beta)