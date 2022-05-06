import os
import torch
import argparse
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from attack import PGD, FGSM
from tqdm import tqdm

def make_attack(dataloader, attacker):
	correct = 0
	total = 0
	for data, label in tqdm(dataloader):
		data = data.cuda()
		label = label.cuda()
		attack_data = attacker.perturb(data, label)
		output = attacker.model(attack_data)
		pred = output.argmax(dim=-1)
		correct += (pred == label).sum().item()
		total += label.shape[0]
	print('Accuracy: %.2f%%' % (correct / total * 100))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', type=str, default='data/')
	parser.add_argument('--load_dir', type=str, default='models_finetune/addernet_best.pt')
	parser.add_argument('--attacker', type=str, default='FGSM')
	parser.add_argument('--eps', type=int, default=4)
	parser.add_argument('--alpha', type=int, default=1)
	parser.add_argument('--n_iter', type=int, default=10)
	parser.add_argument('--gpu', type=int, default=0)
	args = parser.parse_args()

	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

	transform_test = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	data_test = CIFAR10(args.data,
                  train=False,
                  transform=transform_test,
                  download=True)
	data_test_loader = DataLoader(data_test, batch_size=128, num_workers=8)
	model = torch.load(args.load_dir)
	model.eval()

	if args.attacker == 'FGSM':
		attacker = FGSM(model, epsilon=args.eps/255, min_val=0, max_val=1)
	else:
		attacker = PGD(model, epsilon=args.eps/255, alpha=args.alpha/255, min_val=0, max_val=1, max_iters=args.n_iter)
	
	make_attack(data_test_loader, attacker)