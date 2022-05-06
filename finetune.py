#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import os
import torch
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
import argparse
import math
from tqdm import tqdm
import numpy as np
import time

parser = argparse.ArgumentParser(description='train-addernet')

# Basic model parameters.
parser.add_argument('--data', type=str, default='data/')
parser.add_argument('--model_dir', type=str, default='models/addernet_best.pt')
parser.add_argument('--output_dir', type=str, default='models_finetune/')
parser.add_argument('--log', default=False, action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

os.makedirs(args.output_dir, exist_ok=True)
if args.log:
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    fo = open('training_' + time_str + '.log', 'w')
    fo_csv = open('training_' + time_str + '.csv', 'w')

acc = 0
acc_best = 0

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

data_train = CIFAR10(args.data,
                   transform=transform_train,
                   download=True)
data_test = CIFAR10(args.data,
                  train=False,
                  transform=transform_test,
                  download=True)

data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=100, num_workers=0)

net = torch.load(args.model_dir).cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    lr = 0.05 * (1+math.cos(float(epoch)/800*math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def train(epoch):
    adjust_learning_rate(optimizer, epoch)
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    with tqdm(data_train_loader, desc="training") as pbar:
        for i, (images, labels) in enumerate(pbar, 1):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
    
            optimizer.zero_grad()
    
            output = net(images)
    
            loss = criterion(output, labels)

            loss_list.append(loss.data.item())
            batch_list.append(i)
    
            loss.backward()
            optimizer.step()

            pbar.set_description("Epoch: %d, Loss: %0.8f, lr: %0.6f" % (epoch, np.mean(loss_list), optimizer.param_groups[0]['lr']))
            if (i % 30 == 0 or i == len(data_train_loader)) and args.log:
                fo.write('[%d | %d] Loss: %f\n' % (epoch, i, loss.item()))

        print('Train - Epoch %d, Loss: %f' % (epoch, loss.data.item()))
        if args.log:
            fo.write('Train - Epoch %d, Loss: %f\n' % (epoch, loss.data.item()))

    return loss.data.item()

 
def test():
    global acc, acc_best
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            output = net(images)
            avg_loss += criterion(output, labels) * images.shape[0]
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
 
    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)
    if acc_best < acc:
        acc_best = acc
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))
    if args.log:
        fo.write('Test Avg. Loss: %f, Accuracy: %f\n' % (avg_loss.data.item(), acc))
    return avg_loss.data.item(), acc
 
 
def train_and_test(epoch):
    training_loss = train(epoch)
    testing_loss, testing_acc = test()
    if args.log:
        fo_csv.write('%f,%f,%f\n' % (training_loss, testing_loss, testing_acc))
    return testing_acc
 
 
def main():
    start = 600
    epoch = 800
    best_acc = 0
    for e in range(start, epoch + 1):
        testing_acc = train_and_test(e)
        if testing_acc > best_acc:
            best_acc = testing_acc
            torch.save(net, args.output_dir + 'addernet_best.pt')
        if e % 40 == 0 or e == epoch:
            torch.save(net, args.output_dir + 'addernet_{}.pt'.format(e))
        print('Best Accuracy: %f' % best_acc)
        if args.log:
            fo.write('Best Accuracy: %f\n' % best_acc)
 

if __name__ == '__main__':
    main()
