"""
this code is modified from https://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks

original author: Utku Ozbulak - github.com/utkuozbulak
"""
import sys
sys.path.append("..")

import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor

normalize_mean = torch.tensor([0.4914, 0.4822, 0.4465])
normalize_std = torch.tensor([0.2023, 0.1994, 0.2010])

def normalize(x):
    _normalize_mean = normalize_mean.reshape((1, -1) + (1,) * (len(x.shape) - 2)).expand(x.shape).to(x.device)
    _normalize_std = normalize_std.reshape((1, -1) + (1,) * (len(x.shape) - 2)).expand(x.shape).to(x.device)
    return (x - _normalize_mean) / _normalize_std

def denormalize(x):
    _normalize_mean = normalize_mean.reshape((1, -1) + (1,) * (len(x.shape) - 2)).expand(x.shape).to(x.device)
    _normalize_std = normalize_std.reshape((1, -1) + (1,) * (len(x.shape) - 2)).expand(x.shape).to(x.device)
    return x * _normalize_std + _normalize_mean


def project(x, original_x, epsilon, _type='linf'):

    if _type == 'linf':
        _x = denormalize(x)
        _original_x = denormalize(original_x)
        max_x = _original_x + epsilon
        min_x = _original_x - epsilon

        _x = torch.max(torch.min(_x, max_x), min_x)
        x = normalize(_x)

    elif _type == 'l2':
        dist = (x - original_x)

        dist = denormalize(dist)

        dist = dist.view(x.shape[0], -1)

        dist_norm = torch.norm(dist, dim=1, keepdim=True)

        mask = (dist_norm > epsilon).unsqueeze(2).unsqueeze(3)

        # dist = F.normalize(dist, p=2, dim=1)

        dist = dist / dist_norm

        dist *= epsilon

        dist = dist.view(x.shape)

        dist = normalize(dist)

        x = (original_x + dist) * mask.float() + x * (1 - mask.float())

    else:
        raise NotImplementedError

    return x

class PGD:
    b"""
        Fast gradient sign untargeted adversarial attack, minimizes the initial class activation
        with iterative grad sign updates
    """
    def __init__(self, model, epsilon, alpha, min_val, max_val, max_iters, _type='linf'):
        self.model = model
        self.model.eval()

        # Maximum perturbation
        self.epsilon = epsilon
        # Movement multiplier per iteration
        self.alpha = alpha
        # Minimum value of the pixels
        self.min_val = min_val
        # Maximum value of the pixels
        self.max_val = max_val
        # Maximum numbers of iteration to generated adversaries
        self.max_iters = max_iters
        # The perturbation of epsilon
        self._type = _type
        
    def perturb(self, original_images, labels, reduction4loss='mean', random_start=False):
        # original_images: values are within self.min_val and self.max_val

        # The adversaries created from random close points to the original data
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = tensor2cuda(rand_perturb)
            x = original_images + rand_perturb
            x.clamp_(self.min_val, self.max_val)
        else:
            x = original_images.clone()

        x.requires_grad = True 

        # max_x = original_images + self.epsilon
        # min_x = original_images - self.epsilon

        with torch.enable_grad():
            for _iter in range(self.max_iters):
                outputs = self.model(x)

                loss = F.cross_entropy(outputs, labels, reduction=reduction4loss)

                if reduction4loss == 'none':
                    grad_outputs = tensor2cuda(torch.ones(loss.shape))
                    
                else:
                    grad_outputs = None

                grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs, 
                        only_inputs=True)[0]

                x.data += self.alpha * torch.sign(grads.data)

                # the adversaries' pixel value should within max_x and min_x due 
                # to the l_infinity / l2 restriction
                x = project(x, original_images, self.epsilon, self._type)
                # the adversaries' value should be valid pixel value
                x = denormalize(x)
                x = x.clamp(self.min_val, self.max_val)
                x = normalize(x)

        return x


class FGSM:
    b"""
        Fast gradient sign untargeted adversarial attack, minimizes the initial class activation
        with iterative grad sign updates
    """
    def __init__(self, model, epsilon, min_val, max_val, _type='linf'):
        self.model = model
        self.model.eval()

        # Maximum perturbation
        self.epsilon = epsilon
        # Movement multiplier per iteration
        self.min_val = min_val
        # Maximum value of the pixels
        self.max_val = max_val
        # The perturbation of epsilon
        self._type = _type
        
    def perturb(self, original_images, labels, reduction4loss='mean', random_start=False):
        # original_images: values are within self.min_val and self.max_val

        # The adversaries created from random close points to the original data
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = tensor2cuda(rand_perturb)
            x = original_images + rand_perturb
            x.clamp_(self.min_val, self.max_val)
        else:
            x = original_images.clone()

        x.requires_grad = True 

        # max_x = original_images + self.epsilon
        # min_x = original_images - self.epsilon

        with torch.enable_grad():
            outputs = self.model(x)

            loss = F.cross_entropy(outputs, labels, reduction=reduction4loss)

            if reduction4loss == 'none':
                grad_outputs = tensor2cuda(torch.ones(loss.shape))
                
            else:
                grad_outputs = None

            grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs, 
                    only_inputs=True)[0]

            x.data += normalize(self.epsilon * torch.sign(grads.data))
            
            x = denormalize(x)
            x = x.clamp(self.min_val, self.max_val)
            x = normalize(x)

        return x