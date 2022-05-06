import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import transforms
import numpy as np
import cv2
from functools import partial
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from vis_models import *
from vis_models import *
from vis_utils import decode_predictions
from vgg import *

def load_images(img_path):
    # imread from img_path
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    # pytorch must normalize the pic by 
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    
    img = transform(img)
    img.unsqueeze_(0)
    #img_s = img.numpy()
    #img_s = np.transpose(img_s, (1, 2, 0))
    #cv2.imshow("test img", img_s)
    #cv2.waitKey()
    return img

def store(model):
    """
    make hook for feature map
    """
    def hook(module, input, output, key):
        if isinstance(module, nn.MaxPool2d):
           model.feature_maps[key] = output[0]
           model.pool_locs[key] = output[1]
        else:
           model.feature_maps[key] = output
    
    for idx, layer in enumerate(model._modules.get('feature')):   
        # print(layer) 
        # _modules returns an OrderedDict
        layer.register_forward_hook(partial(hook, key=idx))

def vis_layer(layer, adder_conv, adder_deconv):
    """
    visualing the layer deconv result
    """
    num_feat = adder_conv.feature_maps[layer].shape[1]
    
    # set other feature map activations to zero
    new_feat_map = adder_conv.feature_maps[layer].clone()

    # choose the max activations map
    act_lst = []
    for i in range(0, num_feat):
        choose_map = new_feat_map[0, i, :, :]
        activation = torch.max(choose_map)
        act_lst.append(activation.item())

    act_lst = np.array(act_lst)
    mark = np.argmax(act_lst)

    choose_map = new_feat_map[0, mark, :, :]
    max_activation = torch.max(choose_map)
    
    # make zeros for other feature maps
    if mark == 0:
        new_feat_map[:, 1:, :, :] = 0
    else:
        new_feat_map[:, :mark, :, :] = 0
        if mark != adder_conv.feature_maps[layer].shape[1] - 1:
            new_feat_map[:, mark + 1:, :, :] = 0
    
    choose_map = torch.where(choose_map==max_activation,
            choose_map,
            torch.zeros(choose_map.shape).cuda()
            )

    # make zeros for ther activations
    new_feat_map[0, mark, :, :] = choose_map
    
    # print(torch.max(new_feat_map[0, mark, :, :]))    
    print(max_activation)
    
    deconv_output = adder_deconv(new_feat_map, layer, mark, adder_conv.pool_locs)

    new_img = deconv_output.data.cpu().numpy()[0].transpose(1, 2, 0)  # (H, W, C)
    new_img = new_img[2:-2, 2:-2, :]
    # normalize
    new_img = (new_img - new_img.min()) / (new_img.max() - new_img.min()) * 255
    new_img = new_img.astype(np.uint8)
    # cv2.imshow('reconstruction img ' + str(layer), new_img)
    # cv2.waitKey()
    return new_img, int(max_activation)
    

if __name__ == '__main__':
    
    img_path = './vis_data/cat.jpg'

    # forward processing
    img = load_images(img_path).cuda()
    adder_conv = AdderConv_new().cuda()
    adder_conv.eval()
    store(adder_conv)
    conv_output = adder_conv(img)
    pool_locs = adder_conv.pool_locs
    # print('Predicted:', decode_predictions(conv_output, top=3)[0])
    

    
    # backward processing
    adder_deconv = AdderDeconv_new().cuda()
    adder_deconv.eval()
    # plt.figure(num=None, figsize=(16, 12), dpi=80)
    # plt.subplot(2, 3, 1)
    # plt.title('original picture')
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    # plt.imshow(img)    
    # for idx, layer in enumerate([10, 14, 17, 20, 23]):
    # for idx, layer in enumerate(adder_conv.conv_layer_indices):        
    # plt.subplot(2, 3, idx+2)
    layer = 23
    img, activation = vis_layer(layer, adder_conv, adder_deconv)
    # plt.title(f'{layer} layer, the max activations is {activation}')
    # img = img[112:,112:,:]
    # plt.imshow(img)
    # plt.colorbar()

    # plt.show()
    # plt.savefig('result.jpg')
    img = img.mean(2)
    cv2.imwrite('result.jpg', img)
    print('result picture has save at ./result.jpg')
