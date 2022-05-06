import cv2
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
 


class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output
    def close(self):
        self.hook.remove()


def normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


class FilterVisualizer():
    def __init__(self, size=56, upscaling_steps=12, upscaling_factor=1.2, img_path=None):
        if img_path is not None:
            self.from_img = True
            self.img_path = img_path
            self.size = 56
            self.upscaling_steps = 1
            self.upscaling_factor = 1
        else:
            self.from_img = False
            self.size, self.upscaling_steps, self.upscaling_factor = size, upscaling_steps, upscaling_factor
        
        # self.model = torchvision.models.resnet18(pretrained=True).cuda().eval()
        # self.model = torch.load("models_VGG_new/addernet_best.pt").cuda().eval()
        self.model = torchvision.models.vgg16(pretrained=True).cuda().eval()
        # self.model = torch.load("models_finetune/addernet_best.pt").cuda().eval()

        # for param in self.model.parameters(): 
        #     param.requires_grad = True

    def visualize(self, layer, filter, lr=0.1, opt_steps=20, blur=None):
        sz = self.size
        if self.from_img:
            img = cv2.imread(self.img_path)
            img = cv2.blur(img, (blur,blur))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img, (32, 32))
            img = np.array(img) / 255
            img0 = img.copy()
        else:
            img = np.uint8(np.random.uniform(150, 180, (sz, sz, 3))) / 255  # generate random image

        # for id, item in enumerate(list(self.model.modules())):
        #     print(id, '-----', item)
        # raise Exception

        # for name,parameters in self.model.named_parameters():
        #     print(name,':',parameters.size())
        
        activations = SaveFeatures(list(self.model.modules())[layer])

        # activations = SaveFeatures(list(self.model.children())[layer])  # register hook

        for _ in range(self.upscaling_steps):  # scale the image up upscaling_steps times

            tfms_img = normalize(torch.from_numpy(img.transpose(2,0,1)), 
                                 mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]).to(torch.float).cuda()

            img_var = Variable(tfms_img[None], requires_grad=True)  # convert image to Variable that requires grad
            # print(img_var.shape)

            optimizer = torch.optim.SGD([img_var], lr=lr, momentum=0.9)
            
            for n in range(opt_steps):  # optimize pixel values for opt_steps times
                optimizer.zero_grad()
                self.model(img_var)  # NOTE: use self.model.feature(img_var) if using AdderVGG16
                # print(activations.features.shape)  # features: bsz * ch * h * w (activation shape)
                # for filter in range(512):
                #     print(activations.features[0, filter].mean().item())
                loss = -activations.features[0, filter].mean()
                # print(loss.requires_grad)
                # loss = -activations.mean()
                # print(loss)
                # raise Exception
                # print(loss)
                loss.backward()
                # print(activations.features[0, filter].grad)
                # print(img_var.grad)
                optimizer.step()

            # print(img_var.data.shape)
            # print(1)
            img = denormalize(img_var.data.cpu()[0], 
                              mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225]).numpy().transpose(1,2,0)
            
            if self.from_img:
                self.output = ((img-img0)-(img-img0).min())/((img-img0).max()-(img-img0).min())
            else:
                self.output = img

            sz = int(self.upscaling_factor * sz)  # calculate new image size
            img = cv2.resize(img, (sz, sz), interpolation = cv2.INTER_CUBIC)  # scale image up
            if blur is not None: 
                img = cv2.blur(img,(blur,blur))  # blur image to reduce high frequency patterns
        
        self.save(layer, filter)
        activations.close()

    def save(self, layer, filter):
        plt.imsave("visualization/layer_"+str(layer)+"_filter_"+str(filter)+".jpg", np.clip(self.output, 0, 1))



layer = 25

# FV = FilterVisualizer(size=56, upscaling_steps=6, upscaling_factor=1.4)
FV = FilterVisualizer(img_path="vis_data/cat.jpg")
for filter in tqdm(range(512)):
    FV.visualize(layer, filter, lr=0.2, opt_steps=50, blur=50)