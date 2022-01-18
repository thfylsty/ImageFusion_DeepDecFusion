# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
from glob import glob
import os
from PIL import Image
import numpy as np
from torchvision import models
_tensor = transforms.ToTensor()
_pil_rgb = transforms.ToPILImage('RGB')
_pil_gray = transforms.ToPILImage()
device = 'cuda'

class Dataset(Data.Dataset):
    def __init__(self, root, resize=256, gray=True):
        self.files = glob(os.path.join(root, '*.*'))
        self.resize = resize
        self.gray = gray
        self._tensor = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
        ])
        # print(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index])

        if self.gray:
            img = img.convert('L')
        img = self.transform(img)
        return img

def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

def load_img(img_path, img_type='gray'):
    img = Image.open(img_path)
    if img_type=='gray':
        img = img.convert('L')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = transform(img).unsqueeze(0)
    return img

def gradient(input):
    # print(input.sum(),input.mean())

    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(1, 1, 3, bias=False,padding=1)
    # 定义算子参数 [0.,1.,0.],[1.,-4.,1.],[0.,1.,0.] Laplacian 四邻域 八邻域
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    # kernel = np.array([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]], dtype='float32')
    # 将算子转换为适配卷积操作的卷积核
    kernel = kernel.reshape((1, 1, 3, 3))
    # 给卷积操作的卷积核赋值
    conv_op.weight.data = (torch.from_numpy(kernel)).to(device).type(torch.float32)
    # 对图像进行卷积操作
    edge_detect = conv_op(input)
    # print(edge_detect.sum(),edge_detect.mean())

    return edge_detect

def hist_similar(x,y):
    t_min = torch.min(torch.cat((x, y), 1)).item()
    t_max = torch.max(torch.cat((x, y), 1)).item()
    return (torch.norm((torch.histc(x, 255, min=t_min, max=t_max)-torch.histc(y, 255, min=t_min, max=t_max)),2))/255


def fusion_exp( a, b):
    expa = torch.exp(a)
    expb = torch.exp(b)
    pa = expa / (expa + expb)
    pb = expb / (expa + expb)

    return pa * a + pb * b



class VGGLoss(nn.Module):
    def __init__(self,end=5):
        super(VGGLoss, self).__init__()
        # device = 'cuda:3'
        self.vgg = Vgg19(end=end).to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0][:end]
        self.e = end

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.151, 0.131, 0.120]).reshape((1, 3, 1, 1))),
                                       requires_grad=False).to(device)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.037, 0.034, 0.031]).reshape((1, 3, 1, 1))),
                                      requires_grad=False).to(device)

    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class Vgg19(torch.nn.Module):
    def __init__(self, end = 5,requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.e = end

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out[:self.e]




def mulGANloss(input_, is_real):
    criterionGAN = torch.nn.MSELoss()

    if is_real:
        label = 1
    else:
        label = 0

    if isinstance(input_[0], list):
        loss = 0.0
        for i in input_:
            pred = i[-1]
            target = torch.Tensor(pred.size()).fill_(label).to(pred.device)
            loss += criterionGAN(pred, target)
        return loss
    else:
        target = torch.Tensor(input_[-1].size()).fill_(label).to(input_[-1].device)
        return criterionGAN(input_[-1], target)