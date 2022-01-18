# -*- coding: utf-8 -*-
import torch.nn as nn
from network import Decomposition,MultiscaleDiscriminator,downsample
from utils import gradient
from ssim import SSIM
import torch
import torch.optim as optim
import torchvision
import os
import torch.nn.functional as F
from contiguous_params import ContiguousParams


class Model(nn.Module):
    def __init__(self,args):
        super(Model, self).__init__()
        self.fusion = Decomposition()
        self.D = MultiscaleDiscriminator(input_nc=1)
        self.MSE_fun = nn.MSELoss()
        self.L1_loss = nn.L1Loss()
        self.SSIM_fun = SSIM()

        if args.contiguousparams==True:
            print("ContiguousParams---")
            parametersF = ContiguousParams(self.fusion.parameters())
            parametersD = ContiguousParams(self.D.parameters())
            self.optimizer_G = optim.Adam(parametersF.contiguous(), lr=args.lr)
            self.optimizer_D = optim.Adam(parametersD.contiguous(), lr=args.lr)
        else:
            self.optimizer_G = optim.Adam(self.fusion.parameters(), lr=args.lr)
            self.optimizer_D = optim.Adam(self.D.parameters(), lr=args.lr)

        self.g1 = self.g2 = self.g3 = self.s = self.img_re = None
        self.loss = torch.zeros(1)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_G , mode='min', factor=0.5, patience=2,
                                                               verbose=False, threshold=0.0001, threshold_mode='rel',
                                                               cooldown=0, min_lr=0, eps=1e-10)
        self.min_loss = 1000
        self.args = args
        self.downsample = downsample()
        self.criterionGAN = torch.nn.MSELoss()

        if args.multiGPU:
            self.mulgpus()
        self.load()
        self.load_D()

    def load_D(self,):
        if self.args.load_pt:
            print("=========LOAD WEIGHTS D=========")
            path = self.args.weights_path.replace("fusion","D")
            print(path)
            checkpoint = torch.load(path)

            if self.args.multiGPU:
                print("load D")
                self.D.load_state_dict(checkpoint['weight'])
            else:
                print("load D single")
                # 单卡模型读取多卡模型
                state_dict = checkpoint['weight']
                # create new OrderedDict that does not contain `module.`
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '')  # remove `module.`
                    new_state_dict[name] = v
                # load params
                self.D.load_state_dict(new_state_dict)


            print("=========END LOAD WEIGHTS D=========")

    def load(self,):
        start_epoch = 0
        if self.args.load_pt:
            print("=========LOAD WEIGHTS=========")
            checkpoint = torch.load(self.args.weights_path)
            start_epoch = checkpoint['epoch'] + 1
            try:
                if self.args.multiGPU:
                    print("load G")
                    self.fusion.load_state_dict(checkpoint['weight'])
                else:
                    print("load G single")
                    # 单卡模型读取多卡模型
                    state_dict = checkpoint['weight']
                    # create new OrderedDict that does not contain `module.`
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k.replace('module.', '')  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.fusion.load_state_dict(new_state_dict)
            except:
                model = self.fusion
                print("weights not same ,try to load part of them")
                model_dict = model.state_dict()
                pretrained = torch.load(self.args.weights_path)['weight']
                # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in model_dict.items() if k in pretrained}
                left_dict = {k for k, v in model_dict.items() if k  not in pretrained}
                print(left_dict)
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                model.load_state_dict(model_dict)
                print(len(model_dict),len(pretrained_dict))
                # model_dict = self.fusion.state_dict()
                # pretrained_dict = {k: v for k, v in model_dict.items() if k in checkpoint['weight'] }
                # print(len(checkpoint['weight'].items()), len(pretrained_dict), len(model_dict))
                # model_dict.update(pretrained_dict)
                # self.fusion.load_state_dict(model_dict)
            print("start_epoch:", start_epoch)
            print("=========END LOAD WEIGHTS=========")
        print("========START EPOCH: %d========="%start_epoch)
        self.start_epoch = start_epoch



    def mulGANloss(self, input_, is_real):
        if is_real:
            label = 1
        else:
            label = 0

        if isinstance(input_[0], list):
            loss = 0.0
            for i in input_:
                pred = i[-1]
                target = torch.Tensor(pred.size()).fill_(label).to(pred.device)
                loss += self.criterionGAN(pred, target)
            return loss
        else:
            target = torch.Tensor(input_[-1].size()).fill_(label).to(input_[-1].device)
            return self.criterionGAN(input_[-1], target)

    def forward(self,isTest=False):
        self.g1, self.g2, self.g3, self.s, self.img_re = self.fusion(self.img,isTest)

    def set_requires_grad(self, net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def backward_G(self):
        img = self.img
        img_re = self.img_re
        img_g = gradient(img)
        self.img_down = self.downsample(img)
        self.img_g = img_g
        # print(self.g1.sum(),self.g2.sum(),self.g3.sum(),img_g.sum())
        # print(self.g1.mean(), self.g2.mean(), self.g3.mean(), img_g.mean())
        g1 = self.MSE_fun(self.g1, img_g)
        g2 = self.MSE_fun(self.g2, img_g)
        g3 = self.MSE_fun(self.g3, img_g)
        grd_loss = g1+g2+g3
        self.lossg1 ,self.lossg2,self.lossg3 = g1,g2,g3
        # grd_loss = self.MSE_fun(self.g1, img_g) + self.MSE_fun(self.g2, img_g) + self.MSE_fun(self.g3, img_g)
        ssim_loss = 1 - self.SSIM_fun(img_re, img)
        ssim_loss = ssim_loss * 10
        pixel_loss = self.MSE_fun(img_re, img)
        pixel_loss = pixel_loss * 100

        loss_G = self.mulGANloss(self.D(self.s), is_real=True)*0.1

        # 损失求和 回传
        loss = pixel_loss + ssim_loss + grd_loss + loss_G


        loss.backward()
        self.loss,self.pixel_loss,self.ssim_loss, self.grd_loss = loss,pixel_loss,ssim_loss, grd_loss
        self.loss_G = loss_G

    def backward_D(self):
        # RealReal
        # Real
        pred_real = self.D(self.img_down.detach())
        loss_D_real = self.mulGANloss(pred_real, is_real=True)
        # Fake
        pred_fake = self.D(self.s.detach())
        loss_D_fake = self.mulGANloss(pred_fake, is_real=False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        self.loss_D = loss_D
        self.loss_D_real,self.loss_D_fake = loss_D_real,loss_D_fake

    def mulgpus(self):
        self.fusion= nn.DataParallel(self.fusion.cuda(), device_ids=self.args.GPUs, output_device=self.args.GPUs[0])
        self.D = nn.DataParallel(self.D.cuda(), device_ids=self.args.GPUs, output_device=self.args.GPUs[0])

    def setdata(self,img):
        img = img.to(self.args.device)
        self.img = img

    def step(self):
        self.optimizer_G.zero_grad()  # set G gradients to zero
        self.forward()

        self.set_requires_grad(self.D, False)  # D require no gradients when optimizing G

        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # update G weights

        # if it % 10 == 0:
        self.set_requires_grad(self.D, True)
        self.optimizer_D.zero_grad()  # set D gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D weights

        self.print = 'ALL[%.5lf] pixel[%.5lf] grd[%.5lf](%.5lf %.5lf %.5lf) ssim[%.5lf] G[%.5lf]  D[%.5lf][%.5lf %.5lf ]' %\
                     (self.loss.item(), self.pixel_loss.item(), self.grd_loss.item(),self.lossg1.item() ,self.lossg2.item(),self.lossg3.item(), self.ssim_loss.item(),
                      self.loss_G.item(),self.loss_D.item(),self.loss_D_real.item(),self.loss_D_fake.item(),)


    def saveimg(self,epoch,num=0):
        img = torchvision.utils.make_grid(
            [self.img[0].cpu(), self.img_re[0].cpu(), self.img_down[0].cpu(),self.img_g[0].cpu(), self.s[0].cpu(), self.g1[0].cpu(), self.g2[0].cpu(),
             self.g3[0].cpu(), (self.g1+self.g2+self.g3)[0].cpu()], nrow=5)
        torchvision.utils.save_image(img, fp=(os.path.join('output/result_' + str(epoch) + '.jpg')))
        # torchvision.utils.save_image(img, fp=(os.path.join('output/epoch/'+str(num)+'.jpg')))

    def saveimgdemo(self):
        self.img_down = self.downsample(self.img)
        self.img_g = gradient(self.img)

        img = torchvision.utils.make_grid(
            [self.img[0].cpu(), self.img_re[0].cpu(), self.img_down[0].cpu(),self.img_g[0].cpu(), self.s[0].cpu(), self.g1[0].cpu(), self.g2[0].cpu(),
             self.g3[0].cpu(), (self.g1+self.g2+self.g3)[0].cpu()], nrow=5)
        torchvision.utils.save_image(img, fp=(os.path.join('demo_result.jpg')))
        # torchvision.utils.save_image(img, fp=(os.path.join('output/epoch/'+str(num)+'.jpg')))

    def saveimgfuse(self,name=''):
        self.img_down = self.downsample(self.img)
        self.img_g = gradient(self.img)

        img = torchvision.utils.make_grid(
            [self.img[0].cpu(), self.img_g[0].cpu(), ((self.g1+self.g2+self.g3)*1.5)[0].cpu()], nrow=3)
        torchvision.utils.save_image(img, fp=(os.path.join(name.replace('Test','demo'))))
        # torchvision.utils.save_image(img, fp=(os.path.join('output/epoch/'+str(num)+'.jpg')))

    def save(self, epoch):
        ## 保存模型和最佳模型
        if self.min_loss > self.loss.item():
            self.min_loss = self.loss.item()
            torch.save({'weight': self.fusion.state_dict(), 'epoch': epoch, },os.path.join('weights/best_fusion.pt'))
            torch.save({'weight': self.D.state_dict(), 'epoch': epoch, }, os.path.join('weights/best_D.pt'))
            print('[%d] - Best model is saved -' % (epoch))

        if epoch % 1 == 0:
            torch.save({'weight': self.fusion.state_dict(), 'epoch': epoch, },os.path.join('weights/epoch' + str(epoch) + '_fusion.pt'))
            torch.save({'weight': self.D.state_dict(), 'epoch': epoch, },os.path.join('weights/epoch' + str(epoch) + '_D.pt'))

    def getimg(self):
        return self.g1, self.g2,self.g3,self.s
