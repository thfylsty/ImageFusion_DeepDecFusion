# -*- coding: utf-8 -*-
from glob import glob
from network import Decomposition as FusionNet
import torch
from PIL import Image
import torchvision.transforms as transforms
import time
import os
from utils import mkdir
import cv2
import argparse

_tensor = transforms.ToTensor()


def read_img(img0_path,imgtype0):
    # print(img0_path+imgtype0)
    img0 = cv2.imread(img0_path+imgtype0)

    return img0

def load_img(img_path, device,img_type='gray'):
    img = Image.open(img_path)
    if img_type == 'gray':
        img = img.convert('L')
    return _tensor(img).unsqueeze(0).to(device)

def trans(img,device):
    return _tensor(img).unsqueeze(0).to(device)

def fuse(img0,img1,model):
    img0_g1, img0_g2, img0_g3, img0_s, img0_img_re = model(img0, isTest=True)
    img1_g1, img1_g2, img1_g3, img1_s, img1_img_re = model(img1, isTest=True)

    fused_1 = torch.max(img0_g1, img1_g1)+ torch.max(img0_g2,img1_g2)+ torch.max(img0_g3,img1_g3)+(img0_s+img1_s)/2
    fused_2 = torch.max(img0_g1, img1_g1)+ torch.max(img0_g2,img1_g2)+ torch.max(img0_g3,img1_g3)+torch.max(img0_s,img1_s)
    fused_3 = img0_g1 + img1_g1 + img0_g2 + img1_g2 + img0_g3 + img1_g3 + (img0_s + img1_s) / 2
    fused_4 = img0_g1 + img1_g1 + img0_g2 + img1_g2 + img0_g3 + img1_g3 + torch.max(img0_s,img1_s)
    fused_1 = fused_1.squeeze(0).squeeze(0).detach().cpu().numpy()*255
    fused_2 = fused_2.squeeze(0).squeeze(0).detach().cpu().numpy()*255
    fused_3 = fused_3.squeeze(0).squeeze(0).detach().cpu().numpy() * 255
    fused_4 = fused_4.squeeze(0).squeeze(0).detach().cpu().numpy() * 255
    img0_img_re = img0_img_re.squeeze(0).squeeze(0).detach().cpu().numpy() * 255
    return [fused_1,fused_2,fused_3,fused_4,img0_img_re]

def save_image(save_name,i,fused_img,save_path):
    save_path = os.path.join(save_path,  args.data)
    save_patht = os.path.join(save_path, save_name)
    mkdir(save_patht)

    save_name_1 = os.path.join(save_patht, '{}.bmp'.format(i))
    print(save_name_1)
    cv2.imwrite(save_name_1, fused_img)



def test(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devicenum
    device = args.device
    model = FusionNet().to(device)
    save_path = args.savepath
    mkdir(save_path)
    
    test_0 = './testdata/'+args.data+'/1/'
    test_1 = './testdata/'+args.data+'/2/'
    img_list_0 = glob(test_0+'*')
    img_list_1 = glob(test_1 + '*')
    # print(img_list_0)
    imgtype0 = img_list_0[0][-4:]
    imgtype1 = img_list_1[1][-4:]

    img_num = len(img_list_0)
    print("Test images num", img_num)
    checkpoint = torch.load(args.weightpath)
    model.load_state_dict(checkpoint['weight'])
    for i in range(1,img_num+1):
        
        img0_path = test_0+str(i)
        img1_path = test_1+str(i)

        # img0 = Image.open(img0_path)
        # img0b, img0g, img0r = img0.split()
        # img0 = cv2.imread(img0_path)
        img0 = read_img(img0_path,imgtype0)
        img0r,img0g,img0b  = cv2.split(img0)
        img0r = trans(img0r,args.device)
        img0g = trans(img0g,args.device)
        img0b = trans(img0b,args.device)

        # img1 = Image.open(img1_path)
        # img1b,img1g,img1r,  = img1.split()
        # img1 = cv2.imread(img1_path)
        img1 = read_img(img1_path,imgtype1)
        img1r,img1g,img1b  = cv2.split(img1)
        img1r = trans(img1r,args.device)
        img1g = trans(img1g,args.device)
        img1b = trans(img1b,args.device)

        fused_rs = fuse(img0r,img1r,model)
        fused_gs = fuse(img0g,img1g,model)
        fused_bs = fuse(img0b,img1b,model)

        fused_1 = cv2.merge([fused_rs[0], fused_gs[0], fused_bs[0]])
        fused_2 = cv2.merge([fused_rs[1], fused_gs[1], fused_bs[1]])
        fused_3 = cv2.merge([fused_rs[2], fused_gs[2], fused_bs[2]])
        fused_4 = cv2.merge([fused_rs[3], fused_gs[3], fused_bs[3]])
        # re0 = cv2.merge([fused_rs[4], fused_gs[4], fused_bs[4]])
        # fused_1 = Image.merge('RGB', (fused_rs[0], fused_gs[0], fused_bs[0]))
        # fused_2 = Image.merge('RGB', (fused_rs[1], fused_gs[1], fused_bs[1]))
        # fused_3 = Image.merge('RGB', (fused_rs[2], fused_gs[2], fused_bs[2]))
        # fused_4 = Image.merge('RGB', (fused_rs[3], fused_gs[3], fused_bs[3]))
        # save_image('fuse0', i, img0,save_path)
        save_image('fuse1',i,fused_1,save_path)
        save_image('fuse2',i,fused_2,save_path)
        save_image('fuse3',i,fused_3,save_path)
        save_image('fuse4',i,fused_4,save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=r'all', type=str, help='Med , Lytro,Exposure')
    parser.add_argument('--savepath', default="result491", type=str, help='')
    parser.add_argument('--devicenum', default="2", type=str, help='')
    parser.add_argument('--device', default="cuda", type=str, help='')
    parser.add_argument('--weightpath', default="./weights/epoch491_fusion.pt", type=str, help='')

    # parser.add_argument('--data', default=r'/data/Disk_B/MSCOCO2014/train2014', type=str, help='')
    # parser.add_argument('--load_pt', default=True, type=bool, help='')
    # parser.add_argument('--weights_path', default='./weights/epoch323_fusion.pt', type=str, help='')
    # parser.add_argument('--lr', default= 1e-3, type=float, help='')
    # parser.add_argument('--batch_size', default=32, type=int, help='')
    # parser.add_argument('--epochs', default=1000, type=int, help='')
    # parser.add_argument('--multiGPU', default=False, type=bool, help='')
    # parser.add_argument('--GPUs', default=[0, 1], type=list, help='')

    return parser.parse_args()


with torch.no_grad():
    args = parse_args()
    print(args)
    if args.data == 'all':
        for args.data in ["Med","Lytro","Tno","Road","Exposure"]:     
            test(args)
    else:
        test(args)



