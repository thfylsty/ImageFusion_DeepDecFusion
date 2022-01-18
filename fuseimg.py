# -*- coding: utf-8 -*-
import torch
from model import Model
from utils import load_img,mkdir
import os
import argparse
import cv2
import time
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=r'/data/Disk_B/MSCOCO2014/train2014', type=str, help='')
    parser.add_argument('--load_pt', default=True, type=bool, help='')
    parser.add_argument('--weights_path', default='./weights/epoch323_fusion.pt', type=str, help='')
    parser.add_argument('--lr', default= 1e-3, type=float, help='')
    parser.add_argument('--devices', default="0", type=str, help='')
    parser.add_argument('--device', default="cuda", type=str, help='')
    parser.add_argument('--batch_size', default=32, type=int, help='')
    parser.add_argument('--epochs', default=1000, type=int, help='')
    parser.add_argument('--multiGPU', default=False, type=bool, help='')
    parser.add_argument('--GPUs', default=[0, 1], type=list, help='')
    return parser.parse_args()

def getimg(imgir_path):
    img = load_img(imgir_path)
    with torch.no_grad():
        model.setdata(img)
        s_time = time.time()
        model.forward(isTest=True)
        e_time = time.time() - s_time
        print(e_time)
    # model.saveimgfuse(imgir_path)

    return model.getimg()

def sm(x,y):
    ex = torch.exp(x)
    ey = torch.exp(y)
    s = ex+ey
    return x*ex/s +y*ey/s



if __name__ == "__main__":
    save_path = "result"
    test_ir = './Test_ir/'
    test_vi = './Test_vi/'
    # test_ir = './road/ir/'
    # test_vi = './road/vi/'
    img_list_ir = glob(test_ir + '*')
    img_num = len(img_list_ir)
    imgtype = '.bmp'

    args = parse_args()
    os.chdir(r'./')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    model = Model(args).to(args.device)
    model.eval()

    for i in range(1,img_num+1):
        imgir_path = test_ir+str(i)+imgtype
        imgvi_path = test_vi+str(i)+imgtype

        vi_g1, vi_g2, vi_g3, vi_s = getimg(imgir_path)
        ir_g1, ir_g2, ir_g3, ir_s = getimg(imgvi_path)

        fused_1 = torch.max(vi_g1, ir_g1) + torch.max(vi_g2, ir_g2) + torch.max(vi_g3, ir_g3) + (vi_s + ir_s) / 2
        fused_2 = torch.max(vi_g1, ir_g1) + torch.max(vi_g2, ir_g2) + torch.max(vi_g3, ir_g3) + sm(vi_s, ir_s)
        fused_3 = vi_g1 + ir_g1 + vi_g2 + ir_g2 + vi_g3 + ir_g3 + (vi_s + ir_s) / 2
        fused_4 = vi_g1 + ir_g1 + vi_g2 + ir_g2 + vi_g3 + ir_g3 + sm(vi_s, ir_s)
        fused_1 = fused_1.squeeze(0).squeeze(0).detach().cpu().numpy() * 255
        fused_2 = fused_2.squeeze(0).squeeze(0).detach().cpu().numpy() * 255
        fused_3 = fused_3.squeeze(0).squeeze(0).detach().cpu().numpy() * 255
        fused_4 = fused_4.squeeze(0).squeeze(0).detach().cpu().numpy() * 255

        save_path_1 = os.path.join(save_path, 'fuse1')
        mkdir(save_path_1)
        save_name_1 = os.path.join(save_path_1, '{}.bmp'.format(i))
        cv2.imwrite(save_name_1, fused_1)

        save_path_2 = os.path.join(save_path, 'fuse2')
        mkdir(save_path_2)
        save_name_2 = os.path.join(save_path_2, '{}.bmp'.format(i))
        cv2.imwrite(save_name_2, fused_2)

        save_path_3 = os.path.join(save_path, 'fuse3')
        mkdir(save_path_3)
        save_name_3 = os.path.join(save_path_3, '{}.bmp'.format(i))
        cv2.imwrite(save_name_3, fused_3)

        save_path_4 = os.path.join(save_path, 'fuse4')
        mkdir(save_path_4)
        save_name_4 = os.path.join(save_path_4, '{}.bmp'.format(i))
        cv2.imwrite(save_name_4, fused_4)

        print("pic:[%d] %s" % (i, save_name_1))





