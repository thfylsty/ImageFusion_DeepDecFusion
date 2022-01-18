# -*- coding: utf-8 -*-
import torch
from model import Model
from utils import load_img
import os
import argparse


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



if __name__ == "__main__":
    print("===============Initialization===============")
    args = parse_args()
    print(args)
    os.chdir(r'./')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    print("epochs", args.epochs, "batch_size", args.batch_size)
    # Dataset
    model = Model(args).to(args.device)

    model.eval()
    path = "lena.jpg"
    img = load_img(path)
    with torch.no_grad():
        model.setdata(img)
        model.forward(isTest=True)
    model.saveimgdemo()




