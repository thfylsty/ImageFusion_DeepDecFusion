# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from model import Model
from utils import Dataset
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data', default=r'../../data/ir_vi/', type=str, help='')
    parser.add_argument('--data', default=r'/data/Disk_B/MSCOCO2014/train2014', type=str, help='')
    parser.add_argument('--load_pt', default=True, type=bool, help='')
    parser.add_argument('--weights_path', default='./weights/epoch578_fusion.pt', type=str, help='')
    parser.add_argument('--lr', default= 1e-3, type=float, help='')
    parser.add_argument('--devices', default="0", type=str, help='')
    parser.add_argument('--device', default="cuda", type=str, help='')
    parser.add_argument('--batch_size', default=64, type=int, help='')
    parser.add_argument('--epochs', default=1000, type=int, help='')
    parser.add_argument('--multiGPU', default=False, type=bool, help='')
    parser.add_argument('--GPUs', default=[0, 1], type=list, help='')
    parser.add_argument('--backends', default=True, type=bool, help='')
    parser.add_argument('--contiguousparams', default=True, type=bool, help='')
    parser.add_argument('--workers', default=6, type=int, help='')

    return parser.parse_args()



if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    print("===============Initialization===============")
    args = parse_args()
    print(args)
    os.chdir(r'./')
    torch.backends.cudnn.benchmark = args.backends
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    print("epochs", args.epochs, "batch_size", args.batch_size)
    # Dataset
    data = Dataset(args.data, resize= [256,256],  gray = True)
    loader = DataLoader(data, batch_size = args.batch_size, shuffle=True,num_workers=args.workers)

    print("Load ---- {} images".format(len(data)))
    # Model
    model = Model(args).to(args.device)

    start_epoch = model.start_epoch

    # Training
    print('============ Training Begins [epochs:{}] ==============='.format(args.epochs))

    loss = torch.zeros(1)
    num = 0

    for epoch in range(start_epoch,args.epochs):
        model.scheduler.step(loss.item())
        tqdms = tqdm(loader)

        for img in tqdms:
            model.setdata(img)
            model.step()
            tqdms.set_description("epoch:%d %s"%(epoch,model.print))
            num+=1
            # model.saveimg(epoch,num)
        if epoch%1==0:
            model.saveimg(epoch)
        model.save(epoch)

