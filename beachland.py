import pynvml
import numpy as np
import os
import time
pynvml.nvmlInit()
# 这里的0是GPU id
handle0 = pynvml.nvmlDeviceGetHandleByIndex(0)
handle1 = pynvml.nvmlDeviceGetHandleByIndex(1)
handle2 = pynvml.nvmlDeviceGetHandleByIndex(2)
handle3 = pynvml.nvmlDeviceGetHandleByIndex(3)

memInfo0 = pynvml.nvmlDeviceGetMemoryInfo(handle0)
memInfo1 = pynvml.nvmlDeviceGetMemoryInfo(handle1)
memInfo2 = pynvml.nvmlDeviceGetMemoryInfo(handle2)
memInfo3 = pynvml.nvmlDeviceGetMemoryInfo(handle3)

commandList = ['',
               '',
               '',]
commandFalg = np.ones(len(commandList))

def getUsedRate(memInfo):
    return memInfo.used / memInfo.total


def sendCommand(deviceID):
    print(os.system('python train.py --epochs 1002 --devices 0'))
    print(str(deviceID) + ': command')
    exit()


setRate = 0.5


while (True):
    # print('显卡空闲')
    print('显卡被占用')
    time.sleep(2)
    while (getUsedRate(memInfo0) < setRate) :
        print('存在显卡空闲')
        if getUsedRate(memInfo0) < setRate:
            sendCommand(0)
        # elif getUsedRate(memInfo1) < setRate:
        #     sendCommand(1)
        # if getUsedRate(memInfo2) < setRate:
        #     sendCommand(2)
        # if getUsedRate(memInfo3) < setRate:
        #     sendCommand(3)

