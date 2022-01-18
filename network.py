import torch.nn as nn
import numpy as np
from utils import mulGANloss

class Decomposition(nn.Module):
    def __init__(self):
        super(Decomposition, self).__init__()
        self.Conv_in = nn.Sequential(*[nn.Conv2d(1, 16, 3, 1, 1), nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(16, 32, 3, 1, 1), nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(32, 64, 3, 1, 1), nn.LeakyReLU(inplace=True)])

        self.Convd1 = nn.Sequential(*[nn.Conv2d(64, 64, 3, 1, 1),nn.LeakyReLU(inplace=True)])
        self.Convd2 = nn.Sequential(*[nn.Conv2d(64, 64, 3, 1, 1)],nn.LeakyReLU(inplace=True))
        self.Convd3 = nn.Sequential(*[ nn.Conv2d(64, 64, 3, 1, 1)],nn.LeakyReLU(inplace=True))

        self.Convd11 = nn.Conv2d(64, 64, 1, 1, 0,bias=False)
        self.Convd12 = nn.Conv2d(64, 64, 1, 1, 0,bias=False)
        self.Convd13 = nn.Conv2d(64, 64, 1, 1, 0,bias=False)

        self.Conv_g = nn.Sequential(*[nn.Conv2d(64, 32, 3, 1, 1), nn.LeakyReLU(inplace=True),
                                      nn.Conv2d(32, 16, 3, 1, 1), nn.LeakyReLU(inplace=True),
                                      nn.Conv2d(16, 1, 3, 1, 1), nn.Tanh()])

        self.Conv_res = nn.Sequential(*[nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True), ])

        self.Conv_s = nn.Sequential(*[nn.Conv2d(64, 32, 3, 2, 1), nn.ReLU(inplace=True),
                                      nn.Conv2d(32, 16, 3, 2, 1), nn.ReLU(inplace=True),
                                      nn.Conv2d(16, 1, 3, 1, 1), nn.Tanh()])

        self.Upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)


    def forward(self, x,isTest = False):
        x = self.Conv_in(x)
        x1 = x

        x = self.Convd1(x)
        g1 = self.Conv_g(self.Convd11(x))
        # print(g1.sum())
        x = self.Convd2(x)
        g2 = self.Conv_g(self.Convd12(x))
        x = self.Convd3(x)
        g3 = self.Conv_g(self.Convd13(x))

        x1 = self.Conv_res(x1)
        x = x+x1
        x = self.Conv_s(x)

        # 这里测试图像尺寸不固定 训练的时候可以升倍数 测试的时候只能上采样固定尺寸
        if isTest:  # Upsample for Test, The size of some pictures is not a power of 2
            test_upsample = nn.Upsample(size=(g1.shape[2], g1.shape[3]), mode='bilinear', align_corners=True)
            x = test_upsample(x)
        else:
            x = self.Upsample(x)

        out = x + g1 + g2 + g3
        return  g1,g2,g3,x,out


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc=4, ndf=32, n_layers=1, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=True, num_D=3, getIntermFeat=True):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        # print(len(result),"re")
        return result


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class downsample(nn.Module):
    def __init__(self):
        super(downsample, self).__init__()
        self.down0 = nn.Sequential(
            nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False),
            nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        return self.down0(x)