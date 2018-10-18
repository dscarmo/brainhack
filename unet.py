import time

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    '''
    (conv => BN => ReLU) * 2, one UNET Block
    '''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    '''
    Input convolution
    '''
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    '''
    Downsample conv
    '''
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    '''
    Upsample conv
    '''
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        # TODO Upsample deprecated
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    '''
    Output convolution
    '''
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
    

class UNet(nn.Module):
    '''
    Main model class
    '''
    @staticmethod
    def test():
        size = (1, 1, 256, 256)
        inpt = torch.rand(size, requires_grad=True)
        
        plt.subplot(1, 2, 1)
        plt.imshow(inpt.squeeze().detach().numpy(), cmap='gray')
        
        begin = time.time()

        outpt = UNet.forward_test(inpt, 1, 1)
        print("Prediction Time: " + str(time.time() - begin))
        
        plt.subplot(1, 2, 2)
        plt.imshow(outpt.squeeze().detach().numpy(), cmap='gray')
        plt.show()

    @staticmethod
    def forward_test(inpt, n_channels, n_classes):
        unet = UNet(n_channels, n_classes)
        return unet.forward(inpt)
    
    def __init__(self, n_channels, n_classes, mask_ths=None):
        super(UNet, self).__init__()
        self.mask_ths = mask_ths
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        '''
        Saves every downstep output to use in upsteps concatenation
        '''
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        if self.mask_ths is not None:
            x = (x > self.mask_ths).long()
        return x

UNet(1, 1).test()