"""
zerodce.py
Standalone PyTorch implementation of Zero-DCE++ for Low-Light Enhancement.
Uses Depthwise Separable Convolutions (CSDN) for extreme efficiency.
"""

import torch
import torch.nn as nn

class CSDN_Tem(nn.Module):
    """Depthwise Separable Convolution Block"""
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input_tensor):
        out = self.depth_conv(input_tensor)
        out = self.point_conv(out)
        return out

class ZeroDCE_PP(nn.Module):
    """Zero-DCE++ Architecture"""
    def __init__(self):
        super(ZeroDCE_PP, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32
        
        self.e_conv1 = CSDN_Tem(3, number_f)
        self.e_conv2 = CSDN_Tem(number_f, number_f)
        self.e_conv3 = CSDN_Tem(number_f, number_f)
        self.e_conv4 = CSDN_Tem(number_f, number_f)
        self.e_conv5 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv6 = CSDN_Tem(number_f * 2, number_f)
        
        # FIXED: Zero-DCE++ outputs 3 channels (one curve per RGB channel).
        self.e_conv7 = CSDN_Tem(number_f * 2, 3) 

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        # Skip connections
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        # Curve parameter estimation (Shape: Batch, 3, H, W)
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        # Iteratively apply the SAME enhancement curves 8 times
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image = x + x_r * (torch.pow(x, 2) - x)
        
        return enhance_image