###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
TEST network that fit into AI84

Optionally quantize/clamp activations
"""
import torch.nn as nn

import ai8x


class AI85tcn(nn.Module):
    """
    1D TCN
    """
    def __init__(self,num_classes=1, num_channels=1,dimensions=(128,), bias=True, **kwargs):
        """
        num_channels: num INPUT channels
        """
        super().__init__()

        #num_classes HAVE TO LEAVE IT HERE

        

        # Limits
        #assert planes + num_channels <= ai8x.dev.WEIGHT_INPUTS
        
        bias=True
        dilation=1
        num_hidden_channels = 64

        self.conv1 = ai8x.FusedConv1dReLU(
            in_channels=num_channels,
            out_channels=num_hidden_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=dilation,
            bias=bias,
            **kwargs
        )

        self.conv2 = ai8x.FusedConv1dReLU(
            in_channels=num_hidden_channels,
            out_channels=num_hidden_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=dilation,
            bias=bias,
            **kwargs
        )

        self.linear_mix = ai8x.Conv1d(
            in_channels=num_hidden_channels,# * dilation_depth * num_repeat,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=bias,
            wide=True, #32 bit output!
            **kwargs
        )

        #self.fc = ai8x.Linear(fc_inputs*dim*dim, num_classes, bias=True, wide=True, **kwargs)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)

        out = self.linear_mix(x)

        return out


def ai85tcn(pretrained=False, **kwargs):
    """
    Constructs a AI85Net5 model.
    """
    assert not pretrained
    return AI85tcn(**kwargs)


models = [
    {
        'name': 'ai85tcn',
        'min_input': 1, #only useful for 2D models....
        'dim': 1, #the model handles 1D input
    }
]
