###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Wavenet (TCN) network that fit into AI84

"""
import torch.nn as nn

import ai8x

def _conv_stack(dilations, in_channels, out_channels, kernel_size,bias=True,**kwargs):
    """
    Create stack of dilated convolutional layers, outlined in WaveNet paper:
    https://arxiv.org/pdf/1609.03499.pdf
    """
    return nn.ModuleList(
        [
            ai8x.FusedConv1dReLU(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                dilation=d,
                bias=bias,
                **kwargs
            )
            for i, d in enumerate(dilations)
        ]
    )

class AI85tcn(nn.Module):
    """
    1D TCN
    """
    def __init__(self,num_classes=1, num_channels=1,dimensions=(128,),
                num_hidden_channels=12, dilation_depth=10, num_repeat=1,
                kernel_size=3,
                bias=True, **kwargs):
        """
        num_channels: num INPUT channels
        """
        super().__init__()

        #num_classes HAVE TO LEAVE IT HERE
        # Limits
        #assert planes + num_channels <= ai8x.dev.WEIGHT_INPUTS
        
        #bias=True #get bias from parameters

        dilation=1
        #num_hidden_channels = 64
        self.num_channels = num_channels

        dilations = [2 ** d for d in range(dilation_depth)] * num_repeat

        #create dilated conv stack
        self.hidden = _conv_stack(dilations, num_hidden_channels, num_hidden_channels, kernel_size,bias=bias)

        self.input_layer = ai8x.FusedConv1dReLU(
                in_channels=num_channels,#input channels
                out_channels=num_hidden_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                bias=bias,
                **kwargs
            )

        self.linear_mix = ai8x.Conv1d(
            in_channels=num_hidden_channels, #no skips * dilation_depth * num_repeat,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False, #force no bias false for the last layer
            wide=True,
            #wide=True, #32 bit output!
            **kwargs
        )

        #init weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""

        out = self.input_layer(x)

        for hidden in self.hidden:
            out = hidden(out)

        out = self.linear_mix(out)

        return out


def ai85wavenet(pretrained=False, **kwargs):
    """
    Constructs a AI85Net5 model.
    """
    assert not pretrained
    return AI85tcn(**kwargs)


models = [
    {
        'name': 'ai85wavenet',
        'min_input': 1, #only useful for 2D models....
        'dim': 1, #the model handles 1D input
    }
]
