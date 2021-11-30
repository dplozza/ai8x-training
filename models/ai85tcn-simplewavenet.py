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
SIMPLE wavenet (with NO res and NO skip connections)


"""
import torch
import torch.nn as nn

import ai8x


def error_to_signal(y, y_pred):
    """
    Error to signal ratio with pre-emphasis filter:
    https://www.mdpi.com/2076-3417/10/3/766/htm
    """
    y, y_pred = pre_emphasis_filter(y), pre_emphasis_filter(y_pred)
    return (y - y_pred).pow(2).sum(dim=2) / (y.pow(2).sum(dim=2) + 1e-10)


def pre_emphasis_filter(x, coeff=0.95):
    return torch.cat((x[:, :, 0:1], x[:, :, 1:] - coeff * x[:, :, :-1]), dim=2)


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
                kernel_size=3,dilation_power=2,
                bias=True, **kwargs):
        """
        num_channels: num INPUT channels
        """
        super().__init__()

        #print("SHIIIT",kwargs)

        #num_classes HAVE TO LEAVE IT HERE
        # Limits
        #assert planes + num_channels <= ai8x.dev.WEIGHT_INPUTS
        
        #bias=True #get bias from parameters

        #num_hidden_channels = 64
        self.num_channels = num_channels

        dilations = [dilation_power ** d for d in range(dilation_depth)] * num_repeat

        #create dilated conv stack
        self.hidden = _conv_stack(dilations, num_hidden_channels, num_hidden_channels, kernel_size,bias=bias)
        
        #self.residuals = _conv_stack(dilations, num_hidden_channels, num_hidden_channels, 1,bias=bias)

        #self.input_layer = ai8x.FusedConv1dReLU(
        # for first layer NO nonlinearity: simply linar mix
        self.input_layer = ai8x.Conv1d(
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
            in_channels=num_hidden_channels, #*dilation_depth*num_repeat, #no skips * dilation_depth * num_repeat,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=bias, #force no bias false for the last layer
            wide=True,
            #wide=True, #32 bit output!
            **kwargs
        )

        #init weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                pass
                #m.weight.data[:,:,:] = torch.tensor(0)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""

        self.outs = []
        
        skips = [] #stores skip connections

        out = self.input_layer(x)
        self.outs.append(out)

        #for hidden, residual in zip(self.hidden, self.residuals):
        for hidden  in  self.hidden:
            res = out
            out = hidden(out)

            
            skips.append(out) #append skip connections

            #out = residual(out)

            #out = out + res[:, :, -out.size(2) :]

            self.outs.append(out)

        #out = torch.cat([s[:, :, -out.size(2) :] for s in skips], dim=1)

        out = self.linear_mix(out)

        #change linear mix SIZE!!!

        return out

    def get_loss_criterion(self):
        """Creates and return custom loss function"""

        criterion = lambda y_pred,y: error_to_signal(y[:, :, -y_pred.size(2) :],y_pred).mean()
        #criterion = lambda y, y_pred: ((y - y_pred).pow(2).sum(dim=2) / (y.pow(2).sum(dim=2) + 1e-10)).mean()
        print("Using custom loss")
        return criterion


def ai85simplewavenet(pretrained=False, **kwargs):
    """
    Constructs a AI85Net5 model.
    """
    assert not pretrained
    return AI85tcn(**kwargs)


models = [
    {
        'name': 'ai85simplewavenet',
        'min_input': 1, #only useful for 2D models....
        'dim': 1, #the model handles 1D input
    }
]
