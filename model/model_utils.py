"""
Doc.:   Utility files for model / architecture
"""
import torch
from torch import nn
from torch.nn import init



# +++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++[Utility Functions]+++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++
def weights_init_kaiming(m, pi=0.01):
    """
    Doc.:   Custom weigh initialization method.
            Adapted from https://github.com/xx-adeline/MFPE/blob/main/src/model/model.py#L9
            to include RetinaNet-specific bias initialization https://arxiv.org/pdf/1708.02002.pdf

    Args.:  • pi: reference https://arxiv.org/pdf/1708.02002.pdf for details
    """
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        
        # Apply RetinaNet-specific bias initialization to the final convolutional layer
        # Reference: https://arxiv.org/pdf/1708.02002.pdf
        if hasattr(m, 'is_final_conv') and m.is_final_conv:
            bias_value = -init.log((1 - pi) / pi)
            init.constant_(m.bias.data, bias_value)

    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0)


        
# +++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++[Utility Classes]++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++
class Swish(nn.Module):
    def __init__(self, beta:float=1.0):
        """
        Doc.:   Swish activation function. https://arxiv.org/pdf/1710.05941.pdf
                Swish is an activation function f(x) = x·sigmoid(βx),  where β is a learnable parameter. 
                Nearly all implementations do not use the learnable parameter, in which case the activation 
                function is f(x) = x·sigmoid(βx). Ref: https://paperswithcode.com/method/swish

        Args.:  • beta: learnable or constant value
        """
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
    


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels:int=None, out_channels:int=None, kernel_size:int=3, bias:bool=True, is_final_conv:bool=False):
        """
        Doc.:   Custom depthwise separable convolution. https://arxiv.org/pdf/1610.02357.pdf
                Depthwise Separable Convolution splits the computation into two steps: depthwise convolution applies a 
                single convolutional filter per each input channel and pointwise convolution is used to create a 
                linear combination of the output of the depthwise convolution. https://paperswithcode.com/method/depthwise-separable-convolution

        Args.:  • in_channels: number of input channels
                • out_channels: number of output channels
                • bias: use bias or not
                • is_final_conv: treat this as the final layer or not. If last layer, use RetinaNet styled initialization. https://arxiv.org/pdf/1708.02002.pdf
        """
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_convolution = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size= kernel_size, groups=in_channels, bias=bias, padding=1)
        self.pointwise_convolution = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, groups=1, bias=bias)
        self.depthwise_separable_conv = torch.nn.Sequential(self.depthwise_convolution, self.pointwise_convolution)

        # Treat this as the final layer
        self.is_final_conv = False
        if bias and is_final_conv:
            self.is_final_conv = True

        # initialize some weights
        self.depthwise_separable_conv.apply(weights_init_kaiming)

    def forward(self, x):
        return self.depthwise_separable_conv(x)






class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            Swish(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DepthwiseSeparableConvWithSimpleAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, reduction_ratio=16):
        super().__init__()
        self.depthwise_separable_conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size)
        self.channel_attention = ChannelAttention(out_channels, reduction_ratio)

    def forward(self, x):
        x = self.depthwise_separable_conv(x)
        x = self.channel_attention(x)
        return x





