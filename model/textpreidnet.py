# System modules

# 3rd party modules
from torch import nn

# Aplication modules
from model.visual_network import VisualNetwork
from model.model_utils import DepthwiseSeparableConv


class TextReIDNet(nn.Module):
    def __init__(self,configs:dict=None):
        """
        Doc.:   The architecture of TextReIDNet. Deatils in TODO: provide paper ref


                Architecture Summary
                ---------------------
                                                                                    -----------------
                image >> EfficientNetb0 >> DSC >> AP >> DSC >>|=================|---| Ranking loss  |
                                                              | Joint embedding |   -----------------
                text  >>>>>>>> BERT >>>>>> GRU >> AP >> DSC >>|=================|---| Identity loss |
                                                                                    -----------------
                Legends
                --------------------------------------
                • DSC: Deepthwise Seperable Convolution
                • AP: Adaptive Pooling (max)
                • GRU: Gated Recurrent Unit

        """
        super(TextReIDNet, self).__init__()

        if configs is None:
            raise ValueError("`configs` parameter can not be None")
        
        self.configs = configs
        self.visual_network = VisualNetwork()


    def forward(self, x):
        ...
    