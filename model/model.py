# -*- coding: utf-8 -*-
from torch import nn
from model.text_feature_extract import TextExtract_gru
from torchvision import models
import torch
from torchvision.models import EfficientNet_B0_Weights
from torch.nn import init


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0)



class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_convolution = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, groups=in_channels, bias=bias, padding=1)
        self.pointwise_convolution = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, groups=1, bias=bias)
        self.depthwise_separable_conv = torch.nn.Sequential(self.depthwise_convolution, self.pointwise_convolution)

        # apply some intial weights
        self.depthwise_separable_conv.apply(weights_init_kaiming)

    def forward(self, x):
        return self.depthwise_separable_conv(x)
        



class ResNet34Backbone(torch.nn.Module):
    def __init__(self):
        super(ResNet34Backbone, self).__init__()
        # Load the pretrained ResNet34 model
        self.resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Modify the stride of the first convolutional layer in C5 (layer4)
        self.resnet34.layer4[0].conv1.stride = (1, 1)
        self.resnet34.layer4[0].downsample[0].stride = (1, 1)

    def forward(self, x):
        # Outputs will be collected in a list
        outputs = []

        # Forward pass through initial layers
        x = self.resnet34.conv1(x)
        x = self.resnet34.bn1(x)
        x = self.resnet34.relu(x)
        outputs.append(x)  # Output of C1

        x = self.resnet34.maxpool(x)

        # Forward pass through each of the four layers
        x = self.resnet34.layer1(x)
        outputs.append(x)  # Output of C2

        x = self.resnet34.layer2(x)
        outputs.append(x)  # Output of C3

        x = self.resnet34.layer3(x)
        outputs.append(x)  # Output of C4

        x = self.resnet34.layer4(x)
        outputs.append(x)  # Output of C5

        return outputs # C1, C2, C3, C4, C5
    


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, model_name='efficientnet_b0'):
        super(EfficientNetFeatureExtractor, self).__init__()
        # Load the pre-trained EfficientNet
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # final layer feature
        self.feature_blocks = self.model.features

    def forward(self, x):
        features = []

        # Stages 1 to 8
        for block in self.feature_blocks:
            x = block(x)
            features.append(x)

        return features[-1] # final layer output
    



class TextImgPersonReidNet(nn.Module):

    def __init__(self, opt):
        super(TextImgPersonReidNet, self).__init__()
        self.opt = opt
        
        # General stuff
        self.image_extractor_head = DepthwiseSeparableConv(1280, 1024)
        
        
        # Visual network stuff
        # =================================================
        # >> visual backbone
        # self.resnet34 = ResNet34Backbone()
        self.efficient_net = EfficientNetFeatureExtractor()

        # >> pooling: global and local visual features
        self.local_avgpool = nn.AdaptiveMaxPool2d((opt.part, 1))
        self.global_avgpool = nn.AdaptiveMaxPool2d((1, 1))
        
        # >> convolutions: global and local visual features
        # conv_local = []
        # for i in range(self.opt.part):
        #     conv_local.append(DepthwiseSeparableConv(1024, self.opt.feature_length))
        # self.conv_local = nn.Sequential(*conv_local)
        self.conv_global = DepthwiseSeparableConv(1024, self.opt.feature_length)

        # Language network stuff
        # >> word convolution
        self.TextExtract = TextExtract_gru(self.opt)
        self.conv_word_classifier = nn.Sequential(DepthwiseSeparableConv(1024, 6, bias=False),
                                                  nn.Sigmoid())
        
        # print("self.resnet34: {}".format(sum(p.numel() for p in self.resnet34.parameters()) / 1000000.0))
        # print("self.local_avgpool: {}".format(sum(p.numel() for p in self.local_avgpool.parameters()) / 1000000.0))
        # print("self.global_avgpool: {}".format(sum(p.numel() for p in self.global_avgpool.parameters()) / 1000000.0))
        # print("self.conv_local: {}".format(sum(p.numel() for p in self.conv_local.parameters()) / 1000000.0))
        # print("self.conv_global: {}".format(sum(p.numel() for p in self.conv_global.parameters()) / 1000000.0))
        # print("self.conv_global: {}".format(sum(p.numel() for p in self.conv_global.parameters()) / 1000000.0))
        # print("self.TextExtract: {}".format(sum(p.numel() for p in self.TextExtract.parameters()) / 1000000.0))
        # print("self.conv_word_classifier: {}".format(sum(p.numel() for p in self.conv_word_classifier.parameters()) / 1000000.0))


    def forward(self, image, caption_id, text_length):

        img_global, img_local = self.img_embedding(image)
        txt_global, txt_local = self.txt_embedding(caption_id, text_length)

        #return img_global, img_local, txt_global, txt_local
        return img_global, txt_global

    def img_embedding(self, image):

        # image_feature34 = self.resnet34(image)[4] # C5 output
        # image_feature_EFN = EfficientNetFeatureExtractor

        # print("Extracted: {}".format(image_feature34.shape))

        # image_feature34 = self.depthwise(image_feature34)

        # print("Dethwise: {}".format(image_feature34.shape))

        # image_feature = image_feature34 # self.ImageExtract(image)

        image_feature = self.efficient_net(image)
        image_feature = self.image_extractor_head(image_feature)

        image_feature_global = self.global_avgpool(image_feature)
        image_global = self.conv_global(image_feature_global).squeeze(-1).contiguous() #.squeeze(2)

        # image_feature_local = self.local_avgpool(image_feature)

        # image_local = []
        # for i in range(self.opt.part):
        #     image_feature_local_i = image_feature_local[:, :, i, :]
        #     image_feature_local_i = image_feature_local_i.unsqueeze(2)
        #     image_embedding_local_i = self.conv_local[i](image_feature_local_i).unsqueeze(2).contiguous()
        #     image_local.append(image_embedding_local_i)

        # image_local = torch.cat(image_local, 2)
        # image_local = image_local.squeeze(-1).squeeze(-1).contiguous()

        return image_global, 1#, image_local

    def txt_embedding(self, caption_id, text_length):

        text_feature = self.TextExtract(caption_id, text_length)

        text_global, _ = torch.max(text_feature, dim=2, keepdim=True)
        text_global = self.conv_global(text_global).squeeze(-1).contiguous()

        # text_feature_local = []
        # for text_i in range(text_feature.size(0)):
        #     text_feature_local_i = text_feature[text_i, :, :text_length[text_i]].unsqueeze(0).contiguous()

        #     word_classifier_score_i = self.conv_word_classifier(text_feature_local_i)

        #     word_classifier_score_i = word_classifier_score_i.permute(0, 3, 2, 1).contiguous()
        #     text_feature_local_i = text_feature_local_i.repeat(1, 1, 1, 6).contiguous()

        #     text_feature_local_i = text_feature_local_i * word_classifier_score_i

        #     text_feature_local_i, _ = torch.max(text_feature_local_i, dim=2)

        #     text_feature_local.append(text_feature_local_i)

        # text_feature_local = torch.cat(text_feature_local, dim=0)

        # text_local = []
        # for p in range(self.opt.part):
        #     text_feature_local_conv_p = text_feature_local[:, :, p].unsqueeze(2).unsqueeze(2).contiguous()
        #     text_feature_local_conv_p = self.conv_local[p](text_feature_local_conv_p).unsqueeze(2).contiguous()
        #     text_local.append(text_feature_local_conv_p)
        # text_local = torch.cat(text_local, dim=2)
        # text_local = text_local.squeeze(-1).squeeze(-1).contiguous()

        return text_global, 1 #, text_local

