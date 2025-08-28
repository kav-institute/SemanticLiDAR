import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

from .ConvNextV2 import ConvNeXtV2


class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Define convolutional layers for query, key, and value
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Define convolutional layer for attention scores
        self.attention_conv = nn.Conv2d(out_channels, 1, kernel_size=1)
        
        # Softmax layer
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, features):
        # Query, key, and value transformations
        query = self.query_conv(features)
        key = self.key_conv(features)
        value = self.value_conv(features)
        
        # Compute attention scores
        attention_scores = self.attention_conv(torch.tanh(query + key))
        
        # Apply softmax to get attention weights
        attention_weights = self.softmax(attention_scores)
        
        # Apply attention to the value
        attended_features = value * attention_weights
        
        return attended_features

# class AttentionModule(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(AttentionModule, self).__init__()
#         self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.attention_conv = nn.Conv2d(out_channels, 1, kernel_size=1)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, features):
#         query = self.query_conv(features)
#         key = self.key_conv(features)
#         value = self.value_conv(features)
        
#         attention_scores = self.attention_conv(torch.tanh(query + key))
#         B, _, H, W = attention_scores.shape
#         attention_weights = self.softmax(attention_scores.view(B, 1, -1)).view(B, 1, H, W)
        
#         attended = value * attention_weights.expand_as(value)
#         return attended

class SemanticNetworkWithFPN(nn.Module):#
    """
    Semantic Segmentation Network with Feature Pyramid Network (FPN) using a ResNet backbone.

    Args:
        backbone (str): Type of ResNet model to use. Supported types: 'resnet18', 'resnet34', 'resnet50', 'resnet101'.
        meta_channel_dim (int): Number meta channels used in the FPN
        interpolation_mode (str): Interpolation mode used to resize the meta channels (default='nearest'). Supported types: 'nearest', 'bilinear', 'bicubic'.
        num_classes (int): Number of semantic classes
        no_bn (bool): Option to disable batchnorm (not recommended)
        attention (bool): Option to use a attention mechanism in the FPN (see: https://arxiv.org/pdf/1706.03762.pdf)
        
    """
    def __init__(self, backbone='resnet18', input_channels=2, meta_channel_dim=3, interpolation_mode = 'nearest', num_classes = 3, attention=True, multi_scale_meta=True):
        super(SemanticNetworkWithFPN, self).__init__()

        self.backbone_name = backbone
        self.interpolation_mode = interpolation_mode
        self.num_classes = num_classes
        self.attention = attention
        self.multi_scale_meta = multi_scale_meta
        # Load pre-trained ResNet model
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            base_channel = 512  # Number of channels in the last layer of ResNet18
            base_channels = [base_channel, base_channel // 2, base_channel // 4, base_channel // 8, base_channel // 16]
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            base_channel = 512  # Number of channels in the last layer of ResNet34
            base_channels = [base_channel, base_channel // 2, base_channel // 4, base_channel // 8, base_channel // 16]
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            base_channel = 2048  # Number of channels in the last layer of ResNet50
            base_channels = [base_channel, base_channel // 2, base_channel // 4, base_channel // 8, base_channel // 16]
        elif backbone == 'regnet_y_400mf':
            self.backbone = models.regnet_y_400mf(pretrained=True)
            base_channels = [440, 208, 104, 48, 32]
        elif backbone == 'regnet_y_800mf':
            self.backbone = models.regnet_y_800mf(pretrained=True)
            base_channels = [784, 320, 144, 64, 32]
        elif backbone == 'regnet_y_1_6gf':
            self.backbone = models.regnet_y_1_6gf(pretrained=True)
            base_channels = [888, 336, 120, 48, 32]
        elif backbone == 'regnet_y_3_2gf':
            self.backbone = models.regnet_y_3_2gf(pretrained=True)
            base_channels = [1512, 576, 216, 72, 32]
        elif backbone == 'shufflenet_v2_x0_5':
            self.backbone = models.shufflenet_v2_x0_5(pretrained=True)
            base_channels = [1024, 192, 96, 48, 24]
        elif backbone == 'shufflenet_v2_x1_0':
            self.backbone = models.shufflenet_v2_x1_0(pretrained=True)
            base_channels = [1024, 464, 232, 116, 24]
        elif backbone == 'shufflenet_v2_x1_5':
            self.backbone = models.shufflenet_v2_x1_5(pretrained=True)
            base_channels = [1024, 704, 352, 176, 24]
        elif backbone == 'shufflenet_v2_x2_0':
            self.backbone = models.shufflenet_v2_x2_0(pretrained=True)
            base_channels = [2048, 976, 488, 244, 112]
        elif backbone == 'squeezenet1_0':
            self.backbone = models.squeezenet1_0(pretrained=True)
            base_channels = [512, 384, 256, 256, 112]
        elif backbone == 'efficientnet_v2_s':
            # Use EfficientNetV2-S from torchvision
            self.backbone = models.efficientnet_v2_s(pretrained=True)
            base_channels = [128, 128, 64, 48, 168] 
        elif backbone == 'efficientnet_v2_m':
            # Use EfficientNetV2-S from torchvision
            self.backbone = models.efficientnet_v2_m(pretrained=True)
            base_channels = [160, 160, 80, 48, 168] 
        elif backbone == 'efficientnet_v2_l':
            # Use EfficientNetV2-S from torchvision
            self.backbone = models.efficientnet_v2_l(pretrained=True)
            base_channels = [192, 192, 96, 64, 168] 
        elif backbone == 'convnextv2_atto':
            self.backbone = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320],in_chans=input_channels+meta_channel_dim)
            base_channels = [320, 160, 80, 40, 20]
        elif backbone == 'convnextv2_femto':
            self.backbone = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384],in_chans=input_channels+meta_channel_dim)
            base_channels = [384, 192, 96, 48, 24]
        elif backbone == 'convnextv2_pico':
            self.backbone = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512],in_chans=input_channels+meta_channel_dim)
            base_channels = [512, 256, 128, 64, 32]
        elif backbone == 'convnextv2_nano':
            self.backbone = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640],in_chans=input_channels+meta_channel_dim)
            base_channels = [640, 320, 160, 80, 40]
        elif backbone == 'convnextv2_tiny':
            self.backbone = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],in_chans=input_channels+meta_channel_dim)
            base_channels = [768, 384, 192, 96, 48]
        elif backbone == 'convnextv2_base':
            self.backbone = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024],in_chans=input_channels+meta_channel_dim)
            base_channels = [1024, 512, 256, 128, 64]
        elif backbone == 'convnextv2_huge':
            self.backbone = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816],in_chans=input_channels+meta_channel_dim)
            base_channels = [2816, 1408, 704, 352, 176]
        elif backbone == 'convnextv2_large':
            self.backbone = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536],in_chans=input_channels+meta_channel_dim)
            base_channels = [1536, 768, 384, 192, 96]
        else:
            raise ValueError("Invalid ResNet type. Supported types: 'resnet18', 'resnet34', 'resnet50', 'regnet_y_400mf','regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0.")
        
        # Modify the first convolution layer to take 1+meta_channel_dim channels
        self.meta_channel_dim = meta_channel_dim

        is_shuffle = False
        is_squeeze = False
        # extract features from resnet family
        if backbone in ['resnet18', 'resnet34', 'resnet50']:
            self.backbone.conv1 = nn.Conv2d(input_channels + meta_channel_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)

            # Extract feature maps from different layers of ResNet
            self.stem = nn.Sequential(self.backbone.conv1, self.backbone.relu, self.backbone.maxpool)
            self.layer1 = self.backbone.layer1
            self.layer2 = self.backbone.layer2
            self.layer3 = self.backbone.layer3
            self.layer4 = self.backbone.layer4

        elif backbone in ['convnextv2_atto', 'convnextv2_femto', 'convnextv2_pico', 'convnextv2_nano', 'convnextv2_tiny', 'convnextv2_base', 'convnextv2_large', 'convnextv2_huge']:

            # Extract feature maps from different layers of ResNet
            self.stem = self.backbone.stem
            self.layer1 = self.backbone.layer1
            self.layer2 = self.backbone.layer2
            self.layer3 = self.backbone.layer3
            self.layer4 = self.backbone.layer4

        elif backbone in ["squeezenet1_0"]:
            num_channels = 64 if backbone=="squeezenet1_1" else 96
            self.backbone.features[0] = nn.Conv2d(
            2 + meta_channel_dim,  # Adjust the number of input channels
            num_channels,                    # Number of output channels in the first convolutional layer
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
            self.stem = self.backbone.features[0:4]
            self.layer1 = self.backbone.features[4:6]
            self.layer2 = self.backbone.features[6:8]
            self.layer3 = self.backbone.features[8:10]
            self.layer4 = self.backbone.features[10:]
            is_squeeze = True
            
        # extract features from regnet family
        elif backbone in ['regnet_y_400mf','regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf']:
            self.backbone.stem[0] = nn.Conv2d(input_channels + meta_channel_dim, self.backbone.stem[0].out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            # Extract feature maps from different layers of RegNet
            self.stem = self.backbone.stem
            self.layer1 = self.backbone.trunk_output[0]
            self.layer2 = self.backbone.trunk_output[1]
            self.layer3 = self.backbone.trunk_output[2]
            self.layer4 = self.backbone.trunk_output[3]

        elif backbone in ["shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0"]:
            self.backbone.conv1[0] = nn.Conv2d(input_channels + meta_channel_dim, self.backbone.conv1[0].out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            #help_conv = nn.Conv2d(2 + meta_channel_dim, self.backbone.conv1[0].out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            # Extract feature maps using indices or named stages from ShuffleNet
            self.stem = self.backbone.conv1
            self.layer1 = self.backbone.stage2
            self.layer2 = self.backbone.stage3
            self.layer3 = self.backbone.stage4
            self.layer4 = self.backbone.conv5
            is_shuffle = True

        elif backbone in ["efficientnet_v2_s", "efficientnet_v2_m","efficientnet_v2_l"]:
            self.backbone.features[0][0] = nn.Conv2d(input_channels + meta_channel_dim, self.backbone.features[0][0].out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            #help_conv = nn.Conv2d(2 + meta_channel_dim, self.backbone.conv1[0].out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            # Extract feature maps using indices or named stages from ShuffleNet
            self.stem = self.backbone.features[0]
            self.layer1 = self.backbone.features[2]
            self.layer2 = self.backbone.features[3]
            self.layer3 = self.backbone.features[4]
            self.layer4 = self.backbone.features[6:]
            is_shuffle = True

        # Attention blocks
        self.attention4 = AttentionModule(base_channels[1], base_channels[1])
        self.attention3 = AttentionModule(base_channels[2], base_channels[2])
        self.attention2 = AttentionModule(base_channels[3], base_channels[3])
        self.attention1 = AttentionModule(base_channels[4], base_channels[4])
        

        # FPN blocks
        self.fpn_block4 = self._make_fpn_block(base_channels[0], base_channels[1])
        self.fpn_block3 = self._make_fpn_block(base_channels[1], base_channels[2])
        self.fpn_block2 = self._make_fpn_block(base_channels[2], base_channels[3])
        self.fpn_block1 = self._make_fpn_block(base_channels[3], base_channels[4])

        # upsamle layers
        if is_shuffle:
            self.upsample_layer_x4 = nn.ConvTranspose2d(in_channels=base_channels[1], out_channels=base_channels[1]//4, kernel_size=4, stride=4, padding=0)
            self.upsample_layer_x3 = nn.ConvTranspose2d(in_channels=base_channels[2], out_channels=base_channels[2]//4, kernel_size=4, stride=4, padding=0)
            self.upsample_layer_x2 = nn.ConvTranspose2d(in_channels=base_channels[3], out_channels=base_channels[3]//2, kernel_size=2, stride=2, padding=0)
            out_channels_upsample = base_channels[1]//4 + base_channels[2]//4 + base_channels[3]//2
        elif is_squeeze:

            self.upsample_layer_x4 = nn.ConvTranspose2d(in_channels=base_channels[1], out_channels=base_channels[1]//4, kernel_size=4, stride=4, padding=0)
            self.upsample_layer_x3 = nn.ConvTranspose2d(in_channels=base_channels[2], out_channels=base_channels[2]//2, kernel_size=2, stride=2, padding=0)
            self.upsample_layer_x2 = nn.ConvTranspose2d(in_channels=base_channels[3], out_channels=base_channels[3]//2, kernel_size=2, stride=2, padding=0)
            out_channels_upsample = base_channels[1]//4 + base_channels[2]//2 + base_channels[3]//2

        else:
            self.upsample_layer_x4 = nn.ConvTranspose2d(in_channels=base_channels[1], out_channels=base_channels[1]//8, kernel_size=8, stride=8, padding=0)
            self.upsample_layer_x3 = nn.ConvTranspose2d(in_channels=base_channels[2], out_channels=base_channels[2]//4, kernel_size=4, stride=4, padding=0)
            self.upsample_layer_x2 = nn.ConvTranspose2d(in_channels=base_channels[3], out_channels=base_channels[3]//2, kernel_size=2, stride=2, padding=0)
            out_channels_upsample = base_channels[1]//8 + base_channels[2]//4 + base_channels[3]//2
        


        self.decoder_semantic = nn.Sequential(
            nn.Conv2d(out_channels_upsample + base_channels[4], base_channels[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels[4], base_channels[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels[4]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels[4], self.num_classes, kernel_size=4, stride=2, padding=1),
            nn.ELU(inplace=True)
        )

    def _make_fpn_block(self, in_channels, out_channels):
        """
        Create an FPN block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.Module: FPN block.
        """

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
            
    def forward(self, x, meta_channel):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor representing depth prediction.
        """
        
        
        # Inject meta channel before ResNet layers
        #if self.meta_channel_dim > 0:
        if self.multi_scale_meta:
            # Resize Meta Channels
            # Downsample the meta channel
            meta_channel1 = F.interpolate(meta_channel, scale_factor=1/2, mode=self.interpolation_mode)
            meta_channel2 = F.interpolate(meta_channel, scale_factor=1/4, mode=self.interpolation_mode)
            meta_channel3 = F.interpolate(meta_channel, scale_factor=1/8, mode=self.interpolation_mode)
            
            if self.backbone_name in ["squeezenet1_0"]:
                x = torch.cat([x, meta_channel], dim=1)
                xs = self.stem(x)
                x1 = self.layer1(xs)
                x = torch.cat([x1[:,0:-self.meta_channel_dim,...], meta_channel1], dim=1)
                x2 = self.layer2(x)
                x = torch.cat([x2[:,0:-self.meta_channel_dim,...], meta_channel2], dim=1)
                x3 = self.layer3(x)
                x4 = self.layer4(x3)
            elif self.backbone_name in ["efficientnet_v2_s","efficientnet_v2_m","efficientnet_v2_l"]:
                x = torch.cat([x, meta_channel], dim=1)
                xs = self.stem(x)
                x1 = self.layer1(xs)
                x = torch.cat([x1[:,0:-self.meta_channel_dim,...], meta_channel1], dim=1)
                x2 = self.layer2(x)
                x = torch.cat([x2[:,0:-self.meta_channel_dim,...], meta_channel2], dim=1)
                x3 = self.layer3(x)
                x4 = torch.cat([x3[:,0:-self.meta_channel_dim,...], meta_channel3], dim=1)
            else:
                x = torch.cat([x, meta_channel], dim=1)
                xs = self.stem(x)
                x1 = self.layer1(xs)
                x = torch.cat([x1[:,0:-self.meta_channel_dim,...], meta_channel1], dim=1)
                x2 = self.layer2(x)
                x = torch.cat([x2[:,0:-self.meta_channel_dim,...], meta_channel2], dim=1)
                x3 = self.layer3(x)
                x = torch.cat([x3[:,0:-self.meta_channel_dim,...], meta_channel3], dim=1)
                x4 = self.layer4(x)

            
        else:
        
            # Encoder (ResNet)
            x = torch.cat([x, meta_channel], dim=1)
            xs = self.stem(x)
            x1 = self.layer1(xs)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)

        # FPN
        x4 = self.fpn_block4(x4)
        x3 = self.fpn_block3(x3)
        x2 = self.fpn_block2(x2)
        x1 = self.fpn_block1(x1)
        
        # Attention
        if self.attention:
            x4 = self.attention4(x4)
            x3 = self.attention3(x3)
            x2 = self.attention2(x2)
            x1 = self.attention1(x1)

        

        x4 = self.upsample_layer_x4(x4)
        x3 = self.upsample_layer_x3(x3)
        x2 = self.upsample_layer_x2(x2)

        # Concatenate feature maps

        x = torch.cat([x1, x2, x3, x4], dim=1)


        # Decoder
        x_semantics = self.decoder_semantic(x) + 1 # offset of 1 to shift elu to ]0,inf[
        
        return x_semantics
    
if __name__ == "__main__":
    # 'resnet18'
    # 'shufflenet_v2_x0_5'
    # 'resnet50'
    # 'shufflenet_v2_x1_5'
    # 'resnet34'
    # 'shufflenet_v2_x1_0'
    # 'efficientnet_v2_l'
    # 'efficientnet_v2_m'
    # 'efficientnet_v2_s'
    # 'squeezenet1_0'
    # 'regnet_y_3_2gf'
    # 'regnet_y_1_6gf'
    # 'regnet_y_400mf'
    # 'regnet_y_800mf'
    import time
    import numpy as np
    model = SemanticNetworkWithFPN(num_classes=20, backbone="convnextv2_base", input_channels=2, meta_channel_dim=6, interpolation_mode = 'nearest', attention=True, multi_scale_meta=True).cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", pytorch_total_params / 1000000, "M")
    # Timer
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    inference_times = []
    model.eval()
    bs = 1
    with torch.no_grad():
        for i in range(1000):
            inputs = torch.randn(bs, 2, 128, 2048).to(torch.float32).cuda()
            meta = torch.randn(bs, 6, 128, 2048).to(torch.float32).cuda()
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
            start.record()
            outputs = model(inputs, meta)
            end.record()
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
            print("inference took {} ms".format(start.elapsed_time(end)))
            
            inference_times.append(start.elapsed_time(end))
            #time.sleep(0.15)
    print("inference mean {} ms".format(np.median(inference_times)))
