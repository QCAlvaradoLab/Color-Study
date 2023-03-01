import torch
from torch import nn 
from torch.nn import functional as F
from torchsummary import summary

from torchvision.models import vgg19_bn

class DeconvNormActivation(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups = 1, padding=None, num_blocks=2, bias=True):
        
        super().__init__()
        
        self.block = nn.ModuleList([])

        for idx in range(num_blocks):
            deconv = nn.ConvTranspose2d(in_channels = in_channels if idx == 0 else out_channels, 
                                        out_channels = out_channels, 
                                        kernel_size = kernel_size, 
                                        stride = stride,
                                        groups = groups,
                                        padding=padding,
                                        bias=bias and idx > 0)
            batchnorm = nn.BatchNorm2d(out_channels)
            activation = nn.SiLU()
        
            self.block.append(deconv)
            self.block.append(batchnorm)
            self.block.append(activation)
    
    def forward(self, x):   
        
        for module in self.block:
            x = module(x)
        return x

class VGGUNetDecoder(nn.Module):
    
    def __init__(self, 
                 channels = [512, 512, 512, 512, 512, 256, 256, 128, 64], 
                 upsample = [True, False, False, True, False, True, False, True, True], 
                 num_blocks = 2, 
                 out_channels = 1):
        
        super().__init__()

        assert len(channels) == len(upsample) 

        #channels.append(out_channels)
        channels.insert(0, channels[0])

        self.channel_blocks = nn.ModuleList([
                                DeconvNormActivation(
                                        channels[idx] if idx!=1 or not upsample[idx] else channels[idx]*2, 
                                        channels[idx+1], 
                                        kernel_size=3, 
                                        stride=1, 
                                        padding=1,
                                        num_blocks = 0 if idx==0 else 2) \
                                    for idx in range(len(channels)-1)])

        self.conv_blocks = nn.ModuleList([
                                DeconvNormActivation(
                                        channels[idx+1], 
                                        channels[idx+1], 
                                        kernel_size=1, 
                                        stride=1, 
                                        padding=0,
                                        num_blocks = 0 if idx==0 else 2) \
                                    for idx in range(len(channels)-1)]) 
    
        self.final_conv = DeconvNormActivation(channels[-1], 1, 1, 1, padding=0)
        self.channels = channels
        self.upsample = upsample
    
    def forward(self, x, encoder_tensors):
           
        for index, (block1, block2, encoder_tensor) in \
            enumerate(zip(self.channel_blocks, self.conv_blocks, reversed(encoder_tensors))):

            if self.upsample[index]:
                x = F.interpolate(x, scale_factor=2)   
                x = torch.cat((encoder_tensor, x), dim=1)
            
            print (x.shape, block1)
            x = block1(x)
            print (x.shape, block2)
            x = block2(x)
            
        return self.final_conv(x)

class VGGUNetEncoder(nn.Module):
    
    def __init__(self, vgg_classifier, img_size=256):
        
        super().__init__()

        self.net = vgg_classifier.features
        
        self.feature_size = img_size

    def forward(self, x):
        
        forward_blocks = []

        for layer in self.net:

            if isinstance(layer, nn.MaxPool2d):
                forward_blocks.append(x)

            x = layer(x)

        return x, forward_blocks

class VGGUNet(nn.Module):
    
    def __init__(self, vgg_classifier, img_size=256):
      
        super().__init__()

        self.encoder = VGGUNetEncoder(vgg_classifier, img_size)
        self.decoder = VGGUNetDecoder()
    
    def forward(self, x):
        
        x, encoder_tensors = self.encoder.forward(x)
        x = self.decoder.forward(x, encoder_tensors)

        return x

if __name__ == "__main__":
    
    #net = nn.Sequential(vgg19_bn().features)
    #print (summary(net, (3, 256, 256)))
    
    net = vgg19_bn()
    #net = VGGUNetEncoder(net)
    #net.forward(torch.ones((1,3,256,256)))

    #decoder = VGGUNetDecoder()
    #print (summary(decoder, (512, 8, 8)))

    net = VGGUNet(net, 256)
    print (summary(net, (3,256,256)))
