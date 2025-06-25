import torch
from torch import nn
from einops import rearrange, repeat

class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):    
        super(DepthWiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class AdaptiveCrossingConv(nn.Module):
    def __init__(self, in_channels, kernel_size_0, kernel_size_1, stride=1):
        super(AdaptiveCrossingConv, self).__init__()
        self.in_channels = in_channels
        self.norm0 = nn.GroupNorm(2,in_channels)
        self.paddings_0 = [nn.ZeroPad2d((kernel_size_0[0], 0, kernel_size_0[1], 0)), nn.ZeroPad2d((0, kernel_size_0[0], 0, kernel_size_0[1])), 
            nn.ZeroPad2d((0, kernel_size_0[1], kernel_size_0[0], 0)), nn.ZeroPad2d((kernel_size_0[1], 0, 0, kernel_size_0[0]))]
        self.paddings_1 = [nn.ZeroPad2d((kernel_size_1[0], 0, kernel_size_1[1], 0)), nn.ZeroPad2d((0, kernel_size_1[0], 0, kernel_size_1[1])), 
            nn.ZeroPad2d((0, kernel_size_1[1], kernel_size_1[0], 0)), nn.ZeroPad2d((kernel_size_1[1], 0, 0, kernel_size_1[0]))]
        self.conv_w0 = DepthWiseSeparableConv(in_channels//2, in_channels//4, (kernel_size_0[1],kernel_size_0[0]), stride=stride, padding=0)
        self.conv_h0 = DepthWiseSeparableConv(in_channels//2, in_channels//4, (kernel_size_0[0],kernel_size_0[1]), stride=stride, padding=0)
        self.conv_w1 = DepthWiseSeparableConv(in_channels//2, in_channels//4, (kernel_size_1[1],kernel_size_1[0]), stride=stride, padding=0)
        self.conv_h1 = DepthWiseSeparableConv(in_channels//2, in_channels//4, (kernel_size_1[0],kernel_size_1[1]), stride=stride, padding=0)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()
        self.conv_squeeze = DepthWiseSeparableConv(2,2,kernel_size=7,stride=1,padding=3)
        final_kernel = max([kernel_size_0[1],kernel_size_1[1]])*2
        self.last_conv = DepthWiseSeparableConv(in_channels,in_channels,kernel_size=final_kernel*2,stride=1,padding=final_kernel-1)


    def forward(self, x):
        x = self.norm0(x)
        x0, x1 = torch.chunk(x, 2, dim=1)
        x0_w0 = self.conv_w0(self.paddings_0[0](x0))
        x0_w1 = self.conv_w0(self.paddings_0[1](x0))
        x0_h0 = self.conv_h0(self.paddings_0[2](x0))
        x0_h1 = self.conv_h0(self.paddings_0[3](x0))
        x1_w0 = self.conv_w1(self.paddings_1[0](x1))
        x1_w1 = self.conv_w1(self.paddings_1[1](x1))
        x1_h0 = self.conv_h1(self.paddings_1[2](x1))
        x1_h1 = self.conv_h1(self.paddings_1[3](x1))
        x0 = self.act(self.norm1(torch.cat([x0_w0, x0_w1, x0_h0, x0_h1], dim=1)))
        x1 = self.act(self.norm2(torch.cat([x1_w0, x1_w1, x1_h0, x1_h1], dim=1)))
        attn = torch.cat([x0,x1],dim=1)
        avg_attn = torch.mean(attn,dim=1,keepdim=True)
        max_attn, _ = torch.max(attn,dim=1,keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        out = x0 * sig[:,0,:,:].unsqueeze(1) + x1 * sig[:,1,:,:].unsqueeze(1)
        out = self.last_conv(out)
        return x * out
    
class ChannelBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,7), stride=(1,1),padding=(0,3),
                       bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2,1), stride=(1,1),padding=(0,0),
                       bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        attn = torch.cat([avg_out,max_out],dim=3)
        attn = rearrange(attn,'b c h w -> b h w c')
        attn = self.fc0(attn)
        attn = rearrange(attn,'b h w c -> b c h w')
        attn = self.sigmoid(attn)
        return x * attn
    
class DeepCrossingAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_0=(7,3), kernel_size_1=(9,4)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial = AdaptiveCrossingConv(in_channels, kernel_size_0=kernel_size_0, kernel_size_1=kernel_size_1, stride=1)
        self.channel = ChannelBlock()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        attn0 = self.spatial(x)
        attn1 = self.channel(x)
        attn = attn0 + attn1
        out = self.conv(attn)
        return out
    
if __name__ == '__main__':
    x = torch.randn(2, 1024, 32, 32)
    model = DeepCrossingAttention(1024,256)
    y = model(x)
    #print(y.shape)