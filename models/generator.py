import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim , c_dim , gf_dim):
        super(Generator, self).__init__()

        # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
        #                          output_padding=0,groups=1, bias=True, dilation=1, padding_mode='zeros')

        self.trans0 = nn.ConvTranspose2d(z_dim, gf_dim * 7 , 4 ,1 , 0 , bias = False ) # no bias term when applying Batch Normalization
        self.bn0 = nn.BatchNorm2d(gf_dim * 8) # 왜 곱하기 8 ?
        self.elu0 = nn.ELU(inplace = True)


        self.trans1 = nn.ConvTranspose2d(in_channels= gf_dim*8 , out_channels= gf_dim * 4 , kernel_size = 4 ,
                                         stride = 2 , padding = 1 , bias = False)
        self.bn1 = nn.BatchNorm2d(gf_dim * 4)
        self.elu1 = nn.ELU(inplace = True)


        self.trans2 = nn.ConvTranspose2d(in_channels = gf_dim * 4 , out_channels = gf_dim * 2 , kernel_size= 4,
                                         stride = 2 , padding = 1 , bias = False)
        self.bn2 = nn.BatchNorm2d(gf_dim * 4)
        self.elu2 = nn.ELU(inplace = True)


        self.trans4 = nn.ConvTranspose2d(in_channels= gf_dim*2 , out_channels= gf_dim , kernel_size= 4,
                                         stride = 2 , padding = 1 , bias = False)
        self.tanh = nn.Tanh()

        for module in self.modules():
            if isinstance(module , nn.ConvTranspose2d):
                module.weight.data.normal_(mean=0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self,inp):
        h0 = self.elu0(self.bn0(self.trans0(inp)))
        h1 = self.elu1(self.bn1(self.trans1(h0)))
        h2 = self.elu2(self.bn2(self.trans2(h1)))
        h3 = self.elu3(self.bn3(self.trans3(h2)))
        h4 = self.trans4(h3)
        out = self.tanh(h4)
        return out # (c_dim , 64 , 64 )