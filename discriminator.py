import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,c_dim , df_dim):
        super(Discriminator,self).__init__()

        self.conv0 = nn.Conv2d(c_dim, df_dim , 4, 2, 1 ,bias = False)
        self.elu0 = nn.ELU(inplace = True)


        self.conv1 = nn.Conv2d(df_dim , df_dim *2 , kernel_size= 4 ,stride = 2 , padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(df_dim*2)
        self.elu1 = nn.ELU(inplace = True)

        self.conv2 = nn.Conv2d(df_dim , df_dim *2 , kernel_size= 4 ,stride = 2 , padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(df_dim*2)
        self.elu2 = nn.ELU(inplace = True)

        self.conv3 = nn.Conv2d(df_dim , df_dim *2 , kernel_size= 4 ,stride = 2 , padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(df_dim*2)
        self.elu3 = nn.ELU(inplace = True)

        self.conv4 = nn.Conv2d(df_dim*8, 1 , kernel_size=4, stride=1 ,padding= 0 ,bias = False)
        self.sigmoid = nn.Sigmoid()

        for module in self.modules():
            if isinstance(module , nn.Conv2d):
                module.weight.data.normal_(mean=0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, inp):
        h0 = self.elu0(self.conv0(inp))
        h1 = self.elu1(self.bn1(self.conv1(h0)))
        h2 = self.elu2(self.bn2(self.conv2(h1)))
        h3 = self.elu3(self.bn3(self.conv3(h2)))
        h4 = self.conv4(h3)
        out = self.sigmoid(h4)
        return h3,out.view(-1,1).squeeze(1) # by squeeze get just not float Tensor

