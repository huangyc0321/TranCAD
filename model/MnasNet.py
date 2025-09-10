import torch

#定义卷积规则
class conv(torch.nn.Module):
    def __init__(self,input_channel,kernal_size,output_channel,
                 stride = 1,groups = 1,activation = True):
        super().__init__()
        self.activation = activation
        self.padding = kernal_size // 2
        self.con1 = torch.nn.Conv2d(input_channel,output_channel,kernal_size,
                                    stride,padding=self.padding,groups=groups)
        self.BN = torch.nn.BatchNorm2d(output_channel)
        self.ReLu = torch.nn.ReLU()

    def forward(self,x):
        x = self.con1(x)
        x = self.BN(x)
        if self.activation == True:
            x = self.ReLu(x)
        return x

#深度可分离卷积层
class Sepconv(torch.nn.Module):
    def __init__(self,input_channel,output_channel,stride = 1):
        super().__init__()
        self.conv1 = conv(input_channel,3,input_channel,stride,groups=input_channel)
        self.conv2 = conv(input_channel,1,output_channel,stride,activation=False)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

#注意力机制
class SE_block(torch.nn.Module):
    def __init__(self,input_channel,ratio = 1):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv2 = torch.nn.Sequential(
            torch.nn.Linear(input_channel,input_channel * ratio),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(input_channel * ratio,input_channel),
            torch.nn.Hardsigmoid(inplace=True)
        )

    def forward(self,x):
        b,c,_,_ = x.shape
        weight = self.pool(x)
        weight = weight.view(b,c)
        weight = self.conv2(weight)
        weight = weight.view(b,c,1,1)
        return x * weight

#倒残差结构
class MB_conv(torch.nn.Module):
    def __init__(self,input_channel,kernal_size,output_channel,
                 stride = 1,use_attention = False,t = 3):
        super().__init__()
        self.attention = use_attention
        self.stride = stride
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.conv1 = conv(input_channel,1,input_channel * t)
        self.conv2 = conv(input_channel * t,kernal_size,input_channel * t,
                          stride=stride,groups=input_channel)
        self.conv3 = conv(input_channel * t,1,output_channel,activation=False)
        self.SE = SE_block(input_channel * t)



    def forward(self,x):
        input = self.conv1(x)
        input = self.conv2(input)
        if self.attention == True:
            input = self.SE(input)
        input = self.conv3(input)
        if self.stride == 1 and self.input_channel == self.output_channel:
            input += x
        return input

#Mnasnet
class Mnasnet(torch.nn.Module):
    def __init__(self,input_channel,num_classes):
        super().__init__()
        self.feature = torch.nn.Sequential(
            conv(input_channel,3,32,stride=2,activation=False),

            Sepconv(32,16),

            MB_conv(16,3,24,2,False,6),
            MB_conv(24,3,24,1,False,6),

            MB_conv(24,5,40,2,True,3),
            MB_conv(40,5,40,1,True,3),
            MB_conv(40,5,40,1,True,3),

            MB_conv(40,3,80,2,False,6),
            MB_conv(80,3,80,1,False,6),
            MB_conv(80, 3, 80, 1, False, 6),
            MB_conv(80, 3, 80, 1, False, 6),

            MB_conv(80,3,112,1,True,6),
            MB_conv(112,3,112,1,True,6),

            MB_conv(112,5,160,2,True,6),
            MB_conv(160,5,160,1,True,6),
            MB_conv(160,5,160,1,True,6),

            MB_conv(160,3,320,1,False,6),
            torch.nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(320,160),
            torch.nn.ReLU(),
            torch.nn.Linear(160,num_classes)
        )

    def forward(self,x):
        x = self.feature(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    x = torch.randn(1,3,224,224)
    model = Mnasnet(3,10)
    y = model(x)
    print(y.size())
