import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d, panel=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.panel = panel
        if panel == True:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)       

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            if self.stride == 2 and self.panel == True:
                # out = self.relu(out)
                residual = self.downsample(x, out)
                return residual
            else:
                residual = self.downsample(x)
            
        # try:
        out += residual
        # except:
        #     print('x: ', x.shape)
        #     print('out: ', out.shape)
        #     print('residual: ', residual.shape)
        #     print(self.downsample)
        #     out += residual
            
        out = self.relu(out)

        return out

def activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == 'none':
        module = nn.Identity(1,1)
    elif name == 'prelu':
        module = nn.PReLU(init=0.5)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class BaseConv(nn.Module):
    """A Conv2d -> Norm -> Activation block"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, groups=1, dilation=False, bias=False, act="silu"):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        d = 1
        if dilation == True:
            d = pad + d
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            dilation=d,
            groups=groups,
            bias=bias,
            )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.conv(x)
    
class Panel(nn.Module):
    '''Panel Attention'''
    
    def __init__(self, in_channels, out_channels, stride=1, shuffle=True, nl=False):
        
        super(Panel, self).__init__()
            
        self.out_channels = out_channels
        self.stride = stride
        self.nl = nl
        self.shuffle = shuffle
        
        self.c_ffc = nn.Conv2d(int(in_channels*1), int(out_channels), kernel_size=1, groups=1, bias=False)
            
        if self.stride == 2:            
            self.sc_x_downsample = BaseConv(int(out_channels), int(out_channels), ksize=2, stride=2, act='prelu')
            
        if self.shuffle == True:
            
            if self.stride == 2:
                self.s_ffc = nn.Sequential(
                    nn.Conv2d(int(out_channels*4), int(out_channels*4), kernel_size=3, padding=1, groups=int(out_channels*4), bias=False),
                    nn.Conv2d(int(out_channels*4), int(out_channels*1), kernel_size=1, padding=0, groups=1, bias=False),
                    nn.GroupNorm(1,int(out_channels*1)),
                    nn.BatchNorm2d(int(out_channels*1)),                    
                    nn.PReLU(init=0.5)
                    )
            
            if self.stride != 2:           
                self.s_ffc = nn.Sequential(
                    nn.Conv2d(int(out_channels*4), int(out_channels*4), kernel_size=3, padding=1, groups=int(out_channels*4), bias=False),
                    nn.Conv2d(int(out_channels*4), int(out_channels*4), kernel_size=1, padding=0, groups=1, bias=False),
                    nn.GroupNorm(1,int(out_channels*4)),
                    nn.BatchNorm2d(int(out_channels*4)),
                    nn.PReLU(init=0.5)
                    )
            
            if self.nl == True:
                
                self.s_ffc = nn.Conv2d(int(out_channels*4), int(out_channels*4), kernel_size=3, padding=1, groups=int(out_channels*4), bias=False)
                
                self.s_ffcx = nn.Conv2d(int(out_channels*1), int(out_channels*1), kernel_size=1, padding=0, groups=1, bias=False)                
                self.s_ffcq = nn.Conv2d(int(out_channels*1), int(out_channels*1), kernel_size=1, padding=0, groups=1, bias=False)
                self.s_ffck = nn.Conv2d(int(out_channels*1), int(out_channels*1), kernel_size=1, padding=0, groups=1, bias=False)
                self.s_ffcv = nn.Conv2d(int(out_channels*1), int(out_channels*1), kernel_size=1, padding=0, groups=1, bias=False)
                
                if self.stride == 2:
                    self.act = nn.Sequential(
                        nn.GroupNorm(1,int(out_channels*1)),
                        nn.BatchNorm2d(int(out_channels*1)),
                        nn.PReLU(init=0.5)
                        )    
                if self.stride != 2:    
                    self.act = nn.Sequential(
                        nn.GroupNorm(1,int(out_channels*4)),
                        nn.BatchNorm2d(int(out_channels*4)),
                        nn.PReLU(init=0.5)
                        )    
        
        else:
            if self.stride == 2:
                self.s_ffc = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.BatchNorm2d(int(out_channels*1)),                    
                    nn.PReLU(init=0.5)
                    )
            if self.stride != 2:
                self.s_ffc = nn.Sequential(
                    nn.BatchNorm2d(int(out_channels*1)),                    
                    nn.PReLU(init=0.5)
                    )

            
    def forward(self, x, sc_x):    
       
        '''CPA, channel pixel attention'''
        _, c, w, h = x.shape
        c_out = self.c_ffc(x)
        
        '''PuSA, pixel-unshuffle attention'''
        if self.shuffle == True:
            if w % 2:
                c_out = F.interpolate(c_out, int(w+1), mode='nearest')
        
            s_out = F.pixel_unshuffle(c_out, 2)
            
            s_out = self.s_ffc(s_out)
        
            '''Panel, PuSA + non-local spatial attention'''
            if self.nl == True:                
                b, s_c, s_w, s_h = s_out.shape
                
                x_steps = torch.arange(0,s_c,4,device=s_out.device)
                q_steps = x_steps + 1
                k_steps = x_steps + 2
                v_steps = x_steps + 3
                s_x = torch.index_select(s_out, 1, x_steps)
                s_q = torch.index_select(s_out, 1, q_steps)
                s_k = torch.index_select(s_out, 1, k_steps)
                s_v = torch.index_select(s_out, 1, v_steps)
    
                s_q = self.s_ffcq(s_q).reshape(b,-1,s_w*s_h).permute(0,2,1)
                s_k = self.s_ffck(s_k).reshape(b,-1,s_w*s_h)
                s_v = self.s_ffcv(s_v).reshape(b,-1,s_w*s_h).permute(0,2,1)               
                
                s_qk = torch.matmul(s_q, s_k)
                s_qk = F.softmax(s_qk, -1)
                
                s_qkv = torch.matmul(s_qk, s_v).permute(0,2,1).contiguous()
                s_qkv = s_qkv.reshape(b,-1,s_w,s_h)
                
                if self.stride == 2:
                    s_out = self.s_ffcx(s_x + s_qkv)                
                    
                if self.stride != 2:
                    s_out = self.s_ffcx(s_x + s_qkv)
                    s_x = s_out
                    s_q = s_out
                    s_k = s_out
                    s_v = s_out
                    s_out = torch.cat([s_x, s_q, s_k, s_v], 1)
                    
                s_out = self.act(s_out)
        
            if self.stride != 2:
                s_out = F.pixel_shuffle(s_out, 2)
                if w % 2:
                    s_out = F.interpolate(s_out, int(w), mode='nearest')
        
        else:
            s_out = self.s_ffc(c_out)
        
        if self.stride == 2:
            sc_x = self.sc_x_downsample(sc_x)
            
        if s_out.shape == sc_x.shape:
            out = sc_x + s_out
            
        else:
            print('none')
            print('stride: ', self.stride)
            print('sc_x: ', sc_x.shape)
            print('s_out: ', s_out.shape)
            print('c_out: ', c_out.shape)
        
        # out = s_out
        
        return out

class ResNet_Panel(nn.Module):
    """ Adapted from torchvision.models.resnet """

    def __init__(self, layers, block=Bottleneck, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        self.norm_layer = norm_layer
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._make_layer(block, 64, layers[0])
        self._make_layer(block, 128, layers[1], stride=2, panel=True)
        self._make_layer(block, 256, layers[2], stride=2, panel=True)
        self._make_layer(block, 512, layers[3], stride=2, panel=True)

        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    def _make_layer(self, block, planes, blocks, stride=1, panel=False):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       self.norm_layer(planes * block.expansion))
            if stride == 2 and panel == True:
                downsample = Panel(self.inplanes, planes * block.expansion, stride=2, shuffle=True, nl=True)
            
        layers = [block(self.inplanes, planes, stride, downsample, self.norm_layer, panel=panel)]
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer))

        layer = nn.Sequential(*layers)

        self.channels.append(planes * block.expansion)
        self.layers.append(layer)

    def forward(self, x):
        """ Returns a list of convouts for each layer. """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            outs.append(x)

        return tuple(outs)

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=True)
        print(f'\nBackbone is initiated with {path}.\n')
