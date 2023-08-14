#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class DWConv(nn.Module):
    """Depthwise Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, dilation=False, act="silu"):
        super().__init__()
        
        pad = (ksize - 1) // 2
        self.dconv = nn.Conv2d(in_channels, in_channels, kernel_size=ksize, groups=in_channels, stride=1, padding=pad, bias=False)
        
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1,
            stride=1, groups=1, act=act
            )

    def forward(self, x):
        
        x = self.dconv(x)
        out = self.pconv(x)
        
        return out

class Panel(nn.Module):
    '''Panel Attention'''
    
    def __init__(self, in_channels, out_channels, stride=1, identity=False, shuffle=True, nl=False):
        
        super(Panel, self).__init__()
            
        self.out_channels = out_channels
        self.stride = stride
        
        self.identity = identity
        self.shuffle = shuffle
        self.nl = nl
        
        self.c_ffc = nn.Identity(1,1)
        if self.identity == False:                
            self.c_ffc = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, groups=1, bias=False)
            
        if self.stride == 2:            
            self.sc_x_downsample = BaseConv(int(out_channels), int(out_channels), ksize=2, stride=2, act='prelu')
            
        if self.shuffle == True:
            
            if self.stride == 2:
                self.s_ffc = nn.Sequential(
                    nn.Conv2d(int(out_channels*4), int(out_channels*4), kernel_size=3, padding=1, groups=int(out_channels*4), bias=False),
                    nn.Conv2d(int(out_channels*4), int(out_channels*1), kernel_size=1, padding=0, groups=1, bias=False),
                    nn.GroupNorm(1,int(out_channels*1)),
                    nn.BatchNorm2d(int(out_channels*1)),                    
                    nn.PReLU(init=0.5),
                    )
            
            if self.stride != 2:           
                self.s_ffc = nn.Sequential(
                    nn.Conv2d(int(out_channels*4), int(out_channels*4), kernel_size=3, padding=1, groups=int(out_channels*4), bias=False),
                    nn.Conv2d(int(out_channels*4), int(out_channels*4), kernel_size=1, padding=0, groups=1, bias=False),
                    nn.GroupNorm(1,int(out_channels*4)),
                    nn.BatchNorm2d(int(out_channels*4)),
                    nn.PReLU(init=0.5),
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
                        nn.PReLU(init=0.5),
                        )    
                if self.stride != 2:    
                    self.act = nn.Sequential(
                        nn.GroupNorm(1,int(out_channels*4)),
                        nn.BatchNorm2d(int(out_channels*4)),
                        nn.PReLU(init=0.5),
                        )    
        
        else:
            if self.stride == 2:
                self.s_ffc = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.BatchNorm2d(int(out_channels*1)),                    
                    nn.PReLU(init=0.5),
                    )
            if self.stride != 2:
                self.s_ffc = nn.Sequential(
                    nn.BatchNorm2d(int(out_channels*1)),                    
                    nn.PReLU(init=0.5),
                    )

            
    def forward(self, x, sc_x):    
       
        '''
        Identity/Linear transition:
            if True:
                out = identity(x)
            if False (linear transition): 
                out = CPA(x), CPA = channel pixel attention
        '''
        _, c, w, h = x.shape
        c_out = self.c_ffc(x)
        
        '''Pixel-unshuffle'''
        if self.shuffle == True:
            # it is for wanted downsmaling image size isn't shufflable, so create
            # a sufflable one by adding psudo pixels with exisiting value. A restore
            # operation will be conducted later.
            if w % 2:
                c_out = F.interpolate(c_out, int(w+1), mode='nearest')
        
            s_out = F.pixel_unshuffle(c_out, 2)
            
            '''
            Local attention:
                s_ffc = CNN 3x3 do local aggregation                
            '''
            s_out = self.s_ffc(s_out)
        
            '''Global attention'''
            if self.nl == True:                
                b, s_c, s_w, s_h = s_out.shape
                
                # panel grouped by picking in channel every 4 steps
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
                
                # nl attention
                s_qk = torch.matmul(s_q, s_k)
                s_qk = F.softmax(s_qk, -1)                
                s_qkv = torch.matmul(s_qk, s_v).permute(0,2,1).contiguous()
                s_qkv = s_qkv.reshape(b,-1,s_w,s_h)
                
                if self.stride == 2:
                    s_out = self.s_ffcx(s_x + s_qkv)    
                    
                if self.stride != 2:
                    '''
                    if want to maintain the same shape when output, replace xqkv
                    with out
                    '''
                    s_out = self.s_ffcx(s_x + s_qkv)
                    s_x = s_out
                    s_q = s_out
                    s_k = s_out
                    s_v = s_out
                    s_out = torch.cat([s_x, s_q, s_k, s_v], 1)
                    
                s_out = self.act(s_out)
        
            if self.stride != 2:
                s_out = F.pixel_shuffle(s_out, 2)
                
                # restore wanted down-sampling size
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

        return out

class upa_block(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, w=2, cat=False, shuffle=True, nl=False, name=None):
        
        super(upa_block, self).__init__()
        
        self.cat = cat
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.ap = nn.AdaptiveAvgPool1d(out_channels)

        if stride == 2:
            w = 1       
            self.att = Panel(int(in_channels), int(out_channels), stride, shuffle=shuffle, nl=nl)
            
            self.cnn = nn.Sequential(
                    BaseConv(int(in_channels * 1), int(out_channels * w), 3, stride=1, act='prelu'),
                    BaseConv(int(out_channels * w), int(out_channels * 1), 3, act='prelu'),
                    )
            
        if stride != 2:
            w = 2
            self.att = Panel(int(in_channels), int(out_channels), stride, shuffle=shuffle, nl=False)
        
            self.cnn = nn.Sequential(
                    BaseConv(int(in_channels * 1), int(out_channels * w), 3, stride=1, act='prelu'),
                    BaseConv(int(out_channels * w), int(out_channels * 1), 3, act='prelu'),
                    )
           
    def forward(self, x):    
        
        b, c, w, h = x.shape
        out = self.cnn(x)           
        out = self.att(x, out)
        
        if self.cat == True:
            out = torch.cat([x, out], 1)
        
        return out    

class upanets(nn.Module):
    def __init__(self, block, num_blocks, filter_nums):
        
        super(upanets, self).__init__()
        
        self.in_channels = filter_nums
        self.filters = filter_nums
        
        self.root = block(int(3), self.in_channels, stride=2)
        
        self.layer1 = self._make_layer(block, int(self.filters*1), num_blocks[0], stride=2, nl=False, name='layer1')
        self.layer2 = self._make_layer(block, int(self.filters*2), num_blocks[1], stride=2, nl=True, name='layer2')
        self.layer3 = self._make_layer(block, int(self.filters*4), num_blocks[2], stride=2, nl=True, name='layer3')
        self.layer4 = self._make_layer(block, int(self.filters*8), num_blocks[3], stride=2, nl=True, name='layer4')
        
    def _make_layer(self, block, channels, num_blocks, stride, nl=False, name=None):
        
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        inter_channels = channels // num_blocks
        
        for i, stride in enumerate(strides):
            
            if i == 0 and stride == 1:
                layers.append(block(channels, channels, stride, name=name))
                strides.append(1)
                self.in_channels = channels
                
            elif i != 0 and stride == 1:                
                layers.append(block(self.in_channels, inter_channels, stride, cat=True, name=name))                
                self.in_channels = self.in_channels + inter_channels 
                # self.in_channels = inter_channels 
                    
            else:   
                layers.append(block(self.in_channels, channels, stride, nl=nl, name=name))
                # layers.append(block(channels, channels, stride, nl=nl, name=name))
                
                strides.append(1)
                self.in_channels = channels

        return nn.Sequential(*layers)

    def forward(self, x):
        
        out0 = self.root(x) 

        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        
        return (out1, out2, out3, out4)


def UPANets(f, block = 1):
    
    return upanets(upa_block, [int(4*block), int(4*block), int(4*block), int(4*block)], f)

def test():

    net = UPANets(80, 1)
    y = net(torch.randn(1, 3, 544, 544))
    
    return y