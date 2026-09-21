import torch
from torch import nn

# local stuff
from .util import load_model
from .util import PreprocessModule
from torch.cuda.amp import autocast

class InceptionModule_( nn.Module ):
    def __init__(self, in_chan, layers, out_chan, 
                 norm_fn = 'batch', instance_norm=None, 
                 leaky=0.2, groups=8, rectify=True):
        super(InceptionModule_, self).__init__()
        
        if  norm_fn == 'instance' or instance_norm == True:
            _norm_fn = nn.InstanceNorm3d
        elif norm_fn == 'layer' :
            _norm_fn = nn.LayerNorm
        else:
            _norm_fn = nn.BatchNorm3d

        self.mods      = nn.ModuleList()
        out_module_channels = 0
        
        for i in layers:
            (kernel, m_out_chan)=i
            #_pad=int((kernel-1)/2)
            
            out_module_channels += m_out_chan
            self.mods.append(nn.Sequential(
                nn.Conv3d(
                    in_chan, m_out_chan, kernel, 
                    padding='same',
                    padding_mode='replicate' )
                ))

        if out_module_channels == out_chan and len(self.mods) == 1:
            self.merge = None
        else:
            merge_layers = []
            if rectify:
                if norm_fn == 'group' and (out_module_channels % groups) == 0:
                    merge_layers.append(nn.GroupNorm(groups, out_module_channels, affine=True))
                else:
                    merge_layers.append(_norm_fn(out_module_channels, affine=True))
                
                if leaky>0:
                    merge_layers.append(nn.LeakyReLU(leaky, inplace=True))
                else:
                    merge_layers.append(nn.ReLU(inplace=True))
            merge_layers.append(nn.Conv3d(out_module_channels, out_chan, 1))
            self.merge = nn.Sequential(*merge_layers)


    @autocast()
    def forward(self, x):
        v = []
        for i in self.mods:
            v.append( i.forward(x) ) 

        x = torch.cat(v, 1) # merge feature dimension
        
        if self.merge is not None:
            x = self.merge.forward(x)
        
        return x


class ResidualModule( nn.Module ):
    def __init__(self, in_chan, layers, out_chan, 
                 norm_fn = 'batch', instance_norm=None, 
                 leaky=0.2, groups=8, rectify=True):
        super(ResidualModule, self).__init__()
        
        if  norm_fn == 'instance' or instance_norm == True:
            _norm_fn = nn.InstanceNorm3d
        elif norm_fn == 'layer' :
            _norm_fn = nn.LayerNorm
        else:
            _norm_fn = nn.BatchNorm3d

        out_module_channels = 0

        if in_chan!=out_chan:
            self.expand = nn.Conv3d(in_chan, out_chan, 1)
        else:
            self.expand = None

        _blocks=[]
        for i in layers:
            (stages,kernel) = i
            _res_block=[]
            for _ in range(stages):
                if norm_fn == 'group' and (out_chan % groups) == 0:
                    _res_block.append(nn.GroupNorm(groups, out_chan, affine=True))
                else:
                    _res_block.append(_norm_fn(out_chan, affine=True))
                
                if leaky>0:
                    _res_block.append(nn.LeakyReLU(leaky, inplace=True))
                else:
                    _res_block.append(nn.ReLU(inplace=True))
                
                _res_block.append(
                    nn.Conv3d(
                        out_chan, out_chan, kernel, 
                        padding = 'same',
                        padding_mode = 'replicate' )                
                    )
            _blocks.append(nn.Sequential(*_res_block))

        self.blocks=nn.ModuleList(_blocks)

        if rectify:
            if leaky>0:
                self.rectify=nn.LeakyReLU(leaky, inplace=True)
            else:
                self.rectify=nn.ReLU(inplace=True)
        else:
            self.rectify=None

    @autocast()
    def forward(self, x):
        if self.expand is not None:
            x = self.expand.forward(x)
        
        for m in self.blocks:
            identity = x
            x = m.forward(x)
            x += identity

        if self.rectify is not None:
            x = self.rectify.forward(x)
        return x
