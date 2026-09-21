import torch
from torch import nn

# local stuff
from .util import load_model,import_parameters
from .basic2 import InceptionModule_,ResidualModule

from torch.cuda.amp import autocast

class VAE_UNET2(nn.Module):
    """
    Unet  with VAE regularization
    """
    def __init__(self, layers_down, layers_up, layers_add, layers_vae, layers_add_vae,
                 layers_add_group = None,
                 in_chan=1, patch_sz=32, n_cls=2,
                 output_vae = 1, 
                 leaky=0.2, 
                 dropout=None, 
                 dropout_vae=None,
                 dropout_group=None,
                 upsample_mode='trilinear',
                 vae_dimensionality=256, 
                 vae_channels_out=4, 
                 group_norm_channels=8,
                 use_resnet=False,
                 inner_layers=None,
                 use_vae=True,
                 use_seg=True ):
        #
        super(VAE_UNET2, self).__init__()
        # make sure we have equal number of layers
        assert len(layers_down) == len(layers_up)
        assert len(layers_vae) == len(layers_down)

        self.use_vae   = use_vae
        self.output_latent = False
        if isinstance(patch_sz,list):
            self.patch_sz  = patch_sz
        else:
            self.patch_sz  = [patch_sz,patch_sz,patch_sz]
        self.n_cls     = n_cls
        self.vae_dimensionality = vae_dimensionality
        self.vae_channels_out = vae_channels_out

        downsample     = nn.AvgPool3d
        in_layer_chan  = in_chan

        self.mods_down  = nn.ModuleList()
        self.mods_up    = nn.ModuleList()
        self.upsample_mode = upsample_mode
        self.mods_ds    = nn.ModuleList()
        self.mods_vae   = nn.ModuleList()

        self.channels_down = []
        self.channels_up  = []
        self.channels_add = []
        self.channels_vae = []
        self.channels_add_vae = []
        self.channels_add_group = []

        self.skip_vae = False # flag to skip VAE in forward
        self.skip_seg = False # flag to skip segmentation in forward
        self.skip_group = False # flag to skip group prediction in forward

        self.levels      = len(layers_down)
        _size_reduction  = 1
        if use_resnet:
            print("Using ResNet modules")
        else:
            print("Using Inception modules")
        
        # downstream
        for i in layers_down:
            self.channels_down.append(in_layer_chan)
            (_layers, out_layer_chan) = i

            if use_resnet:
                self.mods_down.append(ResidualModule(in_layer_chan, _layers, out_layer_chan,
                                    leaky=leaky, groups=group_norm_channels, norm_fn='group' ))
            else:
                self.mods_down.append(InceptionModule_(in_layer_chan, _layers, out_layer_chan,
                                    leaky=leaky, groups=group_norm_channels, norm_fn='group' ))
            in_layer_chan = out_layer_chan
            self.mods_ds.append(downsample(2))
            _size_reduction *= 2
        
        # layers to be used at the lowest level
        if inner_layers is not None:
            (_layers, out_layer_chan ) = inner_layers
            if use_resnet:
                self.inner = ResidualModule(in_layer_chan, _layers, out_layer_chan,
                                    leaky=leaky, groups=group_norm_channels, norm_fn='group' )
            else:
                self.inner = InceptionModule_(in_layer_chan, _layers, out_layer_chan,
                                    leaky=leaky, groups=group_norm_channels, norm_fn='group' )
            in_layer_chan = out_layer_chan
        else:
            self.inner = None
        
        self.vae_channels_in = in_layer_chan
        self.group_channels_in = in_layer_chan

        if use_seg:
            # upstream
            for j,i in enumerate(layers_up):
                (_layers, out_layer_chan) = i

                if use_resnet:
                    self.mods_up.append(ResidualModule(in_layer_chan, _layers, out_layer_chan,
                                        leaky=leaky, groups=group_norm_channels, norm_fn='group' ))
                else:
                    self.mods_up.append(InceptionModule_(in_layer_chan, _layers, out_layer_chan,
                                        leaky=leaky, groups=group_norm_channels, norm_fn='group' ))

                # concat not downsampled channels here
                in_layer_chan = out_layer_chan + self.channels_down[-j-1]
                self.channels_up.append(in_layer_chan)

            # addon for main branch
            _add = []


            if dropout is not None and dropout>0.0:
                _add.append( nn.Dropout3d(dropout) )


            for i in layers_add:
                self.channels_add.append(in_layer_chan)
                (_layers, out_layer_chan) = i
                
                if use_resnet:
                    _add.append(ResidualModule(in_layer_chan, _layers, out_layer_chan,
                                        leaky=leaky, norm_fn='group',
                                        groups = group_norm_channels ))
                else:
                    _add.append( InceptionModule_(in_layer_chan, _layers, out_layer_chan,
                                leaky = leaky, norm_fn = 'group',
                                groups = group_norm_channels  ))
                in_layer_chan = out_layer_chan

            self.mods_add = nn.Sequential(*_add)
        else:
            print("Not using segmentation branch")
            self.mods_add = None

        # create VAE hidden state
        self.group_downsample=_size_reduction
        self.vae_patch_sz   = [patch_sz[i]//_size_reduction for i in range(3)]
        self.group_patch_sz = [patch_sz[i]//_size_reduction for i in range(3)]

        print("vae_patch_sz=",   self.vae_patch_sz)
        print("vae_channels_in=",self.vae_patch_sz[0] * self.vae_patch_sz[1] * self.vae_patch_sz[2] * self.vae_channels_in)

        if self.use_vae:
            # collapse into a hidden state
            self.vae_in = torch.nn.Linear(self.vae_patch_sz[0] * self.vae_patch_sz[1] * self.vae_patch_sz[2] * self.vae_channels_in, 
                                        self.vae_dimensionality)

            self.vae_mean = torch.nn.Linear(self.vae_dimensionality, self.vae_dimensionality//2)
            self.vae_std  = torch.nn.Linear(self.vae_dimensionality, self.vae_dimensionality//2)

            # expand from hidden state
            self.vae_out = torch.nn.Linear(self.vae_dimensionality//2, 
                                        self.vae_patch_sz[0] * self.vae_patch_sz[1] * self.vae_patch_sz[2] * self.vae_channels_out)

            in_layer_chan_vae = self.vae_channels_out
            print("in_layer_chan_vae=", in_layer_chan_vae)

            # make another upstream branch, this time it's VAE
            # VAE upstream
            for j,i in enumerate(layers_vae) :
                (_layers, out_layer_chan) = i
                if use_resnet:
                    self.mods_vae.append(ResidualModule(in_layer_chan_vae, _layers, out_layer_chan,
                                        leaky=leaky, norm_fn='group',
                                        groups = group_norm_channels ))
                else:
                    self.mods_vae.append(InceptionModule_(in_layer_chan_vae, _layers, out_layer_chan,
                                        leaky = leaky, norm_fn = 'group',
                                        groups = group_norm_channels ))
                in_layer_chan_vae = out_layer_chan
                self.channels_vae.append( in_layer_chan_vae )
            # addon for vae branch
            _add_vae = []
            if dropout_vae is not None and dropout_vae>0.0:
                _add_vae.append( nn.Dropout3d(dropout_vae) )

            for i in layers_add_vae:
                self.channels_add_vae.append(in_layer_chan_vae)
                (_layers, out_layer_chan) = i
                if use_resnet:
                    _add_vae.append(ResidualModule(in_layer_chan_vae, _layers, out_layer_chan,
                                        leaky=leaky, norm_fn='group',
                                        groups = group_norm_channels,
                                        rectify = False ))
                else:
                    _add_vae.append( InceptionModule_(in_layer_chan_vae, _layers, out_layer_chan,
                                    leaky = leaky, norm_fn = 'group',
                                    groups = group_norm_channels,
                                    rectify = False ))
                in_layer_chan_vae = out_layer_chan

            self.mods_add_vae = nn.Sequential(*_add_vae)
        else:
            print("Not using VAE branch")
            self.vae_in = None
            self.vae_mean = None
            self.vae_std  = None
            self.vae_out = None

        if layers_add_group is not None:
            # addon for group prediction
            _add_group = []
            if dropout_group is not None and dropout_group>0.0:
                _add_group.append( nn.Dropout3d( dropout_group ) )

            in_layer_chan_group = self.group_channels_in 
            # this is running at lowest resolution, without upsampling!
            for j,i in enumerate(layers_add_group):
                (_layers, out_layer_chan) = i
                _rectify = j != (len(layers_add_group)-1)
                if use_resnet:
                    _add_group.append(ResidualModule(in_layer_chan_group, _layers, out_layer_chan,
                                    leaky=leaky, norm_fn='group',
                                    groups = group_norm_channels,
                                    rectify = _rectify ))
                else:
                    _add_group.append( InceptionModule_(in_layer_chan_group,
                                    _layers, out_layer_chan,
                                    leaky = leaky, norm_fn = 'group',
                                    groups = group_norm_channels,
                                    rectify = _rectify ))
                in_layer_chan_group = out_layer_chan

            self.mods_add_group = nn.Sequential( *_add_group )
            # will produce a 3D patch at reduced resolution (self.group_patch_sz ^3)
        else:
            self.mods_add_group = None
    
    @autocast()
    def reparameterize(self, mu, logvar):
        """
        VAE reparametrization trick
        """
        std = torch.exp( 0.5*logvar )
        eps = torch.randn_like( std )

        return mu + eps*std

    @autocast()
    def forward(self, x):
        """
        Return primary output and vae output , and variational mean and logvar
        """
        layer_in = []
        output = {}

        #pass throgh the U-net down
        for i, j in enumerate(self.mods_down):
            layer_in.append(x)
            # pass the filter
            x = j(x)
            # pass downsampler
            x = self.mods_ds[i](x)
        #####
        # inner layers
        if self.inner is not None:
            x = self.inner(x)
        ###
        v = x
        _g = x

        if self.output_latent:
            # propagate latent vector outside , with gradient info
            output['latent'] = x 

        if not self.skip_seg:
            #####
            # pass through the U-net up,
            # concatenating skip connections
            for i, j in enumerate(self.mods_up):
                x = j(x)
                # pass through the module
                x = nn.functional.interpolate(x, scale_factor=2, mode=self.upsample_mode, align_corners=False)
                # concatenate with input
                x = torch.cat([x, layer_in[-i-1]], 1)
            #pass throgh addon stages
            output['seg'] = self.mods_add(x)

        # pass through group prediction channel
        if self.mods_add_group is not None and not self.skip_group:
            output['group'] = self.mods_add_group(_g)

        # skip VAE branch , if not needed, mostly to speedup inference
        if not self.skip_vae : # and self.use_vae
            # pass through the VAE up
            # concatenating skip connections
            v = v.view(v.size(0),-1)
            # DEBUG
            v = self.vae_in(v)
            v_mean = self.vae_mean(v)
            v_std = self.vae_std(v)

            # reparametrize
            v = self.reparameterize(v_mean,v_std)

            # produce enough channels to populate patch
            v = self.vae_out(v)

            # reshape
            v = v.reshape(-1, self.vae_channels_out, self.vae_patch_sz[0], self.vae_patch_sz[1], self.vae_patch_sz[2])

            for i, j in enumerate(self.mods_vae):
                v = j(v)
                v = nn.functional.interpolate(v, scale_factor=2, mode=self.upsample_mode,align_corners=False)

            output['vae'] = self.mods_add_vae(v)
            output['vae_mean'] = v_mean
            output['vae_logvar'] = v_std
        
        return output

    def enable_vae(self, flag=True):
        self.skip_vae = not flag
        
    def disable_vae(self, flag=True):
        self.skip_vae = flag

    def enable_seg(self, flag=True):
        self.skip_seg = not flag

    def enable_group(self, flag=True):
        self.skip_group = not flag
    
    def enable_latent(self,flag=True):
        """
        makes forward output "latent" tensor, with shape KxD where K is number of voxels in innermost patch and D number of channels
        """
        self.output_latent = flag
        

def init_model(desc, patch_sz=32, cpu=False):
    """
    Initialize a single model based on description
    :param desc: a dict
    :param n_cls: number of classes
    :return: a model
    """
    _version = desc.get('version', None)
    _type = desc.get('type', 'vae')

    _leaky = desc.get('leaky', 0.1)
    _dropout = desc.get('dropout', None)
    _dropout_vae = desc.get('dropout_vae', None)

    _upsample_mode = desc.get('upsample_mode','nearest')

    _input_n    = desc.get('input')
    _output_n   = desc.get('output')
    _output_vae = desc.get('output_vae', _input_n)
    _vae_dimensionality = desc.get('vae_dimensionality', 256)
    _vae_channels_out = desc.get('vae_channels_out', 256)
    _use_resnet = desc.get('use_resnet',False)
    _use_vae = desc.get('use_vae', True)
    _use_seg = desc.get('use_seg', True)

    print(repr(desc))

    model = VAE_UNET2(desc.get('downstream',
                      [
                        [[[5, 8], [3, 8], [1, 8]], 12],  # downstream
                        [[[5, 8], [3, 8], [1, 8]], 24],
                        [[[5, 8], [3, 8], [1, 8]], 24],
                        [[[1, 32]], 32],
                      ]),
                     desc.get('upstream',
                      [
                         [[[5, 16], [3, 16], [1, 16]], 48],  # upstream
                         [[[3, 16], [1, 16]         ], 32],
                         [[[3, 16], [1, 16]         ], 32],
                         [[[1, 32]                  ], 32],
                      ]),
                     desc.get('addon',[[[[3, 4], [1, 4]], 48],  # addon
                                       [[[1, _output_n]], _output_n]]),
                     desc.get('upstream_vae',[
                         [[[5, 16], [3, 16], [1, 16]], 48],  # upstream VAE
                         [[[3, 16], [1, 16]], 32],
                         [[[3, 16], [1, 16]], 32],
                         [[[1, 32]], 32],
                      ]),
                     desc.get('addon_vae',[[[[3, 4], [1, 4]], 48],  # addon for VAE
                                          [[[1, 1]], 1]]),
                     layers_add_group = desc.get('addon_group', None),
                     leaky = _leaky,
                     dropout = _dropout,
                     dropout_vae = _dropout_vae,
                     dropout_group = desc.get('dropout_group', None),
                     in_chan = _input_n,
                     output_vae = _output_vae,
                     upsample_mode =_upsample_mode,
                     vae_dimensionality = _vae_dimensionality,
                     vae_channels_out =_vae_channels_out,
                     patch_sz = patch_sz,
                     use_resnet = _use_resnet,
                     inner_layers=desc.get('inner',None),
                     use_vae=_use_vae,
                     use_seg=_use_seg)
                     
    if not cpu:
        model.cuda()
    if 'load' in desc:
        load_model(model, desc['load'], map_location=next(model.parameters()).device)
    if 'import' in desc:
        import_parameters(model, desc['import'], map_location=next(model.parameters()).device)
    return model


class Prefilter(nn.Module):
    """
    Prefiltering module, supposed to be pre-trained already
    """
    def __init__(self, desc, patch_sz=32, cpu=False, parallel=False):
        """
        Initialize pre-trained model that will serve as additional input filter
        """
        super(Prefilter, self).__init__()
        # TODO: parallel ?
        import yaml

        self.downsample = desc.get('downsample', 1) # no downsample ?
        self.upsample_mode = desc.get('upsample_mode', 'trilinear')
        self.output = desc.get('output', 2) # output channels
        self.inner_patch_sz = desc.get('patch_sz', 32)
        
        print("Prefilter: downsample:{} upsample_mode:{} patch_sz:{}".format(self.downsample,self.upsample_mode,self.inner_patch_sz))

        self.inner_model = init_model( yaml.load(open(desc['model_description'], 'r'))[desc['model_key']], 
                            parallel=parallel, patch_sz=self.inner_patch_sz, cpu=cpu )

        load_model(self.inner_model, desc['load'], cpu=cpu)

        self.inner_model.enable_vae(False)

        if cpu:
            self.inner_model.cpu()
        else:
            self.inner_model.cuda()
        self.inner_model.eval()

        print("Loaded model:",self.inner_model )

    @autocast()
    def forward(self, x):
        _x = x

        if self.downsample>1:
            x = nn.functional.avg_pool3d(x, self.downsample)
        # pass through inner model
        x = self.inner_model.forward(x)
        # TODO: add softmax here?
        if self.downsample>1:
            x = nn.functional.interpolate(x, scale_factor=self.downsample, mode=self.upsample_mode,align_corners=False)

        # concatenate to the input
        return torch.cat([_x, x], 1)
