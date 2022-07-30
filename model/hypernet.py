import torch.nn as nn
import utils.util_run as util_run

from utils.util_run import EasyDict as edict

class HyperNet(nn.Module):

    def __init__(self,opt):
        super().__init__()
        self.define_network(opt)

    def define_network(self,opt):
        # 3 + 6 * L
        point_dim_level = 3+(opt.impl.posenc_level*6 if opt.impl.posenc_level else 0)
        point_dim_rgb = 3+(opt.impl.posenc_rgb*6 if opt.impl.posenc_rgb else 0)\
            +(opt.arch.layers_level[-2-opt.impl.render_layer] if opt.arch.shape_condition else 0)
        # each is a module list, every element is a modulelist to generate params of specific layer
        self.hyper_level = self.get_module_params(opt,opt.arch.layers_level,k0=point_dim_level,branch='sdf',interm_coord=opt.arch.interm_coord)
        self.hyper_rgb = self.get_module_params(opt,opt.arch.layers_rgb,k0=point_dim_rgb,branch='rgb',interm_coord=opt.arch.interm_coord)

    def get_module_params(self,opt,layers,k0,branch,interm_coord=False):
        impl_params = nn.ModuleList()
        L = util_run.get_layer_dims(layers)
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = k0
            if interm_coord and li>0 and li<len(L)-1:
                k_in += k0
            params = self.define_hyperlayer(opt,dim_in=k_in,dim_out=k_out,branch=branch)
            impl_params.append(params)
        return impl_params

    def define_hyperlayer(self,opt,dim_in,dim_out,branch):
        # return hyperlayer to generate the parameter of the implicit layers: 
        L = util_run.get_layer_dims(opt.arch.layers_hyper)
        hyperlayer = nn.ModuleList()
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = opt.arch["latent_dim_"+branch]
            if li==len(L)-1: k_out = (dim_in+1)*dim_out # weight and bias
            hyperlayer.append(nn.Linear(k_in,k_out))
            if li!=len(L)-1:
                hyperlayer.append(nn.ReLU(inplace=False))
        return hyperlayer

    def forward(self,opt,latent_sdf,latent_rgb):
        point_dim_level = 3+(opt.impl.posenc_level*6 if opt.impl.posenc_level else 0)
        point_dim_rgb = 3+(opt.impl.posenc_rgb*6 if opt.impl.posenc_rgb else 0)+\
            (opt.arch.layers_level[-2-opt.impl.render_layer] if opt.arch.shape_condition else 0)
        impl_weights = edict()
        # each one is a list of BatchLinear layer
        impl_weights.level = self.hyperlayer_forward(opt,latent_sdf,self.hyper_level,opt.arch.layers_level,k0=point_dim_level,interm_coord=opt.arch.interm_coord)
        impl_weights.rgb = self.hyperlayer_forward(opt,latent_rgb,self.hyper_rgb,opt.arch.layers_rgb,k0=point_dim_rgb,interm_coord=opt.arch.interm_coord)
        return impl_weights

    def hyperlayer_forward(self,opt,latent,module,layers,k0,interm_coord=False):
        batch_size = len(latent)
        impl_layers = []
        L = util_run.get_layer_dims(layers)
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = k0
            if interm_coord and li>0 and li<len(L)-1: k_in += k0
            hyperlayer = module[li]
            # get the params from the hyper layers and latent code
            feat = latent.clone()
            for layer in hyperlayer:
                feat = layer(feat) 
            out = feat.view(batch_size,k_in+1,k_out)
            # use the params to build layers
            impl_layers.append([out[:,1:],out[:,:1]])
        return impl_layers
