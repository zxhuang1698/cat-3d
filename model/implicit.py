import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_F

class ImplicitFunc(nn.Module):

    def __init__(self,opt):
        super().__init__()
        self.opt = opt

    def forward(self,opt,points_3D,impl_weights,extra_return=None):
        impl_weights_level = impl_weights.level
        impl_weights_rgb = impl_weights.rgb
        points_level = points_3D
        points_rgb = points_3D
        if opt.impl.posenc_level:
            # positional encoding of sdf
            points_level = self.positional_encoding(opt,points_3D,opt.impl.posenc_level)
            points_level = torch.cat([points_level,points_3D],dim=-1)
        if opt.impl.posenc_rgb:
            # positional encoding of texture
            points_rgb = self.positional_encoding(opt,points_3D,opt.impl.posenc_rgb)
            points_rgb = torch.cat([points_rgb,points_3D],dim=-1)
        
        feat_render = points_level
        # extract features for renderer LSTM
        for li,weight in enumerate(impl_weights_level[:-1-opt.impl.render_layer]):
            if opt.arch.interm_coord and li>0:
                feat_render = torch.cat([feat_render,points_level],dim=-1)
            feat_render = torch_F.relu(self.BatchLinearFunc(feat_render, weight[0], bias=weight[1]), inplace=False)
        
        # extract sdf
        feat_level = feat_render.clone()
        for weight in impl_weights_level[-1-opt.impl.render_layer:-1]:
            if opt.arch.interm_coord:
                feat_level = torch.cat([feat_level,points_level],dim=-1)
            feat_level = torch_F.relu(self.BatchLinearFunc(feat_level, weight[0], bias=weight[1]), inplace=False)
        level = self.BatchLinearFunc(feat_level, impl_weights_level[-1][0], bias=impl_weights_level[-1][1])

        # extract rgb
        if extra_return == 'rgb':
            feat_rgb = [points_rgb]
            if opt.arch.shape_condition:
                feat_rgb.append(feat_render)
            feat_rgb = torch.cat(feat_rgb, dim=-1)
            for li,weight in enumerate(impl_weights_rgb[:-1]):
                if opt.arch.interm_coord and li>0:
                    feat_rgb = torch.cat([feat_rgb,points_rgb],dim=-1)
                feat_rgb = torch_F.relu(self.BatchLinearFunc(feat_rgb, weight[0], bias=weight[1]), inplace=False)
            rgb = self.BatchLinearFunc(feat_rgb, impl_weights_rgb[-1][0], bias=impl_weights_rgb[-1][1])

        # prepare return
        if extra_return is None: return level
        elif extra_return == 'render': return level, feat_render
        elif extra_return == 'rgb': return level, rgb

    def positional_encoding(self,opt,points_3D,length): # [B,...,3]
        shape = points_3D.shape
        points_enc = []
        if length:
            freq = 2**torch.arange(length,dtype=torch.float32,device=points_3D.device)*np.pi # [L]
            spectrum = points_3D[...,None]*freq # [B,...,3,L]
            sin,cos = spectrum.sin(),spectrum.cos()
            points_enc_L = torch.cat([sin,cos],dim=-1).view(*shape[:-1],6*length) # [B,...,6L]
            points_enc.append(points_enc_L)
        points_enc = torch.cat(points_enc,dim=-1) # [B,...,X]
        return points_enc

    def BatchLinearFunc(self, x, weight, bias=None):
        batch_size = len(weight)
        assert(len(x)<=batch_size)
        shape = x.shape
        x = x.view(shape[0],-1,shape[-1]) # sample-wise vectorization
        y = x@weight[:len(x)]
        if bias is not None: y = y+bias[:len(x)]
        y = y.view(*shape[:-1],y.shape[-1]) # reshape back
        return y