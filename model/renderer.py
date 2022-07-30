import torch
import torch.nn as nn
import utils.util_cam as util_cam

from model.implicit import ImplicitFunc

class Renderer(nn.Module):

    def __init__(self,opt):
        super().__init__()
        self.define_ray_LSTM(opt)
        self.impl_func = ImplicitFunc(opt)

    def define_ray_LSTM(self,opt):
        # feature before rgb and sdf head
        feat_dim = opt.arch.layers_level[-2-opt.impl.render_layer]
        # LSTM hidden_size=opt.arch.lstm_dim
        self.ray_lstm = nn.LSTMCell(input_size=feat_dim,hidden_size=opt.arch.lstm_dim)
        self.lstm_pred = nn.Linear(opt.arch.lstm_dim,1)
        # initialize LSTM
        for name,param in self.ray_lstm.named_parameters():
            if not "bias" in name: continue
            n = param.shape[0]
            param.data[n//4:n//2].fill_(1.)

    def forward(self,opt,impl_weights,pose,intr=None):
        batch_size = len(pose)
        # in world frame, intrinsics are for perspective camera
        center,ray = util_cam.get_center_and_ray(opt,pose,intr=intr,device=pose.device) # [B,HW,3]
        num_rays = opt.H*opt.W
        depth = torch.empty(batch_size,num_rays,1,device=pose.device).fill_(opt.impl.init_depth) # [B,HW,1]
        level_all = []
        state = None
        for _ in range(opt.impl.srn_steps):
            points_3D = util_cam.get_3D_points_from_depth(opt,center,ray,depth) # [B,HW,3]
            level,feat = self.impl_func.forward(opt,points_3D,impl_weights,extra_return='render') # [B,HW,K]
            level_all.append(level)
            state = self.ray_lstm(feat.view(batch_size*num_rays,-1),state)
            delta = self.lstm_pred(state[0]).view(batch_size,num_rays,1).abs_() # [B,HW,1]
            depth = depth+delta
        # final endpoint
        points_3D_2ndlast,level_2ndlast = points_3D,level
        points_3D = util_cam.get_3D_points_from_depth(opt,center,ray,depth) # [B,HW,3]
        level = self.impl_func.forward(opt,points_3D,impl_weights) # [B,HW,1]
        level_all.append(level)
        level_all = torch.cat(level_all,dim=-1)
        mask = (torch.min(level_all, dim=-1, keepdim=True)[0]<=0).float()
        # get the intersection point if the last segment includes it
        func = lambda x: self.impl_func.forward(opt,x,impl_weights)
        points_3D_iso0 = self.bisection(x0=points_3D_2ndlast,x1=points_3D,y0=level_2ndlast,y1=level,
                                        func=func,num_iter=opt.impl.bisection_steps) # [B,HW,3]
        level,rgb = self.impl_func.forward(opt,points_3D_iso0,impl_weights,extra_return='rgb') # [B,HW,K]
        depth = util_cam.get_depth_from_3D_points(opt,center,ray,points_3D_iso0) # [B,HW,1]
        rgb = rgb.tanh_() # [B,HW,3]
        soft_mask = torch.sigmoid(-opt.impl.rgb_temp*torch.min(level_all, dim=-1, keepdim=True)[0]) 
        return rgb,depth,level,mask,level_all,soft_mask # [B,HW,K]
    
    def bisection(self,x0,x1,y0,y1,func,num_iter):
        for s in range(num_iter):
            x2 = (x0+x1)/2
            y2 = func(x2)
            side = ((y0<0)^(y2>0)).float() # update x0 if side else update x1
            x0,x1 = x2*side+x0*(1-side),x1*side+x2*(1-side)
            y0,y1 = y2*side+y0*(1-side),y1*side+y2*(1-side)
        x2 = (x0+x1)/2
        return x2
