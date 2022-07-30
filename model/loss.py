import torch
import torch.nn as nn
import torch.nn.functional as torch_F
import utils.util_cam as util_cam

class Loss(nn.Module):

    def __init__(self, impl_func):
        super().__init__()
        self.impl_func = impl_func

    def L1_loss(self,pred,label=0,weight=None,mask=None):
        loss = (pred.contiguous()-label).abs()
        return self.aggregate_loss(loss,weight=weight,mask=mask)

    def MSE_loss(self,pred,label=0,weight=None,mask=None,tolerance=0.):
        batch_size = pred.shape[0]
        loss = (pred.contiguous()-label)**2
        if tolerance > 1.e-5:
            assert len(pred.shape) == 4 and pred.shape[1] == 3
            # [B, HW]
            loss_pixel = loss.mean(dim=1).view(batch_size, -1)
            loss_sorted = torch.sort(loss_pixel, dim=-1, descending=False)[0]
            end_idx = int((1-tolerance) * loss_pixel.shape[1])
            loss_valid = loss_sorted[:, :end_idx].contiguous()
            return self.aggregate_loss(loss_valid,weight=weight,mask=mask)
        return self.aggregate_loss(loss,weight=weight,mask=mask)

    def CE_loss(self,pred,label,weight=None,mask=None):
        loss = torch_F.cross_entropy(pred,label,reduction="none")
        return self.aggregate_loss(loss,weight=weight,mask=mask)

    def BCE_loss(self,pred,label,weight=None,mask=None,tolerance=0.):
        batch_size = pred.shape[0]
        label = label.expand_as(pred)
        loss = torch_F.binary_cross_entropy(pred,label,reduction="none")
        if tolerance > 1.e-5:
            assert len(pred.shape) == 4 and pred.shape[1] == 1
            # [B, HW]
            loss_pixel = loss.view(batch_size, -1)
            loss_sorted = torch.sort(loss_pixel, dim=-1, descending=False)[0]
            end_idx = int((1-tolerance) * loss_pixel.shape[1])
            loss_valid = loss_sorted[:, :end_idx].contiguous()
            return self.aggregate_loss(loss_valid,weight=weight,mask=mask)
        return self.aggregate_loss(loss,weight=weight,mask=mask)

    def ray_intersection_loss(self,opt,var,level_eps=0.01):
        level_in = var.level_all[...,-1:] # [B,HW,1]
        weight = 1/(var.dt_input+1e-8) if opt.impl.importance else None
        loss = self.L1_loss((level_in+level_eps).relu_(),weight=weight,mask=var.mask_input.bool()) \
                +self.L1_loss((-level_in+level_eps).relu_(),weight=weight,mask=~var.mask_input.bool())
        return loss

    def ray_freespace_loss(self,opt,var,level_eps=0.01):
        level_out = var.level_all[...,:-1] # [B,HW,N-1]
        loss = self.L1_loss((-level_out+level_eps).relu_())
        return loss

    def shape_from_silhouette_loss(self,opt,var): # [B,N,H,W]
        batch_size = len(var.idx)
        mask_bg = var.mask_input.long()==0
        weight = 1/(var.dt_input+1e-8) if opt.impl.importance else None
        # randomly sample depth along ray
        depth_min,depth_max = opt.impl.depth_range
        num_rays = var.ray_idx.shape[1] if "ray_idx" in var else opt.H*opt.W
        # [B,HW,N,1]
        depth_samples = torch.rand(batch_size,num_rays,opt.impl.sdf_samples,1,device=var.rgb_input.device)*(depth_max-depth_min)+depth_min 
        center,ray = util_cam.get_center_and_ray(opt,var.pose,intr=var.intr,device=var.rgb_input.device)
        if "ray_idx" in var:
            # var.ray_idx: [B, 1024]
            # gather_idx: [B, 1024, 3]
            gather_idx = var.ray_idx[...,None].repeat(1,1,3)
            # ray: [B, HW, 3]
            ray = ray.gather(dim=1,index=gather_idx)
            # ray: [B, 1024, 3]
            if opt.camera.model=="orthographic":
                center = center.gather(dim=1,index=gather_idx)
        points_3D_samples = util_cam.get_3D_points_from_depth(opt,center,ray,depth_samples,multi_samples=True) # [B,HW,N,3]
        level_samples = self.impl_func.forward(opt,points_3D_samples,var.impl_weights)[...,0] # [B,HW,N]
        # compute lower bound following eq 1 when using SDF representation
        if opt.camera.model=="perspective":
            # grid_3D is u
            _,grid_3D = util_cam.get_camera_grid(opt,batch_size,device=var.rgb_input.device,intr=var.intr) # [B,HW,3]
            # u / ||u|| * D(u)
            offset = torch_F.normalize(grid_3D[...,:2],dim=-1)*var.dt_input_map.view(batch_size,-1,1) # [B,HW,2]
            # v = u + u / ||u|| * D(u)
            _,ray0 = util_cam.get_center_and_ray(opt,var.pose,intr=var.intr,device=var.rgb_input.device,offset=offset) # [B,HW,3]
            if "ray_idx" in var:
                gather_idx = var.ray_idx[...,None].repeat(1,1,3)
                ray0 = ray0.gather(dim=1,index=gather_idx)
            # eq 1
            ortho_dist = (ray-ray0*(ray*ray0).sum(dim=-1,keepdim=True)/(ray0*ray0).sum(dim=-1,keepdim=True)).norm(dim=-1,keepdim=True) # [B,HW,1]
            min_dist = depth_samples[...,0]*ortho_dist # [B,HW,N]
        elif opt.camera.model=="orthographic":
            min_dist = var.dt_input
        loss = self.L1_loss((min_dist-level_samples).relu_(),weight=weight,mask=mask_bg)
        return loss

    def iou_loss(self,inputs,targets):
        batch_size = inputs.shape[0]
        inputs_expand = inputs.view(batch_size, -1)
        targets_expand = targets.view(batch_size, -1)
        loss = 1 - (inputs_expand * targets_expand).sum(dim=1) / (inputs_expand + targets_expand - inputs_expand * targets_expand + 1.e-8).sum(dim=1)
        loss = loss.mean()
        return loss

    def view_consistency_loss(self,opt,var,estimator):
        img_recon = torch.cat([var.rgb_recon_map.detach().clone(), var.mask_map], dim=1)
        img_rnd = torch.cat([var.rgb_rnd_view.detach().clone()*var.mask_rnd_view + opt.data.bgcolor*(1-var.mask_rnd_view), var.mask_rnd_view], dim=1)
        trig_azim_rendered = estimator(img_recon)     # [B,2]
        trig_azim_rnd_view = estimator(img_rnd)      # [B,2]
        loss = torch.mean(1 - trig_azim_rendered * var.trig_azim - trig_azim_rnd_view * var.trig_azim_rnd)
        return loss

    def category_metric_loss(self,opt,var,center):
        shape_code_normed = torch_F.normalize(var.latent_sdf, dim=-1)
        shape_center_normed = torch_F.normalize(center, dim=-1)
        logits = shape_code_normed @ shape_center_normed.permute(1,0)
        loss = self.CE_loss(logits/opt.metric_temp, var.category_label)
        return loss

    def aggregate_loss(self,loss,weight=None,mask=None):
        if mask is not None:
            if mask.sum()==0: return 0.*loss.mean() # keep loss dependent on prediction
            mask = mask.expand_as(loss)
            if weight is not None: weight = weight.expand_as(loss)
            loss = loss[mask]
        if weight is not None:
            if mask is not None: weight = weight[mask]
            loss = (loss*weight).sum()/weight.sum()
        else: loss = loss.mean()
        return loss