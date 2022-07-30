import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as torchvision_F
import matplotlib.pyplot as plt
import PIL
import PIL.ImageDraw
from PIL import Image
import trimesh

@torch.no_grad()
def tb_image(opt,tb,step,group,name,images,masks=None,num_vis=None,from_range=(0,1),poses=None,cmap="gray"):
    images = preprocess_vis_image(opt,images,masks=masks,from_range=from_range,cmap=cmap)
    num_H,num_W = num_vis or opt.tb.num_images
    images = images[:num_H*num_W]
    if poses is not None:
        # poses: [B, 3, 4]
        # rots: [max(B, num_images), 3, 3]
        rots = poses[:num_H*num_W,...,:3]
        images = torch.stack([draw_pose(opt,image,rot,size=20,width=2) for image,rot in zip(images,rots)],dim=0)
    image_grid = torchvision.utils.make_grid(images[:,:3],nrow=num_W,pad_value=1.)
    if images.shape[1]==4:
        mask_grid = torchvision.utils.make_grid(images[:,3:],nrow=num_W,pad_value=1.)[:1]
        image_grid = torch.cat([image_grid,mask_grid],dim=0)
    tag = "{0}/{1}".format(group,name)
    tb.add_image(tag,image_grid,step)

def preprocess_vis_image(opt,images,masks=None,from_range=(0,1),cmap="gray"):
    min,max = from_range
    images = (images-min)/(max-min)
    if masks is not None:
        # then the mask is directly the transparency channel of png
        images = torch.cat([images,masks],dim=1)
    images = images.clamp(min=0,max=1).cpu()
    if images.shape[1]==1:
        images = get_heatmap(opt,images[:,0].cpu(),cmap=cmap)
    return images

def dump_images(opt,idx,name,images,masks=None,from_range=(0,1),poses=None,cmap="gray",folder='dump'):
    images = preprocess_vis_image(opt,images,masks=masks,from_range=from_range,cmap=cmap) # [B,3,H,W]
    if poses is not None:
        rots = poses[...,:3]
        images = torch.stack([draw_pose(opt,image,rot,size=20,width=2) for image,rot in zip(images,rots)],dim=0)
    images = images.cpu().permute(0,2,3,1).numpy() # [B,H,W,3]
    for i,img in zip(idx,images):
        fname = "{}/{}/{}_{}.png".format(opt.output_path,folder,i,name)
        img = Image.fromarray((img*255).astype(np.uint8)).convert('RGB')
        img.save(fname)

def get_heatmap(opt,gray,cmap): # [N,H,W]
    color = plt.get_cmap(cmap)(gray.numpy())
    color = torch.from_numpy(color[...,:3]).permute(0,3,1,2).float() # [N,3,H,W]
    return color

def dump_meshes(opt,idx,name,meshes,folder='dump'):
    for i,mesh in zip(idx,meshes):
        fname = "{}/{}/{}_{}.ply".format(opt.output_path,folder,i,name)
        try:
            mesh.export(fname)
        except:
            print('Mesh is empty!')

def dump_pointclouds_compare(opt,idx,name,preds,gts,folder='dump'):
    for i in range(len(idx)):
        pred = preds[i].cpu().numpy()   # [N,3]
        gt = gts[i].cpu().numpy()   # [N,3]
        color_pred = np.zeros(pred.shape).astype(np.uint8)
        color_pred[:,0] = 255
        color_gt = np.zeros(gt.shape).astype(np.uint8)
        color_gt[:,1] = 255
        pc_vertices = np.vstack([pred,gt])
        colors = np.vstack([color_pred,color_gt])
        pc_color = trimesh.points.PointCloud(vertices=pc_vertices,colors=colors)
        fname = "{}/{}/{}_{}.ply".format(opt.output_path,folder,idx[i],name)
        pc_color.export(fname)

@torch.no_grad()
def draw_pose(opt,image,rot_mtrx,size=15,width=1):
    # rot_mtrx: [3, 4]
    mode = "RGBA" if image.shape[0]==4 else "RGB"
    image_pil = torchvision_F.to_pil_image(image.cpu()).convert("RGBA")
    draw_pil = PIL.Image.new("RGBA",image_pil.size,(0,0,0,0))
    draw = PIL.ImageDraw.Draw(draw_pil)
    center = (size,size)
    # first column of rotation matrix is the rotated vector of [1, 0, 0]'
    # second column of rotation matrix is the rotated vector of [0, 1, 0]'
    # third column of rotation matrix is the rotated vector of [0, 0, 1]'
    # then always take the first two element of each column is a projection to the 2D plane for visualization
    endpoint = [(size+size*p[0],size+size*p[1]) for p in rot_mtrx.t()]
    draw.line([center,endpoint[0]],fill=(255,0,0),width=width)
    draw.line([center,endpoint[1]],fill=(0,255,0),width=width)
    draw.line([center,endpoint[2]],fill=(0,0,255),width=width)
    image_pil.alpha_composite(draw_pil)
    image_drawn = torchvision_F.to_tensor(image_pil.convert(mode))
    return image_drawn
