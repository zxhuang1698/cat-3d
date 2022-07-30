import numpy as np
import torch
import torch.nn.functional as torch_F

class Pose():
    # a pose class with util methods
    def __call__(self,R=None,t=None):
        assert(R is not None or t is not None)
        if R is None:
            if not isinstance(t,torch.Tensor): t = torch.tensor(t)
            R = torch.eye(3,device=t.device).repeat(*t.shape[:-1],1,1)
        elif t is None:
            if not isinstance(R,torch.Tensor): R = torch.tensor(R)
            t = torch.zeros(R.shape[:-1],device=R.device)
        else:
            if not isinstance(R,torch.Tensor): R = torch.tensor(R)
            if not isinstance(t,torch.Tensor): t = torch.tensor(t)
        assert(R.shape[:-1]==t.shape and R.shape[-2:]==(3,3))
        R = R.float()
        t = t.float()
        pose = torch.cat([R,t[...,None]],dim=-1) # [...,3,4]
        assert(pose.shape[-2:]==(3,4))
        return pose

    def invert(self,pose,use_inverse=False):
        R,t = pose[...,:3],pose[...,3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[...,0]
        pose_inv = self(R=R_inv,t=t_inv)
        return pose_inv

    def compose(self,pose_list):
        # pose_new(x) = poseN(...(pose2(pose1(x)))...)
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new,pose)
        return pose_new

    def compose_pair(self,pose_a,pose_b):
        # pose_new(x) = pose_b(pose_a(x))
        R_a,t_a = pose_a[...,:3],pose_a[...,3:]
        R_b,t_b = pose_b[...,:3],pose_b[...,3:]
        R_new = R_b@R_a
        t_new = (R_b@t_a+t_b)[...,0]
        pose_new = self(R=R_new,t=t_new)
        return pose_new

pose = Pose()

def to_hom(X):
    X_hom = torch.cat([X,torch.ones_like(X[...,:1])],dim=-1)
    return X_hom

def world2cam(X,pose): # [B,N,3]
    X_hom = to_hom(X)
    return X_hom@pose.transpose(-1,-2)
def cam2img(X,cam_intr):
    return X@cam_intr.transpose(-1,-2)
def img2cam(X,cam_intr):
    return X@cam_intr.inverse().transpose(-1,-2)
def cam2world(X,pose):
    X_hom = to_hom(X)
    pose_inv = Pose().invert(pose)
    # pose_inv.transpose(-1,-2): [B, 4, 3]
    # originally, p_cam' = [R|t] @ [p_world, 1]'
    # therefore, p_world = [p_cam, 1] @ inv([R|t])'
    return X_hom@pose_inv.transpose(-1,-2)

def euler2mat(angle,trig_x=None,trig_y=None,trig_z=None):
    # Convert euler angles to rotation matrix. y->x->z
    # Reference: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/196bb89f8d8e4d74e34bb0ee75d0763fc1fda3f5/inverse_warp.py#L72
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
    zeros = torch.zeros(z.shape).to(z.device)
    ones = torch.ones(z.shape).to(z.device)
    # generate rot matrix for z
    if trig_z is None:
        cosz = torch.cos(z)
        sinz = torch.sin(z)
    else:
        cosz = trig_z[:, 0]
        sinz = trig_z[:, 1]
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)
    # generate rot matrix for y
    if trig_y is None:
        cosy = torch.cos(y)
        siny = torch.sin(y)
    else:
        cosy = trig_y[:, 0]
        siny = trig_y[:, 1]
    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)
    # generate rot matrix for x
    if trig_x is None:
        cosx = torch.cos(x)
        sinx = torch.sin(x)
    else:
        cosx = trig_x[:, 0]
        sinx = trig_x[:, 1]
    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)
    # combine the rotation
    rotMat = zmat @ xmat @ ymat
    return rotMat

def azim_to_rotation_matrix(azim, representation='rad'):
    """Azim is angle with vector +X, rotated in XZ plane"""
    if representation == 'rad':
        # [B, ]
        cos, sin = torch.cos(azim), torch.sin(azim)
    elif representation == 'angle':
        # [B, ]
        azim = azim * np.pi / 180
        cos, sin = torch.cos(azim), torch.sin(azim)
    elif representation == 'trig':
        # [B, 2]
        cos, sin = azim[:, 0], azim[:, 1]
    R = torch.eye(3, device=azim.device)[None].repeat(len(azim), 1, 1)
    zeros = torch.zeros(len(azim), device=azim.device)
    R[:, 0, :] = torch.stack([cos, zeros, sin], dim=-1)
    R[:, 2, :] = torch.stack([-sin, zeros, cos], dim=-1)
    return R

def elev_to_rotation_matrix(elev, representation='rad'):
    """Angle with vector +Z in YZ plane"""
    if representation == 'rad':
        # [B, ]
        cos, sin = torch.cos(elev), torch.sin(elev)
    elif representation == 'angle':
        # [B, ]
        elev = elev * np.pi / 180
        cos, sin = torch.cos(elev), torch.sin(elev)
    elif representation == 'trig':
        # [B, 2]
        cos, sin = elev[:, 0], elev[:, 1]
    R = torch.eye(3, device=elev.device)[None].repeat(len(elev), 1, 1)
    R[:, 1, 1:] = torch.stack([cos, sin], dim=-1)
    R[:, 2, 1:] = torch.stack([-sin, cos], dim=-1)
    return R

def roll_to_rotation_matrix(roll, representation='rad'):
    """Angle with vector +X in XY plane"""
    if representation == 'rad':
        # [B, ]
        cos, sin = torch.cos(roll), torch.sin(roll)
    elif representation == 'angle':
        # [B, ]
        roll = roll * np.pi / 180
        cos, sin = torch.cos(roll), torch.sin(roll)
    elif representation == 'trig':
        # [B, 2]
        cos, sin = roll[:, 0], roll[:, 1]
    R = torch.eye(3, device=roll.device)[None].repeat(len(roll), 1, 1)
    R[:, 0, :2] = torch.stack([cos, sin], dim=-1)
    R[:, 1, :2] = torch.stack([-sin, cos], dim=-1)
    return R

# only for one axis
def angle_to_rotation_matrix(a,axis):
    roll = dict(X=1,Y=2,Z=0)[axis]
    O = torch.zeros_like(a)
    I = torch.ones_like(a)
    M = torch.stack([torch.stack([a.cos(),-a.sin(),O],dim=-1),
                     torch.stack([a.sin(),a.cos(),O],dim=-1),
                     torch.stack([O,O,I],dim=-1)],dim=-2)
    M = M.roll((roll,roll),dims=(-2,-1))
    return M

# only for one axis
def trig_to_rotation_matrix(trig,axis):
    cos_ = trig[:,0]
    sin_ = trig[:,1]
    roll = dict(X=1,Y=2,Z=0)[axis]
    O = torch.zeros_like(cos_)
    I = torch.ones_like(cos_)
    M = torch.stack([torch.stack([cos_,-sin_,O],dim=-1),
                     torch.stack([sin_,cos_,O],dim=-1),
                     torch.stack([O,O,I],dim=-1)],dim=-2)
    M = M.roll((roll,roll),dims=(-2,-1))
    return M

def get_camera_grid(opt,batch_size,device,intr=None):
    # compute image coordinate grid
    if opt.camera.model=="perspective":
        y_range = torch.arange(opt.H,dtype=torch.float32,device=device).add_(0.5)
        x_range = torch.arange(opt.W,dtype=torch.float32,device=device).add_(0.5)
        Y,X = torch.meshgrid(y_range,x_range) # [H,W]
        xy_grid = torch.stack([X,Y],dim=-1).view(-1,2) # [HW,2]
    elif opt.camera.model=="orthographic":
        assert(opt.H==opt.W)
        y_range = torch.linspace(-1,1,opt.H,device=device)
        x_range = torch.linspace(-1,1,opt.W,device=device)
        Y,X = torch.meshgrid(y_range,x_range) # [H,W]
        xy_grid = torch.stack([X,Y],dim=-1).view(-1,2) # [HW,2]
    xy_grid = xy_grid.repeat(batch_size,1,1) # [B,HW,2]
    if opt.camera.model=="perspective":
        grid_3D = img2cam(to_hom(xy_grid),intr) # [B,HW,3]
    elif opt.camera.model=="orthographic":
        grid_3D = to_hom(xy_grid) # [B,HW,3]
    return xy_grid,grid_3D

def get_center_and_ray(opt,pose,intr=None,offset=None,device=None): # [HW,2]
    if device is None: device = opt.device
    batch_size = len(pose)
    # grid 3D is the 3D location of the 2D pixels (on image plane, d=1)
    # under camera frame
    xy_grid,grid_3D = get_camera_grid(opt,batch_size,device,intr=intr) # [B,HW,3]
    # compute center and ray
    if opt.camera.model=="perspective":
        if offset is not None:
            grid_3D[...,:2] += offset
        # camera pose, [0, 0, 0] under camera frame
        center_3D = torch.zeros(batch_size,1,3,device=xy_grid.device) # [B,1,3]
    elif opt.camera.model=="orthographic":
        # different ray has different camera center
        center_3D = torch.cat([xy_grid,torch.zeros_like(xy_grid[...,:1])],dim=-1) # [B,HW,3]
    # transform from camera to world coordinates
    grid_3D = cam2world(grid_3D,pose) # [B,HW,3]
    center_3D = cam2world(center_3D,pose) # [B,HW,3]
    ray = grid_3D-center_3D # [B,HW,3]
    return center_3D,ray

def get_3D_points_from_depth(opt,center,ray,depth,multi_samples=False):
    if multi_samples: center,ray = center[:,:,None],ray[:,:,None]
    # x = c+dv
    points_3D = center+ray*depth # [B,HW,3]/[B,HW,N,3]/[N,3]
    return points_3D

def get_depth_from_3D_points(opt,center,ray,points_3D):
    # d = ||x-c||/||v|| (x-c and v should be in same direction)
    depth = (points_3D-center).norm(dim=-1,keepdim=True)/ray.norm(dim=-1,keepdim=True) # [B,HW,1]
    return depth