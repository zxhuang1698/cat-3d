import numpy as np
import torch
import torchvision.transforms.functional as torchvision_F
import PIL
import pickle
import utils.util_cam as util_cam

from . import base

class Dataset(base.Dataset):

    def __init__(self,opt,split="train"):
        super().__init__(opt,split)
        assert(opt.H==64 and opt.W==64)
        assert(opt.camera.model=="perspective")
        self.cat_id_all = dict(
            bench="02828884",
            cabinet="02933112",
            car="02958343",
            chair="03001627",
            display="03211117",
            lamp="03636649",
            plane="02691156",
            rifle="04090263",
            sofa="04256520",
            speaker="03691459",
            table="04379243",
            telephone="04401088",
            vessel="04530566",    
        )
        if opt.data.num_classes is None:
            assert opt.data.shapenet.cat is None
            opt.data.num_classes = len(self.cat_id_all.keys())

        # dict map categories to id
        self.cat2label = {}
        accum_idx = 0
        self.cat_id = list(self.cat_id_all.values()) if opt.data.shapenet.cat is None else \
                      [v for k,v in self.cat_id_all.items() if k in opt.data.shapenet.cat.split(",")]
        for cat in self.cat_id:
            self.cat2label[cat] = accum_idx
            accum_idx += 1

        # list map id to category names
        self.label2cat = []
        for cat in self.cat_id:
            key = next(key for key, value in self.cat_id_all.items() if value == cat)
            self.label2cat.append(key) 
        self.path = "data/NMR_Dataset"

        # get the list of cads
        self.list_cads = self.get_list(opt,split)
        # get the list of cads by viewpoint, only 1 fixed view per cad model is used
        self.list = self.get_list_with_viewpoints(opt,split)

    # read the list file, return a list of tuple, (category, model_name, model_index)
    def get_list(self,opt,split):
        cads = []
        for c in self.cat_id:
            list_fname = "data/NMR_Dataset/{}/softras_{}.lst".format(c,split)
            cads += [(c,m,i) for i,m in enumerate(open(list_fname).read().splitlines())]
        return cads

    def get_list_with_viewpoints(self,opt,split):
        self.num_views = 1
        view = (opt.data.shapenet.train_view if split=="train" else opt.data.shapenet.test_view) or self.num_views
        with open("data/shapenet_view.pkl","rb") as file:
            view_idx = pickle.load(file)
        view_idx = { k:v for k,v in view_idx.items() if k in self.cat_id }
        # append view_index after the model_index
        samples = [(c,m,i,v) for c,m,i in self.list_cads for v in view_idx[c][split][m][:view]]
        return samples

    def __getitem__(self,idx):
        opt = self.opt
        sample = dict(
            idx=idx,
        )

        # load camera
        intr,pose = self.get_camera(opt,idx)    
        sample.update(
            pose_gt=pose,
            intr=intr,
        )

        # load images
        # get the jitter if needed
        aug = self.setup_color_jitter(opt) if self.augment else None
        image = self.get_image(opt,idx)
        cat_label = self.get_category(opt,idx)
        rgb,mask = self.preprocess_image(opt,image,aug=aug)
        dt = self.compute_dist_transform(opt,mask,intr=intr)
        sample.update(
            rgb_input_map=rgb,
            mask_input_map=mask,
            dt_input_map=dt,
            category_label=cat_label,
        )

        # vectorize images (and randomly sample)
        rgb = rgb.permute(1,2,0).view(opt.H*opt.W,3)
        mask = mask.permute(1,2,0).view(opt.H*opt.W,1)
        dt = dt.permute(1,2,0).view(opt.H*opt.W,1)
        sample.update(
            rgb_input=rgb,
            mask_input=mask,
            dt_input=dt,
        )

        # load GT point cloud (only for validation!)
        dpc = self.get_pointcloud(opt,idx)
        sample.update(dpc=dpc)
        return sample

    def get_image(self,opt,idx):
        c,m,i,v = self.list[idx]
        image_fname = "{0}/{1}/{2}/image/{3:04d}.png".format(self.path,c,m,v)
        mask_fname = "{0}/{1}/{2}/mask/{3:04d}.png".format(self.path,c,m,v)
        image = PIL.Image.open(image_fname).convert("RGB")
        mask = PIL.Image.open(mask_fname).split()[0]
        image = PIL.Image.merge("RGBA",list(image.split())+[mask])
        return image
    
    def get_category(self,opt,idx):
        c,_,_,_ = self.list[idx]
        label = int(self.cat2label[c])
        return label

    def preprocess_image(self,opt,image,aug=None):
        if aug is not None:
            image = self.apply_color_jitter(opt,image,aug.color_jitter)
        image = image.resize((opt.W,opt.H))
        image = torchvision_F.to_tensor(image)
        rgb,mask = image[:3],image[3:]
        if opt.data.bgcolor is not None and opt.data.masking:
            # replace background color using mask
            rgb = rgb*mask+opt.data.bgcolor*(1-mask)
        return rgb,mask

    def get_camera(self,opt,idx):
        c,m,i,v = self.list[idx]
        cam_fname = "{0}/{1}/{2}/cameras.npz".format(self.path,c,m)
        cam = np.load(cam_fname)
        focal = 1.8660254037844388
        intr = torch.tensor([[focal*opt.W,0,opt.W/2],
                             [0,focal*opt.H,opt.H/2],
                             [0,0,1]])
        extr = torch.from_numpy(cam["world_mat_{}".format(v)][:3]).float()
        pose = util_cam.pose(R=extr[:,:3],t=extr[:,3])
        return intr,pose

    def get_pointcloud(self,opt,idx):
        c,m,i,v = self.list[idx]
        pc_fname = "{0}/{1}/{2}/pointcloud3.npz".format(self.path,c,m)
        npz = np.load(pc_fname)
        dpc = dict(
            points=torch.from_numpy(npz["points"]).float(),
            normals=torch.from_numpy(npz["normals"]).float(),
        )
        return dpc

    def __len__(self):
        return len(self.list)