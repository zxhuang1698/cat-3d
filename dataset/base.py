import numpy as np
import torch
import torchvision
import PIL
import vigra

from utils.util_run import EasyDict as edict


class Dataset(torch.utils.data.Dataset):

    def __init__(self,opt,split):
        super().__init__()
        self.opt = opt
        self.split = split
        self.augment = split=="train" and opt.data.augment

    def setup_loader(self,opt,shuffle=False,drop_last=True):
        loader = torch.utils.data.DataLoader(self,
            batch_size=opt.batch_size,
            num_workers=opt.data.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        return loader

    def get_list(self,opt):
        raise NotImplementedError

    def __getitem__(self,idx):
        raise NotImplementedError

    def get_image(self,opt,idx):
        raise NotImplementedError

    def setup_color_jitter(self,opt):
        brightness = opt.data.augment.brightness or 0.
        contrast = opt.data.augment.contrast or 0.
        saturation = opt.data.augment.saturation or 0.
        hue = opt.data.augment.hue or 0.
        color_jitter = torchvision.transforms.ColorJitter(
            brightness=(1-brightness,1+brightness),
            contrast=(1-contrast,1+contrast),
            saturation=(1-saturation,1+saturation),
            hue=(-hue,hue),
        )
        return color_jitter

    def apply_color_jitter(self,opt,image,color_jitter):
        mode = image.mode
        if mode!="L":
            chan = image.split()
            rgb = PIL.Image.merge("RGB",chan[:3])
            rgb = color_jitter(rgb)
            rgb_chan = rgb.split()
            image = PIL.Image.merge(mode,rgb_chan+chan[3:])
        return image

    def compute_dist_transform(self,opt,mask,intr=None):
        assert(mask.shape[0]==1)
        mask = mask[0] # [H,W]
        mask_binary = mask!=0 # make sure only 0/1
        # use boundaryDistanceTransform instead of distanceTransform (for 0.5 pixel precision)
        bdt = vigra.filters.boundaryDistanceTransform(mask_binary.float().numpy())
        if opt.camera.model=="orthographic":
            # assume square images for now....
            assert(opt.H==opt.W)
            bdt *= 2./float(opt.H) # max distance from H (W) to 2
        elif opt.camera.model=="perspective":
            # assume same focal length for x/y for now....
            assert(intr[0,0]==intr[1,1])
            bdt /= float(intr[0,0])
        bdt = torch.from_numpy(bdt)[None]
        return bdt

    def __len__(self):
        return len(self.list)
