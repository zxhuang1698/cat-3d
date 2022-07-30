# Cat-3D

Code repository for our paper "Planes vs. Chairs: Category-guided 3D shape learning without any 3D cues"

[[Project Page]](https://zixuanh.com/multiclass3D)  [[Paper]](https://arxiv.org/abs/2204.10235)

The repository currently includes training and evaluation code for ShapeNet-13 experiments.

## Dependencies

Please install the dependencies by running
```bash
conda env create --file requirements.yaml
cd external/chamfer3D
python3 setup.py install
cd ../..
```
You may need to modify `CUDA_HOME` accordingly for the compilation.

## Dataset

### ShapeNet

Please download the required data by running
```bash
cd data
bash download_data.sh
```
Make sure your ```data/NMR_Dataset``` folder is structured as follows:
```
├── 02691156/
|   ├── 1a04e3eab45ca15dd86060f189eb133/
|   |   ├── image/
|   |   |   ├── 0000.png
|   |   |   ├── ...
|   |   |   ├── 0023.png
|   |   ├── mask/
|   |   |   ├── 0000.png
|   |   |   ├── ...
|   |   |   ├── 0023.png
|   |   ├── cameras.npz
|   |   ├── pointcloud.npz
|   |   ├── pointcloud3.npz
|   ├── softras_train.lst
|   ├── ...
├── ...
```

## Training

Please first pretrain the model with a spherical SDF similar to SDF-SRN

```bash
python train.py --yaml=options/shapenet13.yaml --name=pretrain --pretrain
```

Then please run

```bash
python train.py --yaml=options/shapenet13.yaml
```

The training logs and visualizations are saved at the output directory.

## Evaluating

To evaluate the model for Chamfer Distance and F-score, Please run

```bash
python evaluate.py --yaml=options/shapenet13.yaml --eval.vox_res=128 --resume
```

The evaluation results are saved at the output directory.

## References

If you are using our code, please consider citing our paper.
```
@article{huang2022planes,
  title={Planes vs. Chairs: Category-guided 3D shape learning without any 3D cues},
  author={Huang, Zixuan and Stojanov, Stefan and Thai, Anh and Jampani, Varun and Rehg, James M},
  journal={arXiv preprint arXiv:2204.10235},
  year={2022}
}
```

This project contains a modified version of [SDF-SRN](https://github.com/chenhsuanlin/signed-distance-SRN) ([MIT License](https://github.com/chenhsuanlin/signed-distance-SRN/blob/main/LICENSE)) - Copyright (c) 2020 Chen-Hsuan Lin. Please also cite their great work if you use this codebase.
```
@inproceedings{lin2020sdfsrn,
  title={SDF-SRN: Learning Signed Distance 3D Object Reconstruction from Static Images},
  author={Lin, Chen-Hsuan and Wang, Chaoyang and Lucey, Simon},
  booktitle={Advances in Neural Information Processing Systems ({NeurIPS})},
  year={2020}
}
```
