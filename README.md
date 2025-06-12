# Diff-UNet

Diff-UNet: A Diffusion Embedded Network for Volumetric Segmentation.

We design the Diff-UNet applying diffusion model to solve the 3D medical image segmentation problem.

Diff-UNet achieves more accuracy in multiple segmentation tasks compared with other 3D segmentation methods.

![](/imgs/framework.png)

## dataset

We release the codes which support the training and testing process of the datasets BraTS2021.

BraTS2021(4 modalities and 3 segmentation targets): <https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1>

Once the data is downloaded, you can begin the training process. Please see the dir of BraTS2021.

## setup

```bash
cd BraTS2021
# cu111 for RTX 3060Ti
conda create -n diff-unet python=3.8 -y
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# cu121 for RTX 5090
conda create -n diff-unet python=3.11 -y
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
# install other packages
pip install -r requirements.txt
conda install conda-forge::simpleitk -y
```

## training

Training use Pytoch DDP mode with four GPUs. You also can modify the parameters to use one GPU to train(refer to the train.py).

```bash
python train.py
```

## testing

When you have trained a model, please modify the model path, then run the code.

```bash
python test.py
```
