# MCD-UNet: A Multi-modal Conditional Diffusion UNet for 3D Medical Image Segmentation

MCD-UNet is a novel, diffusion-based deep learning framework designed for 3D volumetric medical image segmentation. It introduces a sophisticated conditioning mechanism to leverage multi-modal MRI data, achieving state-of-the-art accuracy on challenging segmentation benchmarks like BraTS 2021.

## Framework

The core of MCD-UNet is a conditional denoising diffusion probabilistic model (DDPM). The model is trained to reverse a forward process that incrementally adds noise to a segmentation mask, thereby learning to generate clean masks from a noise distribution. The model's strength lies in two key innovations:

### 1. Multi-Modal Fusion for Precision Segmentation

The model foundationally leverages the complementary information from four different MRI modalities (T1, T1ce, T2, and Flair). These modalities are fused by stacking them into a single multi-channel input tensor. This fusion provides a comprehensive anatomical view, enabling the model to distinguish between different tissue types and tumor sub-regions with high precision, which is critical for accurate brain tumor segmentation.

### 2. Sophisticated Conditional Guidance

To effectively guide the segmentation process, MCD-UNet conditions the diffusion model on multi-modal MRI scans (T1, T1ce, T2, Flair) using a unique dual strategy:

- **Input-Level Concatenation:** The multi-modal MRI scans are directly concatenated with the noisy segmentation map at the input layer of the denoising U-Net. This provides the model with direct, low-level anatomical context.
- **Deep Feature Injection:** A separate, dedicated U-Net encoder first extracts multi-scale feature representations from the input MRI scans. These rich, hierarchical features are then injected as embeddings into the corresponding layers of the main denoising U-Net. This provides high-level semantic guidance throughout the generation process.

This dual-conditioning approach ensures that the model leverages the rich information from the multi-modal scans at every level of abstraction, from raw voxel intensities to complex anatomical features.

## Dataset

The codebase supports training and testing on the **BraTS 2021** dataset.

- **Link:** [BraTS 2021 Task 1 on Kaggle](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)
- **Details:** The dataset consists of 3D multi-modal MRI scans (4 modalities) and corresponding ground-truth segmentations for brain tumors (3 sub-regions).

Please download the dataset and place it in the appropriate directory to begin training. Refer to `BraTS2021/dataset/brats_data_utils_multi_label.py` for data handling details.

## Setup

First, create a Conda environment. We recommend using Python 3.9+.

```bash
# It is recommended to use a recent version of Python
conda create -n mcd-unet python=3.11 -y
conda activate mcd-unet
```

Next, install PyTorch with a compatible CUDA version. The following commands are examples for recent CUDA versions. Please check the [PyTorch website](https://pytorch.org/get-started/locally/) for the command that matches your system's CUDA toolkit.

```bash
# Example of CUDA 11.1 for RTX 3060
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Example of CUDA 12.8 for RTX 5090
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Finally, install the remaining packages:

```bash
# Navigate to the project directory
cd BraTS2021
# Install other packages
pip install -r requirements.txt
conda install conda-forge::simpleitk -y
```

## Usage

### Training

The model is trained using PyTorch DDP (Distributed Data Parallel) mode, by default configured for multiple GPUs. You can easily adapt the training script `train.py` for single-GPU training by modifying the script parameters.

To start the training process, run:

```bash
python train.py
```

### Testing

Once you have a trained model checkpoint, update the model path in `test.py` and run the evaluation script:

```bash
python test.py
```
