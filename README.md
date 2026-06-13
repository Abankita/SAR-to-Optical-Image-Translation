# SAR-to-Optical Image Translation

A deep learning project for translating Synthetic Aperture Radar (SAR) images to optical (multispectral) images using conditional Generative Adversarial Networks (cGANs) with Pix2Pix architecture.

## Overview

This project implements image-to-image translation from SAR imagery to optical (Sentinel-2) imagery. SAR images are radar-based measurements that can penetrate clouds and operate regardless of lighting conditions, while optical images provide visible-spectrum data. This translation enables generation of optical-like images from SAR data, useful for cloud-free image reconstruction, data augmentation, and multi-modal analysis.

### Key Features

- **Pix2Pix Architecture**: Conditional GAN with U-Net generator and PatchGAN discriminator
- **SAR-to-Optical Translation**: Converts Sentinel-1 SAR images to Sentinel-2 optical equivalents
- **Pre-trained Models**: Includes trained generator and discriminator weights
- **PyTorch Implementation**: Built with PyTorch for high performance and flexibility
- **Urban Scene Dataset**: Trained on urban scene pairs

## Project Structure

```
SAR-to-Optical-Image-Translation/
├── main.ipynb                      # Main training and inference notebook
├── generator_pix2pix.pth          # Pre-trained generator model weights
├── discriminator_pix2pix.pth      # Pre-trained discriminator model weights
├── urban/                          # Dataset directory
│   ├── s1/                         # SAR images (Sentinel-1)
│   └── s2/                         # Optical images (Sentinel-2)
└── README.md                       # This file
```

## Methodology

### Pix2Pix (cGAN)

The project uses **Pix2Pix** (Image-to-Image Translation with Conditional Adversarial Networks), which consists of:

- **Generator**: U-Net-based architecture with skip connections for preserving structural information
- **Discriminator**: PatchGAN discriminator that classifies patches of the image
- **Loss Function**: L1 loss and Adversarial loss to ensure realistic outputs

The model is trained to map SAR images to optically similar images while maintaining spatial correspondence.

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.9+ (with CUDA support recommended)
- torchvision
- Pillow
- NumPy
- Matplotlib
- scikit-image

### Setup

```bash
# Clone repository
git clone <repository-url>
cd SAR-to-Optical-Image-Translation

# Install dependencies
pip install torch torchvision pillow numpy matplotlib scikit-image

# Optional: For GPU support (NVIDIA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Dataset

The project uses paired SAR and optical images organized in the following structure:

```
urban/
├── s1/          # Sentinel-1 SAR images (PNG format)
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── s2/          # Sentinel-2 Optical images (PNG format)
    ├── image_001.png
    ├── image_002.png
    └── ...
```

**Dataset Link**: [Sentinel-1&2 Image Pairs (SAR & Optical)](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain)

The dataset used contain registered image pairs where:
- **s1/** contains SAR bands from Sentinel-1 (VV and VH polarizations)
- **s2/** contains optical bands from Sentinel-2 (RGB or multispectral)
- Images are spatially aligned and have matching filenames

## Usage

### Training

Open and run `main.ipynb` to train the model.