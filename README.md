<div align="center">
<h1>Large-Scale-ViT-SLAM</h1>

</div>

## Overview

This project originates from a master's thesis at the Technical University of Munich. We present a SLAM (Simultaneous Localization and Mapping) pipeline that leverages the feed-forward reconstruction model <a href="https://github.com/facebookresearch/vggt">VGGT</a>. Our approach addresses the challenge of achieving efficient feature-level alignment in large-scale environments without solely relying on loop closure for drift mitigation. The pipeline processes image sequences into 3D outputs, including point maps, poses, and depths.

<p align="center"><img src="assets/overview_general.png" alt="Large-Scale-ViT-SLAM" width="95%"/></p>
  <p align="center"><strong><em>Overview of a general chunk-and-align pipeline</em></strong></p>

To process arbitrary amounts of images and circumvent the GPU memory limitations inherent in VGGT, we adopt the chunk-and-align paradigm. An input image sequence of variable length is split into overlapping segments, called chunks, which are subsequently processed by VGGT. The chunks are aligned into a joint coordinate frame via optimization over correspondences between overlapping frames.

<p align="center"><img src="assets/overview_ours.png" alt="Large-Scale-ViT-SLAM" width="95%"/></p>
  <p align="center"><strong><em>Overview of our pipeline</em></strong></p>

Most concurrent work optimizes over VGGT’s decoded quantities, such as points or poses. In contrast, our approach computes inter-chunk alignment at the feature level, clearly setting our methodology apart. To achieve this, we introduce an alignment module. This module processes features from the VGGT encoder and produces per-frame similarity alignment. Subsequently, we apply this alignment to all VGGT decoder outputs.

Additionally, other feed-forward SLAM variants typically employ global optimization or loop closure to ensure global consistency and prevent drift accumulation. As a second objective, we examine replacing dedicated loop closure with a feature-level memory mechanic within our alignment module. We maintain a compressed summary of all past chunks, constructed by extracting and retaining key features that capture each chunk's unique characteristics. This summary is then injected into the current chunk's features during processing. By augmenting the current chunk's features with the past information before decoding the similarity transform, we aim to improve alignment for longer sequences.

In kilometer-scale driving scenes, we observe better inter-chunk alignment with a translational RPE (Relative Pose Error) improvement of around 40% compared to optimization-based alignment on output quantities. However, drift still accumulates significantly, with a 70% worse ATE (Absolute Trajectory Error) compared to methods using explicit loop closure. Our results show that while feature-level alignment provides benefits, the memory mechanic ultimately does not replace loop closure.

## Installation

First, clone this repository to your local machine:

```bash
git clone https://github.com/ruppelb/Large-Scale-ViT-SLAM
cd Large-Scale-ViT-SLAM
```
Create and activate a new conda environment:
```bash
conda create -n ls-vit-slam python=3.10
conda activate ls-vit-slam
```
Before installing our project and VGGT as a package, install PyTorch, torchvision, and PyTorch3D compatible with your CUDA version:
```bash
# pytorch 2.5.1 with cuda 12.1
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```
Clone and install VGGT via pip:
```bash
git clone https://github.com/facebookresearch/vggt.git
pip install -e ./vggt
```
Details on the license for VGGT can be found [here](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt).

Install our project via pip:
```bash
pip install -e .
```

## Quick Start

## Evaluation