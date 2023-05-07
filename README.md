# Stable-Diffusion-Image-to-Prompts
<img src="promptMeme.jpeg" width="400" height="500"/>
Stable-Diffusion Image-to-Prompt is a cutting-edge machine learning project that predicts text prompts from given images. By leveraging the power of the ViT-GPT2, CLIP + Prompt, and ViT models, this project achieved a 0.54 (Top 20%) score in a Kaggle competition.

Kaggle Link: https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts

## Quick Intro
Images were generated from the prompts using Stable Diffusion 2.0 (768-v-ema.ckpt) and were generated with 50 steps at 768x768 px and then downsized to 512x512 for the competition dataset.

Generate image: https://github.com/Stability-AI/stablediffusion/blob/main/scripts/txt2img.py

Online demo: https://huggingface.co/stabilityai/stable-diffusion-2?text=boy+with+hair%2C+long+hairNotebook

Generate code: https://www.kaggle.com/code/wuwenmin/failed-to-reproduce-kaggle-dataset/

## Features
 - Predict text prompts from images
 - Utilizes ViT-GPT2, CLIP + Prompt, and ViT models


## Getting Started

### Prerequisites
- [Python 3.8+](https://www.python.org/downloads/)
- [PyTorch 1.9+](https://pytorch.org/get-started/locally/)
- [torchvision 0.10+](https://pypi.org/project/torchvision/)
- [torchtext 0.9+](https://pypi.org/project/torchtext/)
- [Hugging Face Transformers 4.9+](https://huggingface.co/transformers/)

### Baseline Used:
1. Vit-GPT2
2. BLIP + CLIP
3. ViT

### Data cleaning and validation set division
【14w cleaning data】：https://www.kaggle.com/code/shoheiazuma/diffusiondb-data-cleansing/
【29w cleaning data】：https://www.kaggle.com/code/finlay/clean-diffusiondb-and-save-target/

#### Model Training Data Augmentation
https://pytorch.org/vision/stable/transforms.html

## Follow-up Ideas:
[Generate Dataset] Learn the use of Stable Diffusion
    Can generate new images through Stable Diffusion
    Learn the prompt usage rules of Stable Diffusion
【Train VIT model, open source 3epoch】Train the model from image to text vector
    Comparing the VIT model with other models
    Comparing different loss functions
    Contrastive Data Augmentation
    Train multiple models with cross-validation
[Model Prediction] Load the trained model
    Multiple Model Prediction
    Data Augmentation and Multiple Prediction
    After generating the result, the prediction size is 384 * 384, 512 * 512