# Line Inpainting Autoencoder

This project implements a deep convolutional autoencoder using PyTorch to reconstruct masked line pixels from images. It is designed to learn how to fill in missing horizontal or vertical lines, effectively performing *image inpainting* on structured missing data.

---

##  Objective

The main goal of this model is to restore masked regions (specifically lines) in images, such as when parts of the image are occluded or corrupted. The model is trained to infer the original pixel values in these masked areas, using contextual information from the surrounding pixels.

---

## Model Architecture

The architecture is a symmetric **convolutional autoencoder** consisting of two main components:

###  Encoder

The encoder compresses the input image into a low-dimensional latent representation while preserving spatial features. It reduces the spatial resolution in steps:  
`256x256 → 128x128 → 64x64 → 32x32 → 16x16 → 8x8 → 4x4`.

It uses:
- 6 convolutional layers
- Batch normalization
- LeakyReLU activations
- Dropout (on select layers)

###  Decoder

The decoder upsamples the latent representation back to the original image resolution:  
`4x4 → 8x8 → 16x16 → 32x32 → 64x64 → 128x128 → 256x256`.

It uses:
- 6 transposed convolutional layers
- Batch normalization
- LeakyReLU activations
- Tanh activation in the final layer to normalize pixel values

---

##  Dependencies

- Python 3.x  
- PyTorch  
- NumPy  
- (Optional: OpenCV, matplotlib for visualization)

You can install the required dependencies via:

```bash
pip install torch torchvision
