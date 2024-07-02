# Image Inpainting with CNN

## Description

In this project, a convolutional neural network (CNN) is designed and trained for image inpainting using the CIFAR-100 dataset. Inpainting refers to the process of filling in missing parts of an image. The dataset consists of RGB real-life images with a resolution of 28x28 pixels, which have been preprocessed for this task.

## Requirements

- **PyTorch**: Version 1.9.0 or later
- **torchvision**: Included in PyTorch distributions
- **PIL**: Python Imaging Library
- **NumPy**: Version 1.19.5 or later
- **Matplotlib**: Version 3.4.3 or later

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ilkerozgen/ImageInpainting.git
   cd repository
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
> Ensure CUDA-enabled GPU for faster training or use Colab environment.

## Dataset

The CIFAR-100 dataset is used, downloadable from [this link](https://drive.google.com/file/d/1KiymtjUjuEJjUTO_qB9ifpLC_UvJEhoL/view?usp=share_link). Place the dataset in the appropriate directory as specified in the project.

## Model Architecture

The project implements a convolutional autoencoder for the inpainting task. The architecture details are as follows:

### Encoder
- **Conv2d (3, 16)**: Kernel size 3x3, stride 1, padding 1
- **BatchNorm2d**: Applied after each convolutional layer
- **LeakyReLU (0.2)**: Activation function
- **MaxPool2d**: Used for downsampling

### Decoder
- **ConvTranspose2d**: Used for upsampling
- **BatchNorm2d**: Applied after each transposed convolutional layer
- **LeakyReLU (0.2)**: Activation function
- **ConvTranspose2d (64, 32)**, **ConvTranspose2d (32, 16)**, **ConvTranspose2d (16, 3)**

The final layer uses Tanh activation to ensure outputs are in the range [-1, 1].

## Training

### Training Loop

The training loop implements the following steps:
- Mask the input images to simulate inpainting.
- Forward pass through the network.
- Compute Mean Squared Error (MSE) loss between the reconstructed image and the original.
- Backpropagate and optimize using the Adam optimizer.

### Evaluation

The evaluation function calculates the Mean Squared Error (MSE) on the test dataset to assess model performance.

## Results

Refer to the [inpaint.ipynb](inpaint.ipynb) file for detailed results, analysis, and visualizations.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contribution

Contributions are welcome! Please fork this repository and submit pull requests for any improvements or bug fixes.
