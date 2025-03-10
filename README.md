# README - GAN Training on CelebA Dataset

## Dataset Preprocessing

1. **Dataset Location:** The CelebA dataset images are stored in the directory:
   ```
   /kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/
   ```
2. **Loading Images:** The first 10,000 images are loaded and processed.
3. **Cropping:** Since the original images are 178x208, they are center-cropped to a square shape (178x178) to maintain aspect ratio.
4. **Resizing:** The cropped images are resized to 128x128 pixels.
5. **Normalization:** The pixel values are scaled between 0 and 1.

## Model Training

### Generator Architecture
- Input: A latent vector of size 32.
- Fully connected layer reshaped into 16x16x128.
- Several transposed convolution layers progressively increase the resolution.
- Final layer uses a `tanh` activation function to generate 128x128 RGB images.

### Discriminator Architecture
- Input: 128x128 RGB images.
- Several convolutional layers extract features.
- Flattening and dropout for regularization.
- Final layer with `sigmoid` activation for binary classification (real vs. generated).
- RMSprop optimizer with a learning rate of 0.0001 and gradient clipping.

### Training Process
1. **Hyperparameters:**
   - Iterations: 20,000
   - Batch Size: 16
2. **Training Steps:**
   - Generate images from random latent vectors.
   - Train discriminator on real and generated images.
   - Train the GAN (generator) to fool the discriminator.
   - Save model weights (`gan.h5`) every 50 iterations.
   - Save generated images for visualization.

## Model Testing
- The trained generator can create new face images by inputting random latent vectors.
- `gan.h5` can be loaded to generate samples.

## Expected Outputs
- During training, losses for both the discriminator and adversary (generator) are plotted.
- Generated images improve in quality as training progresses.
- A GIF (`training_visual.gif`) is created from generated images to visualize training progress.

## Visualization
- Generated images are saved in the `res2` directory.
- Training loss plots for the discriminator and generator are provided.

## Post-Processing
- Generated images are compiled into a GIF.
- The directory `res2` is deleted after saving the GIF.

## Running the Model
To train the model, execute the script in a Python environment with TensorFlow, Keras, NumPy, PIL, Matplotlib, and ImageIO installed.

