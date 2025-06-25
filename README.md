# Appleencoder: Unsupervised Deep Learning for Apple Images

This project provides a robust, modular pipeline for unsupervised and self-supervised deep learning on apple images using Keras/TensorFlow. It includes autoencoder-based reconstruction, denoising/inpainting, anomaly detection, latent space clustering, and visualization tools.

## Features
- **Robust Data Loading & Cleaning:** Handles corrupted images, normalizes, and validates input data.
- **Autoencoder Training:** Learns to reconstruct apple images, with training/validation loss plots and reconstruction visualization.
- **Variational Autoencoder (VAE):** Learns a probabilistic latent space for generative modeling and anomaly detection.
- **Masked Image Inpainting:** Trains an autoencoder to reconstruct images with random patches masked out.
- **Latent Space Clustering:** Uses KMeans and t-SNE to visualize and cluster encoded features.
- **Anomaly Detection:** Uses One-Class SVM on latent features to score and visualize outliers.
- **Visualization:** Plots for training curves, reconstructions, latent space, and anomalies.

## File Structure
- `Appleencoder.py` — Main pipeline script (data loading, model training, evaluation, visualization)
- `apple/train/` — Training images (subfolders or flat)
- `apple/test/` — Test images

## Usage
1. **Install dependencies:**
   ```bash
   pip install tensorflow matplotlib scikit-learn pillow
   ```
2. **Prepare your data:**
   - Place training images in `apple/train/`
   - Place test images in `apple/test/`
3. **Run the script:**
   ```bash
   python Appleencoder.py
   ```

## Main Pipeline Steps
1. **Data Loading:**
   - Loads and validates images from `apple/train/` and `apple/test/`.
   - Splits training data into train/validation sets.
2. **Autoencoder Training:**
   - Trains a convolutional autoencoder to reconstruct images.
   - Plots training/validation loss and MAE.
   - Visualizes original vs. reconstructed images.
3. **Variational Autoencoder (VAE):**
   - Trains a VAE for generative modeling and anomaly detection.
   - Visualizes VAE reconstructions.
4. **Latent Space Clustering:**
   - Extracts latent features from the encoder.
   - Applies KMeans clustering and t-SNE for visualization.
5. **Masked Image Inpainting:**
   - Trains an autoencoder to reconstruct images with random patches masked out.
   - Visualizes original, masked, and inpainted images.
6. **Anomaly Detection:**
   - Fits a One-Class SVM on latent features.
   - Scores and visualizes the most anomalous test images.

## Customization
- Adjust `img_height`, `img_width`, `batch_size`, and `epochs` at the top of the script.
- Replace the autoencoder architecture with your own for experimentation.
- Add more advanced unsupervised/self-supervised techniques as needed.

## Notes
- The script is robust to corrupted or invalid images.
- All visualizations are saved/shown using matplotlib.
- The pipeline is modular: you can comment/uncomment blocks to run only the parts you need.

## Requirements
- Python 3.7+
- TensorFlow 2.x
- matplotlib
- scikit-learn
- Pillow



**Contact:** For questions or contributions, please open an issue or pull request on GitHub.
